#!/usr/bin/env python3
"""Train the internal BrainVLM-Midline tumor or WHO model."""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from midline_agent.diagnostic.brainvlm_midline.collate import collate_task_samples
from midline_agent.diagnostic.brainvlm_midline.config import CHECKPOINT_MANIFEST
from midline_agent.diagnostic.brainvlm_midline.datasets import (
    MixedTaskDataset,
    TumorDataset,
    WhoGradeDataset,
    load_training_records,
    upsample_records,
)
from midline_agent.diagnostic.brainvlm_midline.llama import create_llama
from midline_agent.diagnostic.brainvlm_midline.model import BrainVlmMidline
from midline_agent.diagnostic.brainvlm_midline.training import (
    BrainVlmTrainingConfig,
    TrainingContractError,
    WarmupCosineScheduler,
    build_optimizer,
    configure_trainable_parameters,
    load_training_checkpoint,
    save_training_checkpoint,
    train_epoch,
)
from midline_agent.diagnostic.brainvlm_midline.vision import create_biomedclip_vision
from midline_agent.diagnostic.checkpoints import load_brainvlm_checkpoint


def _load_config(path: Path) -> BrainVlmTrainingConfig:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingContractError("unable to read training configuration") from exc
    return BrainVlmTrainingConfig.from_mapping(value)


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint_path: Path
    rank: int
    world_size: int
    global_step: int
    optimizer_steps: int
    mean_loss: float


def _environment_integer(name: str) -> int:
    value = os.environ.get(name)
    try:
        parsed = int(value) if value is not None else None
    except ValueError as exc:
        raise TrainingContractError(f"{name} must be an integer") from exc
    if parsed is None or parsed < 0:
        raise TrainingContractError(f"{name} must be a non-negative integer")
    return parsed


def _resolve_device(
    config: BrainVlmTrainingConfig,
    *,
    local_rank: int,
    world_size: int,
) -> torch.device:
    if not torch.cuda.is_available():
        raise TrainingContractError("CUDA is not available")
    configured_index = int(config.device.split(":", maxsplit=1)[1])
    index = local_rank if world_size > 1 else configured_index
    if index >= torch.cuda.device_count():
        raise TrainingContractError("CUDA device index is unavailable")
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return device


def _distributed(config: BrainVlmTrainingConfig) -> _DistributedContext:
    if not config.distributed:
        return _DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=_resolve_device(config, local_rank=0, world_size=1),
            owns_process_group=False,
        )
    rank = _environment_integer("RANK")
    world_size = _environment_integer("WORLD_SIZE")
    local_rank = _environment_integer("LOCAL_RANK")
    if world_size != config.world_size:
        raise TrainingContractError("WORLD_SIZE does not match configured world size")
    if rank >= world_size or local_rank >= world_size:
        raise TrainingContractError("distributed rank is outside configured world size")
    device = _resolve_device(config, local_rank=local_rank, world_size=world_size)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method=config.dist_url,
            world_size=world_size,
            rank=rank,
        )
    return _DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        owns_process_group=owns_process_group,
    )


def _build_dataset(config: BrainVlmTrainingConfig):
    records = load_training_records(config.annotations, config.image_root)
    if config.task == "tumor":
        balanced = upsample_records(
            records,
            key=lambda item: item.tumor_label,
            ratio=config.upsample_ratio,
            seed=config.seed,
        )
        return TumorDataset(balanced, seed=config.seed, use_age=config.use_age)
    who_records = upsample_records(
        [item for item in records if item.tumor_label in {"<tumor_2>", "<tumor_4>", "<tumor_5>"}],
        key=lambda item: f"{item.tumor_label} {item.grade_label}",
        ratio=config.upsample_ratio,
        seed=config.seed,
    )
    who = WhoGradeDataset(who_records, seed=config.seed, use_age=config.use_age)
    tumor = TumorDataset(records, seed=config.seed + 1, use_age=config.use_age)
    length = max(5, math.ceil(len(who) / 0.8))
    return MixedTaskDataset(who, tumor, length=length, seed=config.seed)


def _build_model(config: BrainVlmTrainingConfig) -> BrainVlmMidline:
    language_model, tokenizer = create_llama(config.llama_model)
    vision = create_biomedclip_vision(config.biomedclip_pretrained)
    model = BrainVlmMidline(language_model, tokenizer, vision)
    configure_trainable_parameters(model)
    if config.initialization_checkpoint is not None:
        load_brainvlm_checkpoint(config.initialization_checkpoint, model)
    return model


def run_training(config: BrainVlmTrainingConfig) -> TrainingRunResult:
    context = _distributed(config)
    try:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        dataset = _build_dataset(config)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=config.seed,
            )
            if context.world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=collate_task_samples,
            pin_memory=context.device.type == "cuda",
        )
        model = _build_model(config).to(context.device)
        trainable_model = model
        if context.world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=(
                    [context.local_rank] if context.device.type == "cuda" else None
                ),
            )
        optimizer = build_optimizer(
            trainable_model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = math.ceil(
            config.iterations_per_epoch / config.gradient_accumulation_steps
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=max(1, steps_per_epoch * config.epochs),
            minimum_learning_rate=config.min_learning_rate,
            warmup_learning_rate=config.warmup_learning_rate,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=config.amp and context.device.type == "cuda"
        )
        start_epoch = 0
        global_step = 0
        if config.resume is not None:
            resumed = load_training_checkpoint(
                config.resume,
                trainable_model,
                optimizer,
                scaler=scaler,
                scheduler=scheduler,
                manifest=CHECKPOINT_MANIFEST,
            )
            start_epoch = resumed.start_epoch
            global_step = resumed.global_step
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = config.output_dir / "checkpoint_last.pth"
        final_result = None
        for epoch in range(start_epoch, config.epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            final_result = train_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                scaler=scaler,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_iterations=config.iterations_per_epoch,
                device=context.device,
                amp=config.amp,
            )
            global_step += final_result.optimizer_steps
            if context.rank == 0:
                save_training_checkpoint(
                    checkpoint,
                    trainable_model,
                    optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config.to_dict(),
                    manifest=CHECKPOINT_MANIFEST,
                )
            if context.world_size > 1:
                dist.barrier()
        if final_result is None:
            raise TrainingContractError("resume checkpoint has no remaining configured epochs")
        return TrainingRunResult(
            checkpoint_path=checkpoint,
            rank=context.rank,
            world_size=context.world_size,
            global_step=global_step,
            optimizer_steps=final_result.optimizer_steps,
            mean_loss=final_result.mean_loss,
        )
    finally:
        if context.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tumor", "who"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = _load_config(args.config)
    if args.task != config.task:
        raise TrainingContractError("command task does not match configuration task")
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
