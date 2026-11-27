#!/usr/bin/env python3
"""Train the internal BrainVLM-Midline tumor or WHO model."""

from __future__ import annotations

import argparse
import math
import os
import random
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


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


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


def run_training(config: BrainVlmTrainingConfig) -> Path:
    rank, world_size, local_rank = _distributed()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.manual_seed_all(config.seed)
    else:
        device = torch.device("cpu")
    dataset = _build_dataset(config)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_task_samples,
        pin_memory=device.type == "cuda",
    )
    model = _build_model(config).to(device)
    trainable_model = model
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    optimizer = build_optimizer(
        trainable_model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=max(1, steps_per_epoch * config.epochs),
        minimum_learning_rate=config.min_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    start_epoch = 0
    global_step = 0
    if config.resume is not None:
        resumed = load_training_checkpoint(
            config.resume,
            trainable_model,
            optimizer,
            scaler=scaler,
            manifest=CHECKPOINT_MANIFEST,
        )
        start_epoch = resumed.start_epoch
        global_step = resumed.global_step
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "checkpoint_last.pth"
    for epoch in range(start_epoch, config.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        result = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler=scaler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device=device,
            amp=config.amp,
        )
        global_step += result.optimizer_steps
        if rank == 0:
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
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()
    return checkpoint


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
