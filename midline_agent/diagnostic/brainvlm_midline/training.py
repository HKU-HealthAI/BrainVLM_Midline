"""Focused training, scheduling, checkpoint, and resume utilities."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from ..checkpoints import BrainVlmCheckpointError, load_brainvlm_checkpoint


TRAINING_FORMAT = "midline-agent-brainvlm-training"
TRAINING_FORMAT_VERSION = "1.0"
_REQUIRED_CONFIG_FIELDS = {
    "task",
    "annotations",
    "image_root",
    "llama_model",
    "biomedclip_pretrained",
    "output_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "min_learning_rate",
    "warmup_learning_rate",
    "weight_decay",
    "warmup_steps",
    "iterations_per_epoch",
    "gradient_accumulation_steps",
    "seed",
    "num_workers",
    "use_age",
    "upsample_ratio",
    "amp",
    "device",
}
_OPTIONAL_CONFIG_FIELDS = {
    "resume",
    "initialization_checkpoint",
    "distributed",
    "world_size",
    "dist_url",
}
_CUDA_DEVICE_PATTERN = re.compile(r"cuda:(0|[1-9][0-9]*)")


class TrainingContractError(ValueError):
    pass


def _positive_int(mapping: Mapping[str, Any], name: str, *, allow_zero: bool = False) -> int:
    value = mapping[name]
    lower = 0 if allow_zero else 1
    if type(value) is not int or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise TrainingContractError(f"{name.replace('_', ' ')} must be a {qualifier} integer")
    return value


def _finite_number(
    mapping: Mapping[str, Any],
    name: str,
    *,
    minimum: float,
    strict: bool,
) -> float:
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingContractError(f"{name.replace('_', ' ')} must be a finite number")
    number = float(value)
    invalid = number <= minimum if strict else number < minimum
    if not math.isfinite(number) or invalid:
        relation = "greater than" if strict else "at least"
        raise TrainingContractError(f"{name.replace('_', ' ')} must be {relation} {minimum}")
    return number


@dataclass(frozen=True)
class BrainVlmTrainingConfig:
    task: str
    annotations: tuple[Path, ...]
    image_root: Path
    llama_model: Path
    biomedclip_pretrained: str
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    warmup_learning_rate: float
    weight_decay: float
    warmup_steps: int
    iterations_per_epoch: int
    gradient_accumulation_steps: int
    seed: int
    num_workers: int
    use_age: bool
    upsample_ratio: float
    amp: bool
    device: str
    resume: Path | None = None
    initialization_checkpoint: Path | None = None
    distributed: bool = False
    world_size: int | None = None
    dist_url: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BrainVlmTrainingConfig":
        if not isinstance(mapping, Mapping):
            raise TrainingContractError("training configuration must be a mapping")
        missing = sorted(_REQUIRED_CONFIG_FIELDS - set(mapping))
        if missing:
            raise TrainingContractError(f"training configuration is missing: {missing[0]}")
        unknown = sorted(set(mapping) - _REQUIRED_CONFIG_FIELDS - _OPTIONAL_CONFIG_FIELDS)
        if unknown:
            raise TrainingContractError(f"training configuration has unknown field: {unknown[0]}")
        task = mapping["task"]
        if task not in {"tumor", "who"}:
            raise TrainingContractError("task must be tumor or who")
        annotations = mapping["annotations"]
        if not isinstance(annotations, list) or not annotations or any(
            not isinstance(item, str) or not item for item in annotations
        ):
            raise TrainingContractError("annotations must be a non-empty list of paths")
        for name in ("image_root", "llama_model", "output_dir"):
            if not isinstance(mapping[name], str) or not mapping[name]:
                raise TrainingContractError(f"{name.replace('_', ' ')} must be a path string")
        biomedclip = mapping["biomedclip_pretrained"]
        if not isinstance(biomedclip, str) or not biomedclip:
            raise TrainingContractError("biomedclip pretrained must be a non-empty string")
        for name in ("use_age", "amp"):
            if type(mapping[name]) is not bool:
                raise TrainingContractError(f"{name.replace('_', ' ')} must be a boolean")
        seed = mapping["seed"]
        if type(seed) is not int or seed < 0:
            raise TrainingContractError("seed must be a non-negative integer")
        upsample = _finite_number(mapping, "upsample_ratio", minimum=0.0, strict=False)
        if upsample > 1:
            raise TrainingContractError("upsample ratio must be at most 1")
        device = mapping["device"]
        if not isinstance(device, str) or _CUDA_DEVICE_PATTERN.fullmatch(device) is None:
            raise TrainingContractError("device must use the indexed cuda:N format")
        learning_rate = _finite_number(mapping, "learning_rate", minimum=0.0, strict=True)
        minimum_learning_rate = _finite_number(
            mapping, "min_learning_rate", minimum=0.0, strict=False
        )
        warmup_learning_rate = _finite_number(
            mapping, "warmup_learning_rate", minimum=0.0, strict=False
        )
        if minimum_learning_rate > learning_rate:
            raise TrainingContractError("minimum learning rate exceeds learning rate")
        if warmup_learning_rate > learning_rate:
            raise TrainingContractError("warmup learning rate exceeds learning rate")
        epochs = _positive_int(mapping, "epochs")
        iterations_per_epoch = _positive_int(mapping, "iterations_per_epoch")
        accumulation_steps = _positive_int(mapping, "gradient_accumulation_steps")
        warmup_steps = _positive_int(mapping, "warmup_steps", allow_zero=True)
        total_steps = epochs * math.ceil(iterations_per_epoch / accumulation_steps)
        if warmup_steps > total_steps:
            raise TrainingContractError("warmup steps exceed total optimizer steps")
        optional_paths: dict[str, Path | None] = {}
        for name in ("resume", "initialization_checkpoint"):
            value = mapping.get(name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TrainingContractError(f"{name.replace('_', ' ')} must be a path string")
            optional_paths[name] = None if value is None else Path(value)
        if optional_paths["resume"] is not None and optional_paths["initialization_checkpoint"] is not None:
            raise TrainingContractError(
                "resume and initialization checkpoint are mutually exclusive"
            )
        distributed = mapping.get("distributed", False)
        if type(distributed) is not bool:
            raise TrainingContractError("distributed must be a boolean")
        world_size = mapping.get("world_size")
        dist_url = mapping.get("dist_url")
        if world_size is not None and (type(world_size) is not int or world_size <= 0):
            raise TrainingContractError("world size must be a positive integer")
        if dist_url is not None and not isinstance(dist_url, str):
            raise TrainingContractError("dist url must be a string")
        if distributed:
            if world_size is None:
                raise TrainingContractError("world size is required for distributed training")
            if dist_url is None:
                raise TrainingContractError("dist url is required for distributed training")
            if dist_url != "env://":
                raise TrainingContractError("dist url must be env://")
        elif world_size is not None or dist_url is not None:
            raise TrainingContractError(
                "distributed must be true when distributed settings are supplied"
            )
        return cls(
            task=task,
            annotations=tuple(Path(item) for item in annotations),
            image_root=Path(mapping["image_root"]),
            llama_model=Path(mapping["llama_model"]),
            biomedclip_pretrained=biomedclip,
            output_dir=Path(mapping["output_dir"]),
            epochs=epochs,
            batch_size=_positive_int(mapping, "batch_size"),
            learning_rate=learning_rate,
            min_learning_rate=minimum_learning_rate,
            warmup_learning_rate=warmup_learning_rate,
            weight_decay=_finite_number(mapping, "weight_decay", minimum=0.0, strict=False),
            warmup_steps=warmup_steps,
            iterations_per_epoch=iterations_per_epoch,
            gradient_accumulation_steps=accumulation_steps,
            seed=seed,
            num_workers=_positive_int(mapping, "num_workers", allow_zero=True),
            use_age=mapping["use_age"],
            upsample_ratio=upsample,
            amp=mapping["amp"],
            device=device,
            resume=optional_paths["resume"],
            initialization_checkpoint=optional_paths["initialization_checkpoint"],
            distributed=distributed,
            world_size=world_size,
            dist_url=dist_url,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "annotations": [str(path) for path in self.annotations],
            "image_root": str(self.image_root),
            "llama_model": str(self.llama_model),
            "biomedclip_pretrained": self.biomedclip_pretrained,
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_learning_rate": self.warmup_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "iterations_per_epoch": self.iterations_per_epoch,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "use_age": self.use_age,
            "upsample_ratio": self.upsample_ratio,
            "amp": self.amp,
            "device": self.device,
            "resume": None if self.resume is None else str(self.resume),
            "initialization_checkpoint": (
                None if self.initialization_checkpoint is None else str(self.initialization_checkpoint)
            ),
            "distributed": self.distributed,
            "world_size": self.world_size,
            "dist_url": self.dist_url,
        }


def _is_trainable_name(name: str) -> bool:
    return (
        ".lora_A." in name
        or ".lora_B." in name
        or name.endswith(".lora_A.weight")
        or name.endswith(".lora_B.weight")
        or name.startswith("visual_encoder.norm.")
        or name.startswith("llama_proj_mlps.")
        or name.startswith("v_q_project.")
        or name.startswith("cross_attention.")
    )


def configure_trainable_parameters(model: nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(_is_trainable_name(name)))
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise TrainingContractError("model exposes no approved trainable parameters")
    return tuple(sorted(names))


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0 or weight_decay < 0:
        raise TrainingContractError("optimizer values are out of range")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or ".norm." in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise TrainingContractError("optimizer has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=learning_rate)


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        minimum_learning_rate: float,
        warmup_learning_rate: float,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0 or warmup_steps > total_steps:
            raise TrainingContractError("scheduler step counts are invalid")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.warmup_learning_rate = float(warmup_learning_rate)
        self.base_learning_rates = tuple(group["lr"] for group in optimizer.param_groups)
        if self.minimum_learning_rate < 0 or self.warmup_learning_rate < 0:
            raise TrainingContractError("scheduler learning rates must be non-negative")
        if any(
            max(self.minimum_learning_rate, self.warmup_learning_rate) > value
            for value in self.base_learning_rates
        ):
            raise TrainingContractError("scheduler learning rate exceeds the base learning rate")
        self.completed_steps = 0

    def step(self) -> float:
        step = min(self.completed_steps, self.total_steps - 1)
        if self.warmup_steps and step < self.warmup_steps:
            fraction = step / max(1, self.warmup_steps - 1)
            values = [
                self.warmup_learning_rate
                + (base - self.warmup_learning_rate) * fraction
                for base in self.base_learning_rates
            ]
        else:
            span = max(1, self.total_steps - self.warmup_steps - 1)
            progress = min(1.0, (step - self.warmup_steps) / span)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            values = [
                self.minimum_learning_rate + (base - self.minimum_learning_rate) * cosine
                for base in self.base_learning_rates
            ]
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value
        self.completed_steps += 1
        return values[0]

    def restore(self, completed_steps: int) -> None:
        if type(completed_steps) is not int or not 0 <= completed_steps <= self.total_steps:
            raise TrainingContractError("scheduler progress is out of bounds")
        self.completed_steps = completed_steps


@dataclass(frozen=True)
class EpochResult:
    mean_loss: float
    optimizer_steps: int


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def train_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    *,
    scaler: Any,
    gradient_accumulation_steps: int,
    max_iterations: int | None = None,
    device: torch.device,
    amp: bool,
) -> EpochResult:
    if gradient_accumulation_steps <= 0:
        raise TrainingContractError("gradient accumulation steps must be positive")
    if max_iterations is not None and (
        type(max_iterations) is not int or max_iterations <= 0
    ):
        raise TrainingContractError("maximum iterations must be a positive integer")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    pending = 0
    for index, raw_batch in enumerate(batches):
        if max_iterations is not None and index >= max_iterations:
            break
        batch = _to_device(dict(raw_batch), device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = model(batch, reduction="mean")
            loss = output["loss"]
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss must be a finite scalar")
            scaled_loss = loss / gradient_accumulation_steps
        losses.append(float(loss.detach()))
        pending += 1
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()
        should_step = pending == gradient_accumulation_steps
        if should_step:
            scheduler.step()
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending = 0
    if pending:
        remainder_scale = gradient_accumulation_steps / pending
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(remainder_scale)
        scheduler.step()
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if not losses:
        raise TrainingContractError("training epoch received no batches")
    total = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return EpochResult(float(total[0] / total[1]), optimizer_steps)


def _training_state(
    model: nn.Module,
    manifest: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    values: dict[str, torch.Tensor] = {}
    selected: set[str] = set()
    for name, shape in manifest.items():
        candidates = [name]
        for marker in (".lora_A.weight", ".lora_B.weight"):
            if marker in name:
                candidates.append(
                    name.replace(marker, marker.replace(".weight", ".default.weight"))
                )
        matches = [candidate for candidate in candidates if candidate in state]
        if not matches:
            raise TrainingContractError(
                "model does not expose the complete trainable manifest"
            )
        if len(matches) != 1 or matches[0] in selected:
            raise TrainingContractError("trainable state key normalization collision")
        selected.add(matches[0])
        tensor = state[matches[0]]
        if tuple(tensor.shape) != tuple(shape):
            raise TrainingContractError(f"trainable tensor {name} has an incompatible shape")
        values[name] = tensor.detach().cpu().contiguous()
    return values


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, tuple[int, ...]],
) -> None:
    if epoch < 0 or global_step < 0:
        raise TrainingContractError("checkpoint counters must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "format": TRAINING_FORMAT,
        "format_version": TRAINING_FORMAT_VERSION,
        "model": _training_state(model, manifest),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": dict(config),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(envelope, temporary)
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    global_step: int
    config: Mapping[str, Any]


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any,
    scheduler: WarmupCosineScheduler | None = None,
    manifest: Mapping[str, tuple[int, ...]],
) -> ResumeState:
    try:
        envelope = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingContractError("unable to read resume checkpoint") from exc
    required = {
        "format",
        "format_version",
        "model",
        "optimizer",
        "scaler",
        "epoch",
        "global_step",
        "config",
    }
    if not isinstance(envelope, Mapping) or set(envelope) != required:
        raise TrainingContractError("resume checkpoint is missing training state")
    if envelope["format"] != TRAINING_FORMAT or envelope["format_version"] != TRAINING_FORMAT_VERSION:
        raise TrainingContractError("resume checkpoint has an unsupported format")
    if type(envelope["epoch"]) is not int or envelope["epoch"] < 0:
        raise TrainingContractError("resume checkpoint epoch is invalid")
    if type(envelope["global_step"]) is not int or envelope["global_step"] < 0:
        raise TrainingContractError("resume checkpoint global step is invalid")
    try:
        load_brainvlm_checkpoint(Path(path), model, manifest=manifest)
        optimizer.load_state_dict(envelope["optimizer"])
        if scaler is not None and envelope["scaler"] is not None:
            scaler.load_state_dict(envelope["scaler"])
    except (BrainVlmCheckpointError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingContractError("resume checkpoint state is incompatible") from exc
    if not isinstance(envelope["config"], Mapping):
        raise TrainingContractError("resume checkpoint config is invalid")
    if scheduler is not None:
        scheduler.restore(envelope["global_step"])
    return ResumeState(
        start_epoch=envelope["epoch"] + 1,
        global_step=envelope["global_step"],
        config=dict(envelope["config"]),
    )
