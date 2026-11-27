"""Inference for the released learnable-logits MIL checkpoint."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from midline_agent.datasets.annotations import MODALITIES
from midline_agent.datasets.inference_annotations import InferenceSlice


TUMOR_TYPES = ("tumor_1", "tumor_2", "tumor_3", "tumor_4", "tumor_5")
TUMOR_TYPE_NAMES = MappingProxyType(
    {
        "tumor_1": "Diffuse midline glioma, H3 K27-altered",
        "tumor_2": "Paediatric-type diffuse glioma (checkpoint superclass)",
        "tumor_3": "Glioblastoma, IDH-wildtype",
        "tumor_4": "Astrocytoma, IDH-mutant",
        "tumor_5": "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted",
    }
)
CHECKPOINT_CLASS_DESCRIPTIONS = MappingProxyType(
    {
        "tumor_1": "H3K27M-mutated midline glioma",
        "tumor_2": "Pediatric-type diffuse glioma",
        "tumor_3": "IDH-wildtype glioblastoma",
        "tumor_4": "IDH-mutated astrocytoma",
        "tumor_5": "IDH-mutated oligodendroglioma",
    }
)
SLICES_PER_MODALITY = 5


class LogitsMilCheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class LogitsMilResult:
    tumor_type: str
    logits: torch.Tensor
    probabilities: torch.Tensor


def _selected_positions(items: Sequence[tuple[int, InferenceSlice]]) -> list[int]:
    ordered = sorted(items, key=lambda item: item[1].slice_index)
    if len(ordered) >= SLICES_PER_MODALITY:
        start = (len(ordered) - SLICES_PER_MODALITY) // 2
        return [position for position, _ in ordered[start : start + SLICES_PER_MODALITY]]
    return [ordered[index % len(ordered)][0] for index in range(SLICES_PER_MODALITY)]


def align_slice_logits(slices: Sequence[InferenceSlice], slice_logits: torch.Tensor) -> torch.Tensor:
    """Reproduce the 20-slice ordering used to train the released MIL model."""

    if not isinstance(slice_logits, torch.Tensor) or slice_logits.ndim != 2 or slice_logits.shape != (len(slices), 5):
        raise ValueError("slice logits must have shape [number of slices, 5]")
    if not slice_logits.is_floating_point():
        raise ValueError("slice logits must use a floating-point dtype")
    if not bool(torch.isfinite(slice_logits).all()):
        raise ValueError("slice logits must contain only finite values")

    grouped: dict[str, list[tuple[int, InferenceSlice]]] = {modality: [] for modality in MODALITIES}
    for position, item in enumerate(slices):
        if item.modality not in grouped:
            raise ValueError(f"unsupported modality: {item.modality}")
        grouped[item.modality].append((position, item))

    selected: list[int] = []
    for modality in MODALITIES:
        if not grouped[modality]:
            raise ValueError(f"at least one {modality} slice is required")
        selected.extend(_selected_positions(grouped[modality]))
    return slice_logits[selected]


def _load_checkpoint(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise LogitsMilCheckpointError("MIL checkpoint path must be a file")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise LogitsMilCheckpointError("unable to read MIL checkpoint safely") from exc
    if not isinstance(checkpoint, Mapping):
        raise LogitsMilCheckpointError("MIL checkpoint must contain a mapping")
    args = checkpoint.get("args")
    if (
        not isinstance(args, Mapping)
        or args.get("model_type") != "logits"
        or args.get("aggregation") != "learnable"
    ):
        raise LogitsMilCheckpointError("MIL checkpoint architecture is not learnable logits")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or set(state) != {"weights", "temperature"}:
        raise LogitsMilCheckpointError("MIL checkpoint tensor names do not match the released model")
    weights = state["weights"]
    temperature = state["temperature"]
    if not isinstance(weights, torch.Tensor) or weights.shape != (5,) or not weights.is_floating_point():
        raise LogitsMilCheckpointError("MIL checkpoint weights have an incompatible shape or dtype")
    if not isinstance(temperature, torch.Tensor) or temperature.shape != () or not temperature.is_floating_point():
        raise LogitsMilCheckpointError("MIL checkpoint temperature has an incompatible shape or dtype")
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(temperature)):
        raise LogitsMilCheckpointError("MIL checkpoint tensors must be finite")
    if float(temperature) <= 0:
        raise LogitsMilCheckpointError("MIL checkpoint temperature must be positive")
    return weights.detach().cpu(), temperature.detach().cpu()


class LogitsMilPredictor:
    def __init__(self, checkpoint_path: Path) -> None:
        self.class_weights, self.temperature = _load_checkpoint(checkpoint_path)

    def predict(self, aligned_logits: torch.Tensor) -> LogitsMilResult:
        if not isinstance(aligned_logits, torch.Tensor) or aligned_logits.shape != (20, 5):
            raise ValueError("MIL input must be a 20 by 5 tensor")
        if not aligned_logits.is_floating_point() or not bool(torch.isfinite(aligned_logits).all()):
            raise ValueError("MIL input must contain finite floating-point values")
        logits = (aligned_logits.cpu() * self.class_weights).mean(dim=0) / self.temperature
        probabilities = torch.softmax(logits, dim=0)
        tumor_type = TUMOR_TYPES[int(torch.argmax(probabilities).item())]
        return LogitsMilResult(tumor_type=tumor_type, logits=logits, probabilities=probabilities)
