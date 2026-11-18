"""Split-safe training helpers for the standalone AttentionMIL model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..datasets.splits import load_patient_splits
from .attention_mil import AttentionMIL
from .checkpoints import save_attention_mil_checkpoint
from .feature_store import validate_feature_store


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    validation_loss: float


def build_optimizer(model: AttentionMIL, learning_rate: float) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def _patient_tensors(
    patient: Mapping[str, Any],
    num_classes: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        "label" not in patient
        or type(patient["label"]) is not int
        or not 0 <= patient["label"] < num_classes
    ):
        raise ValueError("every training patient label must be in the configured class range")
    logits = patient.get("logits")
    hidden = patient.get("hidden")
    if logits is not None:
        logits = logits.unsqueeze(0)
    if hidden is not None:
        hidden = hidden.unsqueeze(0)
    modalities = torch.as_tensor(patient["modalities"], dtype=torch.long).unsqueeze(0)
    mask = torch.as_tensor(patient["valid_mask"], dtype=torch.bool).unsqueeze(0)
    label = torch.tensor([patient["label"]], dtype=torch.long)
    return logits, hidden, modalities, mask, label


def training_step(
    model: AttentionMIL,
    logits: torch.Tensor | None,
    modalities: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    hidden: torch.Tensor | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(logits=logits, hidden=hidden, modalities=modalities, mask=mask)
    if labels.dtype != torch.long or bool(((labels < 0) | (labels >= output.logits.shape[1])).any()):
        raise ValueError("training labels must be integer indices in the configured class range")
    loss = nn.functional.cross_entropy(output.logits, labels)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _validation_loss(model: AttentionMIL, patients: list[Mapping[str, Any]]) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for patient in patients:
            logits, hidden, modalities, mask, label = _patient_tensors(patient, model.config.num_classes)
            output = model(logits=logits, hidden=hidden, modalities=modalities, mask=mask)
            loss = nn.functional.cross_entropy(output.logits, label)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    if not math.isfinite(mean_loss):
        raise FloatingPointError("non-finite validation loss")
    return mean_loss


def train_attention_mil(
    *,
    store: dict[str, Any],
    train_split_path: Path,
    validation_split_path: Path,
    checkpoint_path: Path,
    epochs: int,
    learning_rate: float = 1e-3,
    model_hidden_dimension: int = 128,
    dropout: float = 0.1,
    num_classes: int = 5,
    seed: int = 0,
) -> TrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    validate_feature_store(store)
    patients = store["patients"]
    splits = load_patient_splits(train_split_path, validation_split_path, patients.keys())
    train_patients = [patients[patient_id] for patient_id in splits.train_ids]
    validation_patients = [patients[patient_id] for patient_id in splits.validation_ids]
    for patient in [*train_patients, *validation_patients]:
        _patient_tensors(patient, num_classes)
    metadata = store["metadata"]
    torch.manual_seed(seed)
    model = AttentionMIL(
        logit_dimension=metadata["logit_dimension"],
        input_hidden_dimension=metadata["hidden_dimension"],
        hidden_dimension=model_hidden_dimension,
        num_modalities=len(metadata["modality_map"]),
        num_classes=num_classes,
        dropout=dropout,
    )
    optimizer = build_optimizer(model, learning_rate)
    best_loss = math.inf
    best_epoch = -1
    for epoch in range(epochs):
        for patient in train_patients:
            logits, hidden, modalities, mask, label = _patient_tensors(patient, num_classes)
            training_step(model, logits, modalities, mask, label, optimizer, hidden=hidden)
        validation_loss = _validation_loss(model, validation_patients)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            checkpoint_metadata = {
                "feature_schema_version": metadata["schema_version"],
                "feature_kind": metadata["feature_kind"],
                "logit_dimension": metadata["logit_dimension"],
                "hidden_dimension": metadata["hidden_dimension"],
                "modality_map": metadata["modality_map"],
                "train_split_sha256": splits.train_sha256,
                "validation_split_sha256": splits.validation_sha256,
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
            }
            save_attention_mil_checkpoint(checkpoint_path, model, checkpoint_metadata)
    return TrainingResult(best_epoch=best_epoch, validation_loss=best_loss)
