"""Attention-based patient-level aggregation of MRI slice features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionMilConfig:
    logit_dimension: int = 5
    input_hidden_dimension: int = 0
    hidden_dimension: int = 128
    num_modalities: int = 4
    modality_embedding_dimension: int = 4
    num_classes: int = 5
    dropout: float = 0.1


@dataclass(frozen=True)
class MilOutput:
    logits: torch.Tensor
    evidence_weights: torch.Tensor
    attention_weights: torch.Tensor

    @property
    def attention(self) -> torch.Tensor:
        return self.attention_weights


class AttentionMIL(nn.Module):
    def __init__(
        self,
        logit_dimension: int = 5,
        input_hidden_dimension: int = 0,
        hidden_dimension: int = 128,
        num_modalities: int = 4,
        modality_embedding_dimension: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = AttentionMilConfig(
            logit_dimension=logit_dimension,
            input_hidden_dimension=input_hidden_dimension,
            hidden_dimension=hidden_dimension,
            num_modalities=num_modalities,
            modality_embedding_dimension=modality_embedding_dimension,
            num_classes=num_classes,
            dropout=dropout,
        )
        if logit_dimension < 0 or input_hidden_dimension < 0 or logit_dimension + input_hidden_dimension == 0:
            raise ValueError("AttentionMIL requires logits, hidden features, or both")
        if hidden_dimension <= 0 or num_modalities <= 0 or modality_embedding_dimension <= 0 or num_classes <= 0:
            raise ValueError("AttentionMIL dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.modality_embedding = nn.Embedding(
            num_modalities + 1,
            modality_embedding_dimension,
            padding_idx=num_modalities,
        )
        self.evidence_calibration = nn.Linear(logit_dimension, 1) if logit_dimension else None
        feature_dimension = logit_dimension + input_hidden_dimension + modality_embedding_dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_dimension,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.score = nn.Linear(hidden_dimension, 1)
        self.classifier = nn.Linear(hidden_dimension, num_classes)

    def configuration(self) -> dict[str, int | float]:
        return asdict(self.config)

    def _validate_inputs(
        self,
        logits: torch.Tensor | None,
        hidden: torch.Tensor | None,
        modalities: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("mask must be a rank-2 boolean tensor")
        if modalities.dtype not in {torch.int32, torch.int64} or modalities.shape != mask.shape:
            raise ValueError("modalities must be an integer tensor aligned with mask")
        if mask.shape[1] == 0 or not bool(mask.any(dim=1).all()):
            raise ValueError("each bag must contain at least one valid instance")
        allowed_modalities = (modalities == -1) | ((modalities >= 0) & (modalities < self.config.num_modalities))
        if not bool(allowed_modalities.all()) or not bool((modalities[mask] >= 0).all()):
            raise ValueError("modality index is invalid")
        features: list[torch.Tensor] = []
        for name, tensor, dimension in (
            ("logits", logits, self.config.logit_dimension),
            ("hidden", hidden, self.config.input_hidden_dimension),
        ):
            if dimension == 0:
                if tensor is not None:
                    raise ValueError(f"{name} were not configured for this model")
                continue
            if tensor is None or tensor.ndim != 3 or tensor.shape[:2] != mask.shape or tensor.shape[2] != dimension:
                raise ValueError(f"{name} have an incompatible shape")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain finite floating-point values")
            features.append(tensor)
        return tuple(features)

    def forward(
        self,
        logits: torch.Tensor | None,
        modalities: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> MilOutput:
        features = self._validate_inputs(logits, hidden, modalities, mask)
        instance_mask = mask.unsqueeze(-1)
        safe_logits = torch.where(instance_mask, logits, torch.zeros_like(logits)) if logits is not None else None
        safe_hidden = torch.where(instance_mask, hidden, torch.zeros_like(hidden)) if hidden is not None else None
        calibrated_features: list[torch.Tensor] = []
        if safe_logits is not None:
            if self.evidence_calibration is None:
                raise RuntimeError("logit calibration layer is unavailable")
            evidence_weights = nn.functional.softplus(self.evidence_calibration(safe_logits).squeeze(-1))
            evidence_weights = (evidence_weights + torch.finfo(safe_logits.dtype).eps).masked_fill(~mask, 0.0)
            calibrated_features.append(safe_logits * evidence_weights.unsqueeze(-1))
        else:
            evidence_weights = mask.to(dtype=features[0].dtype)
        if safe_hidden is not None:
            calibrated_features.append(safe_hidden)
        safe_modalities = torch.where(
            mask,
            modalities,
            torch.full_like(modalities, self.config.num_modalities),
        )
        inputs = torch.cat([*calibrated_features, self.modality_embedding(safe_modalities)], dim=-1)
        encoded = self.feature_projection(inputs)
        contextual, _ = self.self_attention(encoded, encoded, encoded, key_padding_mask=~mask)
        scores = self.score(contextual).squeeze(-1).masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)
        pooled = (contextual * attention_weights.unsqueeze(-1)).sum(dim=1)
        return MilOutput(
            logits=self.classifier(pooled),
            evidence_weights=evidence_weights,
            attention_weights=attention_weights,
        )
