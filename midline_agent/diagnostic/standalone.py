"""Framework-independent services for standalone diagnostic stages."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import torch

from midline_agent.datasets.annotations import MODALITIES
from midline_agent.datasets.inference_annotations import InferenceCase, InferenceSlice

from .logits_mil import LogitsMilPredictor, TUMOR_TYPES, TUMOR_TYPE_NAMES, align_slice_logits
from .who_grade import vote_who_grade_for_tumor


RESULT_SCHEMA_VERSION = "1.1"
FIXED_GRADE_4_TUMORS = frozenset({"tumor_1", "tumor_3"})


class TumorRuntime(Protocol):
    preprocessing_metadata: Mapping[str, object]

    def predict_tumor_logits(self, image_path: Path, metadata: InferenceSlice) -> torch.Tensor: ...


class WhoRuntime(Protocol):
    preprocessing_metadata: Mapping[str, object]

    def predict_who_generation(
        self,
        image_path: Path,
        metadata: InferenceSlice,
        tumor_type: str,
        age: float | None,
    ) -> str: ...


def checkpoint_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError("checkpoint path must be a file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TumorPredictionResult:
    patient_id: str
    tumor_type: str
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    slice_count: int
    aligned_slice_count: int
    modality_counts: Mapping[str, int]
    checkpoint_sha256: Mapping[str, str]
    preprocessing: Mapping[str, object]
    schema_version: str = RESULT_SCHEMA_VERSION

    @property
    def tumor_name(self) -> str:
        return TUMOR_TYPE_NAMES[self.tumor_type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "tumor_type": self.tumor_type,
            "tumor_name": self.tumor_name,
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
            "slice_count": self.slice_count,
            "aligned_slice_count": self.aligned_slice_count,
            "modality_counts": dict(self.modality_counts),
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "preprocessing": dict(self.preprocessing),
        }


@dataclass(frozen=True)
class WhoGradePredictionResult:
    patient_id: str
    tumor_type: str
    who_grade: str
    method: str
    slice_count: int
    vote_counts: Mapping[str, int]
    rejected_vote_count: int
    invalid_vote_count: int
    checkpoint_sha256: str | None
    preprocessing: Mapping[str, object]
    schema_version: str = RESULT_SCHEMA_VERSION

    @property
    def tumor_name(self) -> str:
        return TUMOR_TYPE_NAMES[self.tumor_type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "tumor_type": self.tumor_type,
            "tumor_name": self.tumor_name,
            "who_grade": self.who_grade,
            "method": self.method,
            "slice_count": self.slice_count,
            "vote_counts": dict(self.vote_counts),
            "rejected_vote_count": self.rejected_vote_count,
            "invalid_vote_count": self.invalid_vote_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing": dict(self.preprocessing),
        }


class TumorPredictionService:
    def __init__(
        self,
        runtime: TumorRuntime,
        mil_predictor: LogitsMilPredictor,
        model_checkpoint: Path,
        mil_checkpoint: Path,
    ) -> None:
        self.runtime = runtime
        self.mil_predictor = mil_predictor
        self.model_checkpoint = Path(model_checkpoint)
        self.mil_checkpoint = Path(mil_checkpoint)

    def predict(self, case: InferenceCase) -> TumorPredictionResult:
        rows: list[torch.Tensor] = []
        for item in case.slices:
            row = self.runtime.predict_tumor_logits(item.image_path, item)
            if not isinstance(row, torch.Tensor) or row.shape != (5,) or not row.is_floating_point():
                raise ValueError("tumor runtime must return one five-value floating-point tensor per slice")
            if not bool(torch.isfinite(row).all()):
                raise ValueError("tumor runtime returned non-finite logits")
            rows.append(row.detach().cpu())
        if not rows:
            raise ValueError("tumor prediction requires at least one slice")
        aligned = align_slice_logits(case.slices, torch.stack(rows))
        prediction = self.mil_predictor.predict(aligned)
        counts = Counter(item.modality for item in case.slices)
        return TumorPredictionResult(
            patient_id=case.patient_id,
            tumor_type=prediction.tumor_type,
            logits=tuple(float(value) for value in prediction.logits.tolist()),
            probabilities=tuple(float(value) for value in prediction.probabilities.tolist()),
            slice_count=len(case.slices),
            aligned_slice_count=int(aligned.shape[0]),
            modality_counts=MappingProxyType({name: counts[name] for name in MODALITIES}),
            checkpoint_sha256=MappingProxyType(
                {
                    "brainvlm_midline": checkpoint_sha256(self.model_checkpoint),
                    "mil": checkpoint_sha256(self.mil_checkpoint),
                }
            ),
            preprocessing=MappingProxyType(dict(self.runtime.preprocessing_metadata)),
        )


class WhoGradePredictionService:
    def __init__(
        self,
        runtime_factory: Callable[[], WhoRuntime],
        model_checkpoint: Path | None,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.model_checkpoint = Path(model_checkpoint) if model_checkpoint is not None else None

    def predict(self, case: InferenceCase, tumor_type: str) -> WhoGradePredictionResult:
        if tumor_type not in TUMOR_TYPES:
            raise ValueError("unsupported tumor type")
        if tumor_type in FIXED_GRADE_4_TUMORS:
            return WhoGradePredictionResult(
                patient_id=case.patient_id,
                tumor_type=tumor_type,
                who_grade="grade_4",
                method="fixed_rule",
                slice_count=0,
                vote_counts=MappingProxyType({"grade_4": 1}),
                rejected_vote_count=0,
                invalid_vote_count=0,
                checkpoint_sha256=None,
                preprocessing=MappingProxyType({}),
            )
        if self.model_checkpoint is None:
            raise ValueError("WHO model checkpoint is required for this tumor type")
        model_sha256 = checkpoint_sha256(self.model_checkpoint)
        runtime = self.runtime_factory()
        generations = [
            runtime.predict_who_generation(item.image_path, item, tumor_type, case.age)
            for item in case.slices
        ]
        vote = vote_who_grade_for_tumor(generations, tumor_type)
        if vote.status != "completed" or vote.grade is None:
            raise ValueError(vote.reason or "no valid WHO-grade predictions")
        return WhoGradePredictionResult(
            patient_id=case.patient_id,
            tumor_type=tumor_type,
            who_grade=vote.grade,
            method="model_vote",
            slice_count=len(case.slices),
            vote_counts=MappingProxyType(dict(vote.vote_counts or {})),
            rejected_vote_count=len(vote.rejected_votes),
            invalid_vote_count=vote.invalid_vote_count,
            checkpoint_sha256=model_sha256,
            preprocessing=MappingProxyType(dict(runtime.preprocessing_metadata)),
        )
