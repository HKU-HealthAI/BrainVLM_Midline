"""Compose standalone checkpoint services into the Diagnostic workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ..datasets.inference_annotations import InferenceCase
from ..schema import CaseInput, DiagnosticResult, StepStatus
from .standalone import (
    TumorPredictionResult,
    TumorPredictionService,
    WhoGradePredictionResult,
    WhoGradePredictionService,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")


def _checkpoint_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} checkpoint digest is invalid")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validated_modality_counts(value: Mapping[str, int]) -> dict[str, int]:
    allowed = ("T1", "T1CE", "T2", "FLAIR")
    if set(value) != set(allowed):
        raise ValueError("tumor modality counts must cover the canonical modalities")
    return {name: _count(value[name], f"{name} count") for name in allowed}


def _validated_vote_counts(value: Mapping[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for grade, count in value.items():
        if grade not in {"grade_2", "grade_3", "grade_4"}:
            raise ValueError("WHO vote counts contain an invalid grade")
        counts[grade] = _count(count, f"{grade} vote count")
    return counts


def _preprocessing_version(
    tumor: TumorPredictionResult,
    who: WhoGradePredictionResult,
) -> str | None:
    versions = {
        value
        for value in (
            tumor.preprocessing.get("preprocessing_version"),
            who.preprocessing.get("preprocessing_version"),
        )
        if value is not None
    }
    if not versions:
        return None
    if len(versions) != 1:
        raise ValueError("diagnostic preprocessing versions do not match")
    version = next(iter(versions))
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("diagnostic preprocessing version is invalid")
    return version


def _diagnostic_provenance(
    tumor: TumorPredictionResult,
    who: WhoGradePredictionResult,
) -> dict[str, object]:
    checkpoint_keys = set(tumor.checkpoint_sha256)
    if checkpoint_keys != {"brainvlm_midline", "mil"}:
        raise ValueError("tumor checkpoint digests are incomplete")
    checkpoint_sha256 = {
        "tumor_model": _checkpoint_digest(
            tumor.checkpoint_sha256["brainvlm_midline"],
            "tumor model",
        ),
        "mil": _checkpoint_digest(tumor.checkpoint_sha256["mil"], "MIL"),
    }
    if who.checkpoint_sha256 is not None:
        checkpoint_sha256["who_model"] = _checkpoint_digest(
            who.checkpoint_sha256,
            "WHO model",
        )
    if who.method not in {"fixed_rule", "model_vote"}:
        raise ValueError("WHO method is invalid")
    if who.method == "model_vote" and who.checkpoint_sha256 is None:
        raise ValueError("model-voted WHO result requires a checkpoint digest")

    provenance: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256,
        "tumor_slice_count": _count(tumor.slice_count, "tumor slice count"),
        "aligned_slice_count": _count(
            tumor.aligned_slice_count,
            "aligned tumor slice count",
        ),
        "modality_counts": _validated_modality_counts(tumor.modality_counts),
        "who_method": who.method,
        "who_slice_count": _count(who.slice_count, "WHO slice count"),
        "vote_counts": _validated_vote_counts(who.vote_counts),
        "rejected_vote_count": _count(
            who.rejected_vote_count,
            "rejected WHO vote count",
        ),
        "invalid_vote_count": _count(
            who.invalid_vote_count,
            "invalid WHO vote count",
        ),
    }
    version = _preprocessing_version(tumor, who)
    if version is not None:
        provenance["preprocessing_version"] = version
    return provenance


@dataclass(frozen=True)
class DiagnosticWorkflow:
    """Run tumor prediction first, then tumor-conditioned WHO grading."""

    inference_case: InferenceCase
    tumor_service: TumorPredictionService
    who_grade_service: WhoGradePredictionService

    def run(self, case_input: CaseInput) -> DiagnosticResult:
        if case_input.case_id != self.inference_case.patient_id:
            raise ValueError("case_id does not match the diagnostic inference case")
        tumor = self.tumor_service.predict(self.inference_case)
        if not isinstance(tumor, TumorPredictionResult):
            raise TypeError("tumor service must return TumorPredictionResult")
        if tumor.patient_id != self.inference_case.patient_id:
            raise ValueError("tumor result patient_id does not match the inference case")
        who = self.who_grade_service.predict(self.inference_case, tumor.tumor_type)
        if not isinstance(who, WhoGradePredictionResult):
            raise TypeError("WHO service must return WhoGradePredictionResult")
        if who.patient_id != self.inference_case.patient_id:
            raise ValueError("WHO result patient_id does not match the inference case")
        if who.tumor_type != tumor.tumor_type:
            raise ValueError("WHO result tumor_type does not match the tumor result")
        return DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type=tumor.tumor_type,
            who_grade=who.who_grade,
            probabilities=tumor.probabilities,
            provenance=_diagnostic_provenance(tumor, who),
        )
