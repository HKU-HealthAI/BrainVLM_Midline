"""Connect canonical MRI slices, external adapters, features, and diagnosis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import torch
from PIL import Image

from ..datasets.annotations import AnnotationRecord, MODALITIES
from ..schema import CaseInput, DiagnosticResult, StepStatus
from .adapters import DiagnosticModelAdapter, WhoGradeRequest
from .attention_mil import MilOutput
from .who_grade import vote_who_grade_for_tumor


TUMOR_TYPES = ("tumor_1", "tumor_2", "tumor_3", "tumor_4", "tumor_5")
MODALITY_MAP = {name: index for index, name in enumerate(MODALITIES)}


class PatientMilPredictor(Protocol):
    def predict_patient(self, patient: Mapping, feature_metadata: Mapping) -> MilOutput: ...


@dataclass(frozen=True)
class BrainVlmAttentionMilWorkflow:
    """Execute BrainVLM slice inference followed by patient-level AttentionMIL."""

    records: tuple[AnnotationRecord, ...]
    adapter: DiagnosticModelAdapter
    predictor: PatientMilPredictor

    def run(self, case_input: CaseInput) -> DiagnosticResult:
        store = build_feature_store(list(self.records), self.adapter)
        return diagnose_patient(store, case_input.case_id, self.predictor)


def _load_image(path: Path | str) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[None, ...], 3, axis=0)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    if image.shape[0] != 3:
        raise ValueError(f"expected 2D or RGB image, got shape {image.shape}")
    return image


def build_feature_store(records: list[AnnotationRecord], adapter: DiagnosticModelAdapter) -> dict:
    grouped: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)
    patients = {}
    for patient_id, patient_records in grouped.items():
        labels = [record.tumor_type for record in patient_records]
        if any(label not in TUMOR_TYPES for label in labels):
            raise ValueError(f"patient {patient_id} has an invalid tumor label")
        if len(set(labels)) != 1:
            raise ValueError(f"patient {patient_id} has inconsistent tumor labels")
        ordered = sorted(patient_records, key=lambda item: (MODALITY_MAP[item.modality], item.slice_index))
        logits, generations, slice_metadata = [], [], []
        prompt_versions: set[str] = set()
        for record in ordered:
            result = adapter.predict_who_grade(WhoGradeRequest(slice_id=f"{patient_id}:{record.modality}:{record.slice_index}", image=_load_image(record.image_path)))
            prompt_versions.add(result.prompt_version)
            logits.append(torch.as_tensor(result.logits, dtype=torch.float32))
            generations.append(result.raw_text)
            slice_metadata.append({"modality": record.modality, "slice_index": record.slice_index, "checksum": ""})
        label = TUMOR_TYPES.index(labels[0])
        patients[patient_id] = {
            "label": label,
            "slice_ids": [f"{patient_id}:{item.modality}:{item.slice_index}" for item in ordered],
            "modalities": [MODALITY_MAP[item.modality] for item in ordered],
            "logits": torch.stack(logits),
            "valid_mask": [True] * len(ordered),
            "slice_metadata": slice_metadata,
            "raw_generations": generations,
        }
    return {
        "metadata": {
            "schema_version": "1.0",
            "feature_kind": "logits",
            "logit_dimension": 5,
            "hidden_dimension": 0,
            "modality_map": MODALITY_MAP,
            "prompt_version": next(iter(prompt_versions)) if len(prompt_versions) == 1 else "mixed",
        },
        "patients": patients,
    }


def diagnose_patient(store: dict, patient_id: str, predictor: PatientMilPredictor) -> DiagnosticResult:
    patient = store["patients"][patient_id]
    prediction = predictor.predict_patient(patient, store["metadata"])
    if (
        not isinstance(prediction.logits, torch.Tensor)
        or prediction.logits.shape != (1, len(TUMOR_TYPES))
        or not bool(torch.isfinite(prediction.logits).all())
    ):
        raise ValueError("AttentionMIL predictor must return one finite tumor logit vector")
    patient_logits = prediction.logits[0]
    class_index = int(patient_logits.argmax())
    tumor_type = TUMOR_TYPES[class_index]
    grade = vote_who_grade_for_tumor(patient.get("raw_generations", []), tumor_type)
    rejected_votes = [
        {
            "slice_index": vote.slice_index,
            "tumor_type": vote.tumor_type,
            "who_grade": vote.grade,
            "reason": vote.reason,
        }
        for vote in grade.rejected_votes
    ]
    provenance = {
        "raw_generations": patient["raw_generations"],
        "who_vote_counts": grade.vote_counts,
        "rejected_votes": rejected_votes,
        "rejected_vote_count": len(rejected_votes),
        "mil_evidence_weights": prediction.evidence_weights.detach().cpu().flatten().tolist(),
        "mil_attention_weights": prediction.attention_weights.detach().cpu().flatten().tolist(),
    }
    if grade.status != "completed":
        return DiagnosticResult(
            status=StepStatus("failed", grade.reason),
            tumor_type=tumor_type,
            provenance=provenance,
        )
    probabilities = torch.softmax(patient_logits, dim=0).tolist()
    return DiagnosticResult(
        status=StepStatus("completed"),
        tumor_type=tumor_type,
        who_grade=grade.grade,
        probabilities=tuple(float(value) for value in probabilities),
        provenance=provenance,
    )
