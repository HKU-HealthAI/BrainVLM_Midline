from pathlib import Path

import numpy as np
import pytest
import torch

from midline_agent.datasets.annotations import AnnotationRecord
from midline_agent.diagnostic.adapters import WhoGradeSliceResult
from midline_agent.diagnostic.attention_mil import MilOutput
from midline_agent.diagnostic.workflow import build_feature_store, diagnose_patient


class StubMilPredictor:
    def __init__(self) -> None:
        self.calls = []

    def predict_patient(self, patient, metadata) -> MilOutput:
        self.calls.append((patient, metadata))
        return MilOutput(
            logits=torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]]),
            evidence_weights=torch.tensor([[0.8]]),
            attention_weights=torch.tensor([[1.0]]),
        )


class StubDiagnosticAdapter:
    def predict_who_grade(self, request) -> WhoGradeSliceResult:
        return WhoGradeSliceResult(
            raw_text="<tumor_1> <grade_4>",
            logits=np.array([5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            prompt_version=request.prompt_version,
        )


def test_workflow_extracts_features_and_requires_mil_prediction(tmp_path: Path) -> None:
    image_path = tmp_path / "slice.npy"
    np.save(image_path, np.ones((8, 8), dtype=np.float32))
    records = [AnnotationRecord("synthetic-1", image_path, "T1", 0, "tumor_1", "grade_4")]

    store = build_feature_store(records, StubDiagnosticAdapter())
    store["patients"]["synthetic-1"]["raw_generations"] = ["<tumor_3> <grade_4>"]
    predictor = StubMilPredictor()
    result = diagnose_patient(store, "synthetic-1", predictor)

    assert store["patients"]["synthetic-1"]["logits"].shape == (1, 5)
    assert len(predictor.calls) == 1
    assert result.status.state == "completed"
    assert result.tumor_type == "tumor_3"
    assert result.who_grade == "grade_4"
    assert result.probabilities[2] > 0.9


def test_workflow_has_no_mean_aggregation_fallback() -> None:
    source = Path("midline_agent/diagnostic/workflow.py").read_text(encoding="utf-8")

    assert ".mean(" not in source
    with pytest.raises(TypeError):
        diagnose_patient({"metadata": {}, "patients": {"p": {}}}, "p")


def test_workflow_preserves_tumor_mismatched_votes_in_provenance(tmp_path: Path) -> None:
    image_path = tmp_path / "slice.npy"
    np.save(image_path, np.ones((8, 8), dtype=np.float32))
    store = build_feature_store(
        [AnnotationRecord("patient-a", image_path, "T1", 0, "tumor_1", "grade_4")],
        StubDiagnosticAdapter(),
    )
    store["patients"]["patient-a"]["raw_generations"] = [
        "<tumor_1> <grade_4>",
        "<tumor_3> <grade_4>",
    ]

    result = diagnose_patient(store, "patient-a", StubMilPredictor())

    assert result.status.state == "completed"
    assert result.tumor_type == "tumor_3"
    assert result.who_grade == "grade_4"
    assert result.provenance["rejected_vote_count"] == 1
    assert result.provenance["rejected_votes"][0]["tumor_type"] == "tumor_1"


def test_build_feature_store_rejects_inconsistent_patient_tumor_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "slice.npy"
    np.save(image_path, np.ones((8, 8), dtype=np.float32))
    records = [
        AnnotationRecord("patient-a", image_path, "T1", 0, "tumor_1"),
        AnnotationRecord("patient-a", image_path, "T2", 1, "tumor_2"),
    ]

    with pytest.raises(ValueError, match="inconsistent tumor labels"):
        build_feature_store(records, StubDiagnosticAdapter())


@pytest.mark.parametrize("tumor_type", ["", None, "unknown"])
def test_build_feature_store_rejects_missing_or_invalid_tumor_label(
    tmp_path: Path,
    tumor_type,
) -> None:
    image_path = tmp_path / "slice.npy"
    np.save(image_path, np.ones((8, 8), dtype=np.float32))
    records = [AnnotationRecord("patient-a", image_path, "T1", 0, tumor_type)]

    with pytest.raises(ValueError, match="invalid tumor label"):
        build_feature_store(records, StubDiagnosticAdapter())
