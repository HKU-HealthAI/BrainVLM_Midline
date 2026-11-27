from pathlib import Path

import pytest

from midline_agent.datasets.inference_annotations import InferenceCase, InferenceSlice
from midline_agent.diagnostic.standalone import (
    TumorPredictionResult,
    WhoGradePredictionResult,
)
from midline_agent.diagnostic.workflow import DiagnosticWorkflow
from midline_agent.schema import CaseInput


def _case(tmp_path: Path, patient_id: str = "patient-a") -> InferenceCase:
    image = tmp_path / "slice.png"
    image.write_bytes(b"not-read-by-workflow")
    item = InferenceSlice(patient_id, image, "T1", 0, 52.0)
    return InferenceCase(patient_id, (item,), 52.0)


def _tumor_result(patient_id: str = "patient-a") -> TumorPredictionResult:
    return TumorPredictionResult(
        patient_id=patient_id,
        tumor_type="tumor_4",
        logits=(0.0, 0.0, 0.0, 4.0, 0.0),
        probabilities=(0.05, 0.05, 0.05, 0.8, 0.05),
        slice_count=30,
        aligned_slice_count=20,
        modality_counts={"T1": 7, "T1CE": 8, "T2": 7, "FLAIR": 8},
        checkpoint_sha256={
            "brainvlm_midline": "a" * 64,
            "mil": "b" * 64,
        },
        preprocessing={
            "preprocessing_version": "brainvlm-midline-v1",
            "private_path": "/private/tumor/checkpoint",
        },
    )


def _who_result(patient_id: str = "patient-a") -> WhoGradePredictionResult:
    return WhoGradePredictionResult(
        patient_id=patient_id,
        tumor_type="tumor_4",
        who_grade="grade_3",
        method="model_vote",
        slice_count=30,
        vote_counts={"grade_2": 2, "grade_3": 27},
        rejected_vote_count=1,
        invalid_vote_count=0,
        checkpoint_sha256="c" * 64,
        preprocessing={
            "preprocessing_version": "brainvlm-midline-v1",
            "raw_generation": "PRIVATE_GENERATION_SENTINEL",
            "url": "https://private.invalid/model",
        },
    )


class StubTumorService:
    def __init__(self, result: TumorPredictionResult, calls: list[object]) -> None:
        self.result = result
        self.calls = calls

    def predict(self, case: InferenceCase) -> TumorPredictionResult:
        self.calls.append(("tumor", case))
        return self.result


class StubWhoService:
    def __init__(self, result: WhoGradePredictionResult, calls: list[object]) -> None:
        self.result = result
        self.calls = calls

    def predict(self, case: InferenceCase, tumor_type: str) -> WhoGradePredictionResult:
        self.calls.append(("who", case, tumor_type))
        return self.result


def test_workflow_runs_shared_services_in_order_and_converts_results(tmp_path: Path) -> None:
    inference_case = _case(tmp_path)
    calls: list[object] = []
    workflow = DiagnosticWorkflow(
        inference_case,
        StubTumorService(_tumor_result(), calls),
        StubWhoService(_who_result(), calls),
    )

    result = workflow.run(CaseInput(case_id="patient-a"))

    assert calls == [
        ("tumor", inference_case),
        ("who", inference_case, "tumor_4"),
    ]
    assert result.status.state == "completed"
    assert result.tumor_type == "tumor_4"
    assert result.who_grade == "grade_3"
    assert result.probabilities == (0.05, 0.05, 0.05, 0.8, 0.05)
    assert result.provenance == {
        "checkpoint_sha256": {
            "tumor_model": "a" * 64,
            "mil": "b" * 64,
            "who_model": "c" * 64,
        },
        "preprocessing_version": "brainvlm-midline-v1",
        "tumor_slice_count": 30,
        "aligned_slice_count": 20,
        "modality_counts": {"T1": 7, "T1CE": 8, "T2": 7, "FLAIR": 8},
        "who_method": "model_vote",
        "who_slice_count": 30,
        "vote_counts": {"grade_2": 2, "grade_3": 27},
        "rejected_vote_count": 1,
        "invalid_vote_count": 0,
    }
    serialized = str(result.provenance)
    assert "PRIVATE_GENERATION_SENTINEL" not in serialized
    assert "/private/" not in serialized
    assert "private.invalid" not in serialized


def test_workflow_rejects_case_input_for_another_patient_before_inference(tmp_path: Path) -> None:
    calls: list[object] = []
    workflow = DiagnosticWorkflow(
        _case(tmp_path),
        StubTumorService(_tumor_result(), calls),
        StubWhoService(_who_result(), calls),
    )

    with pytest.raises(ValueError, match="case_id"):
        workflow.run(CaseInput(case_id="patient-b"))

    assert calls == []


@pytest.mark.parametrize("stage", ["tumor", "who"])
def test_workflow_rejects_standalone_result_for_another_patient(
    tmp_path: Path,
    stage: str,
) -> None:
    calls: list[object] = []
    tumor = _tumor_result("patient-b" if stage == "tumor" else "patient-a")
    who = _who_result("patient-b" if stage == "who" else "patient-a")
    workflow = DiagnosticWorkflow(
        _case(tmp_path),
        StubTumorService(tumor, calls),
        StubWhoService(who, calls),
    )

    with pytest.raises(ValueError, match="patient_id"):
        workflow.run(CaseInput(case_id="patient-a"))


def test_workflow_rejects_who_result_for_another_tumor_type(tmp_path: Path) -> None:
    calls: list[object] = []
    who = _who_result()
    object.__setattr__(who, "tumor_type", "tumor_2")
    workflow = DiagnosticWorkflow(
        _case(tmp_path),
        StubTumorService(_tumor_result(), calls),
        StubWhoService(who, calls),
    )

    with pytest.raises(ValueError, match="tumor_type"):
        workflow.run(CaseInput(case_id="patient-a"))


def test_workflow_contains_no_legacy_adapter_or_attention_mil_path() -> None:
    source = Path("midline_agent/diagnostic/workflow.py").read_text(encoding="ascii")

    for removed in (
        "BrainVlmAttentionMilWorkflow",
        "build_feature_store",
        "diagnose_patient",
        "DiagnosticModelAdapter",
        "AttentionMIL",
        "raw_generations",
    ):
        assert removed not in source
