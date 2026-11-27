import hashlib
from pathlib import Path

import pytest
import torch

from midline_agent.datasets.inference_annotations import InferenceCase, InferenceSlice
from midline_agent.diagnostic.logits_mil import LogitsMilPredictor
from midline_agent.diagnostic.standalone import TumorPredictionService, WhoGradePredictionService


def _case(tmp_path: Path) -> InferenceCase:
    slices = tuple(
        InferenceSlice(
            patient_id="patient-a",
            image_path=tmp_path / f"{modality}-{position}.png",
            modality=modality,
            slice_index=position,
            age=52.0,
        )
        for position, modality in enumerate(("T1", "T1CE", "T2", "FLAIR"))
    )
    return InferenceCase(patient_id="patient-a", slices=slices, age=52.0)


def _write_mil(path: Path) -> None:
    torch.save(
        {
            "model": {"weights": torch.ones(5), "temperature": torch.tensor(1.0)},
            "args": {"model_type": "logits", "aggregation": "learnable"},
        },
        path,
    )


def _write_checkpoint(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


class FakeTumorRuntime:
    preprocessing_metadata = {"image_size": 224, "normalization": "biomedclip"}

    def __init__(self) -> None:
        self.calls: list[tuple[Path, InferenceSlice]] = []

    def predict_tumor_logits(self, image_path: Path, metadata: InferenceSlice) -> torch.Tensor:
        self.calls.append((image_path, metadata))
        result = torch.zeros(5)
        result[3] = float(metadata.slice_index + 4)
        return result


class FakeWhoRuntime:
    preprocessing_metadata = {"image_size": 224, "normalization": "biomedclip"}

    def __init__(self, generations: list[str]) -> None:
        self.generations = iter(generations)
        self.calls: list[tuple[Path, InferenceSlice, str, float | None]] = []

    def predict_who_generation(
        self,
        image_path: Path,
        metadata: InferenceSlice,
        tumor_type: str,
        age: float | None,
    ) -> str:
        self.calls.append((image_path, metadata, tumor_type, age))
        return next(self.generations)


def test_tumor_service_calls_every_slice_and_returns_path_free_result(tmp_path: Path) -> None:
    case = _case(tmp_path)
    model_checkpoint = tmp_path / "tumor.pth"
    model_sha = _write_checkpoint(model_checkpoint, b"tumor checkpoint")
    mil_checkpoint = tmp_path / "mil.pth"
    _write_mil(mil_checkpoint)
    mil_sha = hashlib.sha256(mil_checkpoint.read_bytes()).hexdigest()
    runtime = FakeTumorRuntime()

    result = TumorPredictionService(
        runtime=runtime,
        mil_predictor=LogitsMilPredictor(mil_checkpoint),
        model_checkpoint=model_checkpoint,
        mil_checkpoint=mil_checkpoint,
    ).predict(case)
    payload = result.to_dict()

    assert len(runtime.calls) == 4
    assert result.tumor_type == "tumor_4"
    assert result.slice_count == 4
    assert result.modality_counts == {"T1": 1, "T1CE": 1, "T2": 1, "FLAIR": 1}
    assert result.aligned_slice_count == 20
    assert result.checkpoint_sha256 == {"brainvlm_midline": model_sha, "mil": mil_sha}
    assert abs(sum(result.probabilities) - 1.0) < 1e-6
    assert payload["schema_version"] == "1.0"
    assert str(tmp_path) not in str(payload)


@pytest.mark.parametrize("tumor_type", ["tumor_1", "tumor_3"])
def test_who_rules_do_not_construct_runtime(tmp_path: Path, tumor_type: str) -> None:
    constructed = 0

    def factory():
        nonlocal constructed
        constructed += 1
        raise AssertionError("WHO runtime must not be constructed for a fixed rule")

    result = WhoGradePredictionService(runtime_factory=factory, model_checkpoint=None).predict(
        _case(tmp_path), tumor_type
    )

    assert constructed == 0
    assert result.who_grade == "grade_4"
    assert result.method == "fixed_rule"
    assert result.slice_count == 0
    assert result.checkpoint_sha256 is None


def test_who_service_strictly_votes_matching_generations(tmp_path: Path) -> None:
    checkpoint = tmp_path / "who.pth"
    checkpoint_sha = _write_checkpoint(checkpoint, b"who checkpoint")
    runtime = FakeWhoRuntime(
        [
            "<tumor_4> <grade_3>",
            "<tumor_2> <grade_2>",
            "unparseable",
            "<tumor_4> <grade_3>",
        ]
    )
    service = WhoGradePredictionService(runtime_factory=lambda: runtime, model_checkpoint=checkpoint)

    result = service.predict(_case(tmp_path), "tumor_4")
    payload = result.to_dict()

    assert len(runtime.calls) == 4
    assert all(call[2:] == ("tumor_4", 52.0) for call in runtime.calls)
    assert result.who_grade == "grade_3"
    assert result.method == "model_vote"
    assert result.vote_counts == {"grade_3": 2}
    assert result.rejected_vote_count == 1
    assert result.invalid_vote_count == 1
    assert result.checkpoint_sha256 == checkpoint_sha
    assert "unparseable" not in str(payload)
    assert str(tmp_path) not in str(payload)


def test_who_service_fails_when_no_matching_vote_remains(tmp_path: Path) -> None:
    checkpoint = tmp_path / "who.pth"
    _write_checkpoint(checkpoint, b"who checkpoint")
    runtime = FakeWhoRuntime(["<tumor_2> <grade_2>"] * 4)

    with pytest.raises(ValueError, match="no valid WHO-grade predictions"):
        WhoGradePredictionService(lambda: runtime, checkpoint).predict(_case(tmp_path), "tumor_4")


def test_who_service_breaks_grade_ties_deterministically(tmp_path: Path) -> None:
    checkpoint = tmp_path / "who.pth"
    _write_checkpoint(checkpoint, b"who checkpoint")
    runtime = FakeWhoRuntime(
        [
            "<tumor_5> <grade_3>",
            "<tumor_5> <grade_2>",
            "<tumor_5> <grade_3>",
            "<tumor_5> <grade_2>",
        ]
    )

    result = WhoGradePredictionService(lambda: runtime, checkpoint).predict(_case(tmp_path), "tumor_5")

    assert result.who_grade == "grade_2"
    assert result.vote_counts == {"grade_2": 2, "grade_3": 2}


def test_who_service_rejects_unsupported_tumor_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported tumor type"):
        WhoGradePredictionService(lambda: None, None).predict(_case(tmp_path), "tumor_9")
