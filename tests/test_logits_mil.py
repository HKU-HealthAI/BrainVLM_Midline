from pathlib import Path

import pytest
import torch

from midline_agent.datasets.inference_annotations import InferenceSlice
from midline_agent.diagnostic.logits_mil import (
    LogitsMilCheckpointError,
    LogitsMilPredictor,
    align_slice_logits,
)


def _slice(modality: str, slice_index: int) -> InferenceSlice:
    return InferenceSlice("patient-a", Path(f"{modality}-{slice_index}.png"), modality, slice_index)


def _write_checkpoint(
    path: Path,
    model: dict[str, torch.Tensor] | None = None,
    args: dict[str, object] | None = None,
) -> None:
    torch.save(
        {
            "model": model
            if model is not None
            else {"weights": torch.arange(1, 6, dtype=torch.float32), "temperature": torch.tensor(2.0)},
            "args": args if args is not None else {"model_type": "logits", "aggregation": "learnable"},
            "epoch": 4,
        },
        path,
    )


def test_aligns_modalities_with_center_selection_and_cyclic_repetition() -> None:
    slices = [
        *[_slice("T1", index) for index in range(7)],
        *[_slice("T1CE", index) for index in range(5)],
        *[_slice("T2", index) for index in range(2)],
        _slice("FLAIR", 9),
    ]
    logits = torch.stack(
        [torch.full((5,), float(position), dtype=torch.float32) for position in range(len(slices))]
    )

    aligned = align_slice_logits(slices, logits)

    assert aligned.shape == (20, 5)
    assert aligned[:, 0].tolist() == [
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        12,
        13,
        12,
        14,
        14,
        14,
        14,
        14,
    ]


def test_alignment_sorts_each_modality_by_slice_index() -> None:
    slices = []
    rows = []
    for modality_position, modality in enumerate(("T1", "T1CE", "T2", "FLAIR")):
        for slice_index in (4, 2, 3, 0, 1):
            slices.append(_slice(modality, slice_index))
            rows.append(torch.full((5,), modality_position * 10 + slice_index, dtype=torch.float32))

    aligned = align_slice_logits(slices, torch.stack(rows))

    assert aligned[:, 0].tolist() == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34]


def test_alignment_requires_all_modalities_and_valid_logits() -> None:
    slices = [_slice("T1", 0), _slice("T1CE", 0), _slice("T2", 0)]
    with pytest.raises(ValueError, match="FLAIR"):
        align_slice_logits(slices, torch.ones(3, 5))
    with pytest.raises(ValueError, match="shape"):
        align_slice_logits([*slices, _slice("FLAIR", 0)], torch.ones(4, 4))
    bad = torch.ones(4, 5)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        align_slice_logits([*slices, _slice("FLAIR", 0)], bad)


def test_predictor_uses_exact_learnable_logits_arithmetic(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "mil.pth"
    class_weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
    temperature = torch.tensor(2.0)
    _write_checkpoint(checkpoint_path, {"weights": class_weights, "temperature": temperature})
    slice_logits = torch.arange(100, dtype=torch.float32).reshape(20, 5) / 10

    result = LogitsMilPredictor(checkpoint_path).predict(slice_logits)

    expected_logits = (slice_logits * class_weights).mean(dim=0) / temperature
    torch.testing.assert_close(result.logits, expected_logits)
    torch.testing.assert_close(result.probabilities, torch.softmax(expected_logits, dim=0))
    assert result.tumor_type == f"tumor_{int(expected_logits.argmax()) + 1}"


def test_checkpoint_loader_requests_weights_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "mil.pth"
    _write_checkpoint(checkpoint_path)
    original_load = torch.load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    LogitsMilPredictor(checkpoint_path)

    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"weights": torch.ones(5)}, "tensor names"),
        ({"weights": torch.ones(5), "temperature": torch.tensor(1.0), "extra": torch.tensor(0)}, "tensor names"),
        ({"weights": torch.ones(4), "temperature": torch.tensor(1.0)}, "weights"),
        ({"weights": torch.ones(5), "temperature": torch.ones(1)}, "temperature"),
        ({"weights": torch.tensor([1.0, 1.0, float("nan"), 1.0, 1.0]), "temperature": torch.tensor(1.0)}, "finite"),
        ({"weights": torch.ones(5), "temperature": torch.tensor(0.0)}, "positive"),
    ],
)
def test_rejects_invalid_checkpoint_tensors(
    tmp_path: Path,
    model: dict[str, torch.Tensor],
    message: str,
) -> None:
    checkpoint_path = tmp_path / "mil.pth"
    _write_checkpoint(checkpoint_path, model)

    with pytest.raises(LogitsMilCheckpointError, match=message):
        LogitsMilPredictor(checkpoint_path)


@pytest.mark.parametrize(
    "args",
    [
        {"model_type": "attention", "aggregation": "learnable"},
        {"model_type": "logits", "aggregation": "mean"},
    ],
)
def test_rejects_checkpoint_for_another_mil_architecture(tmp_path: Path, args: dict[str, object]) -> None:
    checkpoint_path = tmp_path / "mil.pth"
    _write_checkpoint(checkpoint_path, args=args)

    with pytest.raises(LogitsMilCheckpointError, match="architecture"):
        LogitsMilPredictor(checkpoint_path)


def test_rejects_non_file_checkpoint_and_invalid_prediction_input(tmp_path: Path) -> None:
    with pytest.raises(LogitsMilCheckpointError, match="file"):
        LogitsMilPredictor(tmp_path / "missing.pth")

    checkpoint_path = tmp_path / "mil.pth"
    _write_checkpoint(checkpoint_path)
    predictor = LogitsMilPredictor(checkpoint_path)
    with pytest.raises(ValueError, match="20 by 5"):
        predictor.predict(torch.ones(19, 5))
