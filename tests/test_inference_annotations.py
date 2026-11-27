import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from PIL import Image

from midline_agent.datasets.annotations import AnnotationError
from midline_agent.datasets.inference_annotations import load_inference_case


def _write_png(path: Path, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), color=value).save(path)


def _write_annotations(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _record(
    image_path: str,
    modality: str = "T1",
    slice_index: int = 1,
    patient_id: str = "patient-a",
    age: object = 42,
) -> dict[str, object]:
    value: dict[str, object] = {
        "patient_id": patient_id,
        "image_path": image_path,
        "modality": modality,
        "slice_index": slice_index,
    }
    if age is not None:
        value["age"] = age
    return value


def test_loads_one_immutable_case_in_canonical_order(tmp_path: Path) -> None:
    for name in ("flair.png", "t1ce.png", "t1.png", "t2.png"):
        _write_png(tmp_path / "images" / name)
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(
        annotation_path,
        [
            _record("images/flair.png", "flair", 8),
            _record("images/t1ce.png", "t1c", 4),
            _record("images/t1.png", "t1", 6),
            _record("images/t2.png", "T2", 3),
        ],
    )

    case = load_inference_case(annotation_path, tmp_path)

    assert case.patient_id == "patient-a"
    assert case.age == 42.0
    assert [(item.modality, item.slice_index) for item in case.slices] == [
        ("T1", 6),
        ("T1CE", 4),
        ("T2", 3),
        ("FLAIR", 8),
    ]
    assert all(isinstance(item.image_path, Path) and item.image_path.is_absolute() for item in case.slices)
    with pytest.raises(FrozenInstanceError):
        case.age = 50.0  # type: ignore[misc]


@pytest.mark.parametrize("raw_path", ["../outside.png", "/tmp/absolute.png"])
def test_rejects_unsafe_image_paths(tmp_path: Path, raw_path: str) -> None:
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(annotation_path, [_record(raw_path)])

    with pytest.raises(AnnotationError, match="image_path"):
        load_inference_case(annotation_path, tmp_path)


def test_rejects_missing_and_undecodable_images(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(annotation_path, [_record("missing.png")])
    with pytest.raises(AnnotationError, match="does not exist"):
        load_inference_case(annotation_path, tmp_path)

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not an image", encoding="ascii")
    _write_annotations(annotation_path, [_record("invalid.png")])
    with pytest.raises(AnnotationError, match="decode"):
        load_inference_case(annotation_path, tmp_path)


def test_rejects_duplicate_slice_identity(tmp_path: Path) -> None:
    _write_png(tmp_path / "one.png")
    _write_png(tmp_path / "two.png")
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(
        annotation_path,
        [_record("one.png", "T1", 3), _record("two.png", "t1", 3)],
    )

    with pytest.raises(AnnotationError, match="duplicate"):
        load_inference_case(annotation_path, tmp_path)


def test_rejects_multiple_patients_and_inconsistent_ages(tmp_path: Path) -> None:
    _write_png(tmp_path / "one.png")
    _write_png(tmp_path / "two.png")
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(
        annotation_path,
        [_record("one.png"), _record("two.png", modality="T2", patient_id="patient-b")],
    )
    with pytest.raises(AnnotationError, match="one patient"):
        load_inference_case(annotation_path, tmp_path)

    _write_annotations(
        annotation_path,
        [_record("one.png", "T1", 1, age=42), _record("two.png", "T2", 1, age=43)],
    )
    with pytest.raises(AnnotationError, match="age"):
        load_inference_case(annotation_path, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modality", "DWI", "unsupported modality"),
        ("age", float("inf"), "finite"),
        ("age", True, "numeric"),
        ("slice_index", True, "integer"),
    ],
)
def test_rejects_invalid_metadata(tmp_path: Path, field: str, value: object, message: str) -> None:
    _write_png(tmp_path / "image.png")
    record = _record("image.png")
    record[field] = value
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(annotation_path, [record])

    with pytest.raises(AnnotationError, match=message):
        load_inference_case(annotation_path, tmp_path)


def test_rejects_empty_or_incomplete_annotations(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    _write_annotations(annotation_path, [])
    with pytest.raises(AnnotationError, match="at least one"):
        load_inference_case(annotation_path, tmp_path)

    _write_annotations(annotation_path, [{"patient_id": "patient-a"}])
    with pytest.raises(AnnotationError, match="missing fields"):
        load_inference_case(annotation_path, tmp_path)
