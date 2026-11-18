import json
from pathlib import Path

import pytest

from midline_agent.datasets.annotations import AnnotationError, load_annotations


def test_loader_normalizes_modality_and_relative_path(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [{"patient_id": "synthetic-1", "image_path": "images/t1.png", "modality": "t1", "slice_index": 3, "tumor_type": "tumor_1"}]
        ),
        encoding="utf-8",
    )

    record = load_annotations(annotation_path, tmp_path)[0]

    assert record.modality == "T1"
    assert record.image_path == tmp_path / "images/t1.png"


def test_loader_rejects_paths_outside_image_root(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [{"patient_id": "synthetic-1", "image_path": "../private.png", "modality": "T1", "slice_index": 0, "tumor_type": "tumor_1"}]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AnnotationError, match="escapes image root"):
        load_annotations(annotation_path, tmp_path)
