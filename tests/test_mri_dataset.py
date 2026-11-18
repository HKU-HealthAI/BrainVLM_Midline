import numpy as np

from midline_agent.datasets.annotations import AnnotationRecord
from midline_agent.datasets.mri_dataset import evaluation_image_array, ordered_records


def test_evaluation_image_array_is_deterministic_and_rgb() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    first = evaluation_image_array(image, size=8)
    second = evaluation_image_array(image, size=8)

    assert first.shape == (3, 8, 8)
    assert np.array_equal(first, second)


def test_ordered_records_uses_modality_then_slice_index() -> None:
    records = [
        AnnotationRecord("p", "b.png", "FLAIR", 0, "tumor_1"),
        AnnotationRecord("p", "a.png", "T1", 3, "tumor_1"),
        AnnotationRecord("p", "c.png", "T1CE", 1, "tumor_1"),
    ]

    assert [record.modality for record in ordered_records(records)] == ["T1", "T1CE", "FLAIR"]
