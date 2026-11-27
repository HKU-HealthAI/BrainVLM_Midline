import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image

from midline_agent.diagnostic.brainvlm_midline.collate import collate_task_samples
from midline_agent.diagnostic.brainvlm_midline.datasets import (
    DatasetContractError,
    TaskSample,
    TumorDataset,
    WhoGradeDataset,
    build_mixed_schedule,
    load_training_records,
    upsample_records,
)
from midline_agent.diagnostic.brainvlm_midline.prompts import (
    TUMOR_TRAINING_INSTRUCTIONS,
    WHO_TRAINING_INSTRUCTIONS,
)


FIXTURE = Path(__file__).parents[1] / "fixtures/brainvlm_training/annotations.json"


def _write_images(root: Path, names=("a.png", "b.png", "c.png")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        Image.new("RGB", (16, 12), color=(index, 64, 128)).save(root / name)


def test_object_annotations_accept_historical_field_aliases(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_images(image_root)

    records = load_training_records(FIXTURE, image_root)

    assert [record.tumor_label for record in records] == [
        "<tumor_2>",
        "<tumor_4>",
        "<tumor_1>",
    ]
    assert [record.grade_label for record in records] == [
        "<grade_3>",
        "<grade_3>",
        "<grade_4>",
    ]
    assert [record.age for record in records] == [17.0, 45.0, 51.0]


def test_jsonl_requires_figure_and_preserves_order(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_images(image_root, ("one.png", "two.png"))
    annotations = tmp_path / "records.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"figure": "two.png", "who_class": "<tumor_5>", "who_grade": "<grade_2>"}),
                json.dumps({"figure": "one.png", "who_class": "<tumor_3>", "who_grade": "<grade_4>"}),
            ]
        )
        + "\n",
        encoding="ascii",
    )

    records = load_training_records(annotations, image_root)

    assert [record.image_path.name for record in records] == ["two.png", "one.png"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"../escape.png": {"who_class": "<tumor_1>", "who_grade": "<grade_4>"}}, "escapes"),
        ({"a.png": {"who_class": "<tumor_9>", "who_grade": "<grade_4>"}}, "tumor"),
        ({"a.png": {"who_class": "<tumor_4>"}}, "grade"),
        ({"a.png": {"who_class": "<tumor_4>", "who_grade": "<grade_2>", "age": float("nan")}}, "age"),
    ],
)
def test_records_reject_invalid_paths_labels_and_age(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    image_root = tmp_path / "images"
    _write_images(image_root)
    annotations = tmp_path / "records.json"
    annotations.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(DatasetContractError, match=message):
        load_training_records(annotations, image_root)


def test_records_reject_undecodable_images(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "bad.png").write_text("bad", encoding="ascii")
    annotations = tmp_path / "records.json"
    annotations.write_text(
        json.dumps({"bad.png": {"who_class": "<tumor_2>", "who_grade": "<grade_3>"}}),
        encoding="ascii",
    )

    with pytest.raises(DatasetContractError, match="decode"):
        load_training_records(annotations, image_root)


def test_records_reject_inconsistent_labels_for_one_patient(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_images(image_root, ("one.png", "two.png"))
    annotations = tmp_path / "records.json"
    annotations.write_text(
        json.dumps(
            {
                "one.png": {
                    "patient_id": "same-patient",
                    "who_class": "<tumor_2>",
                    "who_grade": "<grade_3>",
                },
                "two.png": {
                    "patient_id": "same-patient",
                    "who_class": "<tumor_4>",
                    "who_grade": "<grade_3>",
                },
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(DatasetContractError, match="inconsistent"):
        load_training_records(annotations, image_root)


def test_task_datasets_are_seeded_filtered_and_preprocessed(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_images(image_root)
    records = load_training_records(FIXTURE, image_root)
    tumor = TumorDataset(records, seed=7, use_age=True)
    same = TumorDataset(records, seed=7, use_age=True)
    who = WhoGradeDataset(records, seed=7, use_age=True)

    assert len(tumor) == 3
    assert len(who) == 2
    assert len(TUMOR_TRAINING_INSTRUCTIONS) == 3
    assert len(WHO_TRAINING_INSTRUCTIONS) == 4
    assert tumor[0].prompt == same[0].prompt
    assert tumor[0].image.shape == (3, 224, 224)
    assert tumor[0].image.dtype == torch.float32
    assert tumor[0].answer == "<tumor_2>"
    assert who[0].answer == "<tumor_2> <grade_3>"
    assert {item.tumor_label for item in who.records} == {"<tumor_2>", "<tumor_4>"}
    tumor.set_epoch(1)
    same.set_epoch(1)
    assert tumor[0].prompt == same[0].prompt


def test_threshold_upsampling_is_reproducible() -> None:
    records = [
        type("Record", (), {"tumor_label": label})()
        for label in ("a", "a", "a", "a", "b")
    ]
    first = upsample_records(records, key=lambda item: item.tumor_label, ratio=0.75, seed=5)
    second = upsample_records(records, key=lambda item: item.tumor_label, ratio=0.75, seed=5)

    assert [item.tumor_label for item in first] == [item.tumor_label for item in second]
    assert Counter(item.tumor_label for item in first) == {"a": 4, "b": 3}


def test_mixed_schedule_is_seeded_and_uses_four_to_one_ratio() -> None:
    first = build_mixed_schedule(20, who_fraction=0.8, seed=11, epoch=2)
    second = build_mixed_schedule(20, who_fraction=0.8, seed=11, epoch=2)

    assert first == second
    assert Counter(first) == {"who": 16, "tumor": 4}


def test_collator_stacks_complete_finite_samples() -> None:
    samples = [
        TaskSample(torch.ones(3, 224, 224), "p1", "a1", "one.png", "tumor"),
        TaskSample(torch.zeros(3, 224, 224), "p2", "a2", "two.png", "tumor"),
    ]

    batch = collate_task_samples(samples)

    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["instruction_input"] == ["p1", "p2"]
    assert batch["answer"] == ["a1", "a2"]


@pytest.mark.parametrize(
    "samples",
    [
        [TaskSample(torch.ones(3, 224, 224), "p", "", "x", "tumor")],
        [TaskSample(torch.full((3, 224, 224), float("nan")), "p", "a", "x", "tumor")],
        [
            TaskSample(torch.ones(3, 224, 224), "p", "a", "x", "tumor"),
            TaskSample(torch.ones(3, 10, 10), "p", "a", "y", "tumor"),
        ],
    ],
)
def test_collator_rejects_invalid_samples(samples: list[TaskSample]) -> None:
    with pytest.raises(DatasetContractError):
        collate_task_samples(samples)
