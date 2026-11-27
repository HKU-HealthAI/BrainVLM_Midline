"""Strict tumor and WHO-grade training datasets for BrainVLM-Midline."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .prompts import (
    TUMOR_LABELS,
    TUMOR_TRAINING_INSTRUCTIONS,
    WHO_TRAINING_INSTRUCTIONS,
    build_tumor_training_prompt,
    build_who_training_prompt,
    validate_tumor_grade,
)
from .vision import preprocess_image


class DatasetContractError(ValueError):
    """Raised when public training data violates the declared schema."""


@dataclass(frozen=True)
class TrainingRecord:
    image_path: Path
    image_name: str
    tumor_label: str
    grade_label: str
    age: float | None
    patient_id: str | None


@dataclass(frozen=True)
class TaskSample:
    image: torch.Tensor
    prompt: str
    answer: str
    image_name: str
    task: str


def _read_rows(path: Path) -> list[tuple[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            rows: list[tuple[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("figure"), str):
                    raise DatasetContractError(
                        f"JSONL record {line_number} requires a string figure field"
                    )
                rows.append((value["figure"], value))
            return rows
        value = json.loads(text)
    except DatasetContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError("unable to read training annotations") from exc
    if not isinstance(value, dict) or not value:
        raise DatasetContractError("annotation JSON must contain a non-empty object")
    return list(value.items())


def _safe_image_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise DatasetContractError("image name must be a non-empty string")
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / name).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetContractError("image path escapes image root") from exc
    if not candidate.is_file():
        raise DatasetContractError(f"image file does not exist: {name}")
    try:
        with Image.open(candidate) as image:
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise DatasetContractError(f"unable to decode image: {name}") from exc
    return candidate


def _age(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DatasetContractError("age must be a finite non-negative number")
    try:
        age = float(value)
    except ValueError as exc:
        raise DatasetContractError("age must be a finite non-negative number") from exc
    if not math.isfinite(age) or age < 0:
        raise DatasetContractError("age must be a finite non-negative number")
    return age


def load_training_records(
    annotation_paths: Path | Sequence[Path],
    image_root: Path,
) -> tuple[TrainingRecord, ...]:
    """Parse validated JSON object or JSONL annotations in stable file order."""
    paths = [annotation_paths] if isinstance(annotation_paths, (str, Path)) else list(annotation_paths)
    if not paths:
        raise DatasetContractError("at least one annotation path is required")
    records: list[TrainingRecord] = []
    patient_labels: dict[str, tuple[str, str]] = {}
    for annotation_path in paths:
        for image_name, raw in _read_rows(Path(annotation_path)):
            if not isinstance(raw, dict):
                raise DatasetContractError("each annotation record must be an object")
            tumor = raw.get("who_class", raw.get("WHO_class"))
            grade = raw.get("who_grade")
            if tumor not in TUMOR_LABELS:
                raise DatasetContractError("record has an unsupported tumor label")
            if not isinstance(grade, str):
                raise DatasetContractError("record is missing a WHO grade label")
            try:
                validate_tumor_grade(tumor, grade)
            except ValueError as exc:
                raise DatasetContractError(str(exc)) from exc
            patient_id = raw.get("patient_id")
            if patient_id is not None and (not isinstance(patient_id, str) or not patient_id):
                raise DatasetContractError("patient_id must be a non-empty string when supplied")
            pair = (tumor, grade)
            if patient_id is not None and patient_id in patient_labels and patient_labels[patient_id] != pair:
                raise DatasetContractError("patient has inconsistent tumor or grade labels")
            if patient_id is not None:
                patient_labels[patient_id] = pair
            records.append(
                TrainingRecord(
                    image_path=_safe_image_path(Path(image_root), image_name),
                    image_name=image_name,
                    tumor_label=tumor,
                    grade_label=grade,
                    age=_age(raw.get("age", raw.get("Age"))),
                    patient_id=patient_id,
                )
            )
    if not records:
        raise DatasetContractError("training annotations contain no records")
    return tuple(records)


T = TypeVar("T")


def upsample_records(
    records: Sequence[T],
    *,
    key: Callable[[T], str],
    ratio: float,
    seed: int,
) -> tuple[T, ...]:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise DatasetContractError("upsampling ratio must be finite and in [0, 1]")
    groups: dict[str, list[T]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    if not groups:
        raise DatasetContractError("cannot upsample an empty record set")
    target = int(max(map(len, groups.values())) * float(ratio))
    generator = random.Random(seed)
    output: list[T] = []
    for name in sorted(groups):
        values = groups[name]
        output.extend(values)
        current = len(values)
        while current < target:
            output.append(generator.choice(values))
            current += 1
    generator.shuffle(output)
    return tuple(output)


class _TaskDataset(Dataset[TaskSample]):
    task: str

    def __init__(
        self,
        records: Sequence[TrainingRecord],
        *,
        seed: int,
        use_age: bool,
    ) -> None:
        if not records:
            raise DatasetContractError("task dataset requires at least one record")
        self.records = tuple(records)
        self.seed = int(seed)
        self.use_age = bool(use_age)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise DatasetContractError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _instruction(self, index: int, instructions: Sequence[str]) -> str:
        generator = random.Random(self.seed + self.epoch * 1_000_003 + index)
        return instructions[generator.randrange(len(instructions))]


class TumorDataset(_TaskDataset):
    task = "tumor"

    def __getitem__(self, index: int) -> TaskSample:
        record = self.records[index]
        instruction = self._instruction(index, TUMOR_TRAINING_INSTRUCTIONS)
        prompt = build_tumor_training_prompt(
            instruction,
            record.age if self.use_age else None,
        )
        return TaskSample(
            preprocess_image(record.image_path),
            prompt,
            record.tumor_label,
            record.image_name,
            self.task,
        )


class WhoGradeDataset(_TaskDataset):
    task = "who"
    _TRAINABLE_TUMORS = frozenset({"<tumor_2>", "<tumor_4>", "<tumor_5>"})

    def __init__(
        self,
        records: Sequence[TrainingRecord],
        *,
        seed: int,
        use_age: bool,
    ) -> None:
        super().__init__(
            [record for record in records if record.tumor_label in self._TRAINABLE_TUMORS],
            seed=seed,
            use_age=use_age,
        )

    def __getitem__(self, index: int) -> TaskSample:
        record = self.records[index]
        instruction = self._instruction(index, WHO_TRAINING_INSTRUCTIONS)
        prompt = build_who_training_prompt(
            instruction,
            record.age if self.use_age else None,
        )
        return TaskSample(
            preprocess_image(record.image_path),
            prompt,
            f"{record.tumor_label} {record.grade_label}",
            record.image_name,
            self.task,
        )


def build_mixed_schedule(
    length: int,
    *,
    who_fraction: float = 0.8,
    seed: int,
    epoch: int,
) -> tuple[str, ...]:
    if type(length) is not int or length <= 0:
        raise DatasetContractError("schedule length must be a positive integer")
    if not isinstance(who_fraction, (int, float)) or not math.isfinite(who_fraction) or not 0 <= who_fraction <= 1:
        raise DatasetContractError("WHO fraction must be finite and in [0, 1]")
    who_count = round(length * float(who_fraction))
    schedule = ["who"] * who_count + ["tumor"] * (length - who_count)
    random.Random(int(seed) + int(epoch) * 1_000_003).shuffle(schedule)
    return tuple(schedule)
