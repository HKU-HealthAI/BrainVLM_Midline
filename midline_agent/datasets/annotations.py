"""Load validated public MRI slice annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODALITIES = ("T1", "T1CE", "T2", "FLAIR")
MODALITY_ALIASES = {"T1": "T1", "T1C": "T1CE", "T1CE": "T1CE", "T2": "T2", "FLAIR": "FLAIR"}


class AnnotationError(ValueError):
    pass


@dataclass(frozen=True)
class AnnotationRecord:
    patient_id: str
    image_path: Path | str
    modality: str
    slice_index: int
    tumor_type: str
    who_grade: str | None = None
    keyframe_label: int | None = None
    report_text: str | None = None


def _records_from_file(annotation_path: Path) -> list[dict[str, Any]]:
    text = annotation_path.read_text(encoding="utf-8")
    if annotation_path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise AnnotationError("annotation JSON must contain a top-level list")
    return parsed


def _safe_image_path(image_root: Path, raw_path: str) -> Path:
    resolved_root = image_root.resolve()
    candidate = (resolved_root / raw_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise AnnotationError("image_path escapes image root") from exc
    return candidate


def load_annotations(annotation_path: Path, image_root: Path) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    for index, value in enumerate(_records_from_file(Path(annotation_path))):
        required = {"patient_id", "image_path", "modality", "slice_index", "tumor_type"}
        missing = required - value.keys()
        if missing:
            raise AnnotationError(f"record {index} missing fields: {', '.join(sorted(missing))}")
        modality = MODALITY_ALIASES.get(str(value["modality"]).upper())
        if modality is None:
            raise AnnotationError(f"record {index} has unsupported modality: {value['modality']}")
        records.append(
            AnnotationRecord(
                patient_id=str(value["patient_id"]),
                image_path=_safe_image_path(Path(image_root), str(value["image_path"])),
                modality=modality,
                slice_index=int(value["slice_index"]),
                tumor_type=str(value["tumor_type"]),
                who_grade=value.get("who_grade"),
                keyframe_label=value.get("keyframe_label"),
                report_text=value.get("report_text"),
            )
        )
    return records
