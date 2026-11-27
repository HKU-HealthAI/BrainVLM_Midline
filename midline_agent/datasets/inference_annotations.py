"""Validate label-free MRI annotations for standalone inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .annotations import AnnotationError, MODALITIES, MODALITY_ALIASES


@dataclass(frozen=True)
class InferenceSlice:
    patient_id: str
    image_path: Path
    modality: str
    slice_index: int
    age: float | None = None


@dataclass(frozen=True)
class InferenceCase:
    patient_id: str
    slices: tuple[InferenceSlice, ...]
    age: float | None = None


def _load_records(annotation_path: Path) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"unable to read annotation JSON: {annotation_path.name}") from exc
    if not isinstance(parsed, list):
        raise AnnotationError("annotation JSON must contain a top-level list")
    if not parsed:
        raise AnnotationError("annotation JSON must contain at least one record")
    if not all(isinstance(record, dict) for record in parsed):
        raise AnnotationError("each annotation record must be an object")
    return parsed


def _resolve_image(image_root: Path, raw_path: object, index: int) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise AnnotationError(f"record {index} image_path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise AnnotationError(f"record {index} image_path must be relative")
    resolved_root = image_root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise AnnotationError(f"record {index} image_path escapes image root") from exc
    if not candidate.is_file():
        raise AnnotationError(f"record {index} image does not exist")
    try:
        with Image.open(candidate) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AnnotationError(f"record {index} image cannot be decoded") from exc
    return candidate


def _parse_age(raw_age: object, index: int) -> float | None:
    if raw_age is None:
        return None
    if isinstance(raw_age, bool) or not isinstance(raw_age, (int, float)):
        raise AnnotationError(f"record {index} age must be numeric")
    age = float(raw_age)
    if not math.isfinite(age):
        raise AnnotationError(f"record {index} age must be finite")
    return age


def load_inference_case(annotation_path: Path, image_root: Path) -> InferenceCase:
    """Load one patient case without requiring diagnostic labels."""

    slices: list[InferenceSlice] = []
    seen_identities: set[tuple[str, int]] = set()
    patient_ids: set[str] = set()
    ages: set[float] = set()
    required = {"patient_id", "image_path", "modality", "slice_index"}

    for index, value in enumerate(_load_records(Path(annotation_path))):
        missing = required - value.keys()
        if missing:
            raise AnnotationError(f"record {index} missing fields: {', '.join(sorted(missing))}")
        patient_id = value["patient_id"]
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise AnnotationError(f"record {index} patient_id must be a non-empty string")
        modality = MODALITY_ALIASES.get(str(value["modality"]).upper())
        if modality is None:
            raise AnnotationError(f"record {index} has unsupported modality: {value['modality']}")
        raw_slice_index = value["slice_index"]
        if isinstance(raw_slice_index, bool) or not isinstance(raw_slice_index, int):
            raise AnnotationError(f"record {index} slice_index must be an integer")
        if raw_slice_index < 0:
            raise AnnotationError(f"record {index} slice_index must be non-negative")
        identity = (modality, raw_slice_index)
        if identity in seen_identities:
            raise AnnotationError(f"record {index} has a duplicate modality and slice_index")
        seen_identities.add(identity)
        age = _parse_age(value.get("age"), index)
        if age is not None:
            ages.add(age)
        patient_ids.add(patient_id)
        slices.append(
            InferenceSlice(
                patient_id=patient_id,
                image_path=_resolve_image(Path(image_root), value["image_path"], index),
                modality=modality,
                slice_index=raw_slice_index,
                age=age,
            )
        )

    if len(patient_ids) != 1:
        raise AnnotationError("inference annotations must contain exactly one patient")
    if len(ages) > 1:
        raise AnnotationError("all provided age values must agree")
    modality_order = {name: position for position, name in enumerate(MODALITIES)}
    ordered = tuple(sorted(slices, key=lambda item: (modality_order[item.modality], item.slice_index)))
    return InferenceCase(
        patient_id=next(iter(patient_ids)),
        slices=ordered,
        age=next(iter(ages)) if ages else None,
    )
