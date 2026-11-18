"""Patient-level split loading with explicit leakage checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class PatientSplits:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_sha256: str
    validation_sha256: str


def _load_split(path: Path, name: str) -> tuple[tuple[str, ...], str]:
    path = Path(path)
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise SplitError(f"unable to read {name} split file") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"patient_ids"}:
        raise SplitError(f"{name} split must contain only patient_ids")
    patient_ids = parsed["patient_ids"]
    if not isinstance(patient_ids, list) or not patient_ids:
        raise SplitError(f"{name} split must contain a non-empty patient_ids list")
    if any(not isinstance(value, str) or not value for value in patient_ids):
        raise SplitError(f"{name} split patient IDs must be non-empty strings")
    if len(patient_ids) != len(set(patient_ids)):
        raise SplitError(f"{name} split contains duplicate patient IDs")
    return tuple(patient_ids), hashlib.sha256(payload).hexdigest()


def load_patient_splits(
    train_path: Path,
    validation_path: Path,
    available_patient_ids: Collection[str],
) -> PatientSplits:
    train_ids, train_hash = _load_split(train_path, "train")
    validation_ids, validation_hash = _load_split(validation_path, "validation")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise SplitError("train and validation patient splits overlap")
    unknown = (set(train_ids) | set(validation_ids)) - set(available_patient_ids)
    if unknown:
        raise SplitError("patient split contains unknown patient IDs")
    return PatientSplits(train_ids, validation_ids, train_hash, validation_hash)
