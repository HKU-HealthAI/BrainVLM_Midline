#!/usr/bin/env python3
"""Extract BrainVLM-Midline slice logits and hidden features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from midline_agent.datasets.annotations import (
    MODALITIES,
    AnnotationError,
    AnnotationRecord,
    load_annotations,
)
from midline_agent.datasets.brainvlm import PREPROCESSING_VERSION
from midline_agent.datasets.inference_annotations import (
    InferenceCase,
    InferenceSlice,
    load_inference_case,
)
from midline_agent.datasets.prompts import PROMPT_VERSION, TUMOR_LABELS
from midline_agent.diagnostic.artifact_io import sha256_file
from midline_agent.diagnostic.brainvlm_runtime import BrainVlmRuntime
from midline_agent.diagnostic.feature_store import save_feature_store


MODALITY_MAP = {name: index for index, name in enumerate(MODALITIES)}
_LABELED_REQUIRED_FIELDS = {"patient_id", "image_path", "modality", "slice_index", "tumor_type"}
_LABELED_OPTIONAL_FIELDS = {"who_grade", "keyframe_label", "report_text"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--biomedclip-pretrained")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labeled", action="store_true")
    return parser


def _read_labeled_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"unable to read annotation JSON: {Path(path).name}") from exc
    if not isinstance(records, list) or not records:
        raise AnnotationError("labeled annotations must contain a non-empty list")
    if not all(isinstance(record, dict) for record in records):
        raise AnnotationError("each labeled annotation record must be an object")
    return records


def _validate_labeled_source(records: list[dict[str, Any]]) -> None:
    allowed_fields = _LABELED_REQUIRED_FIELDS | _LABELED_OPTIONAL_FIELDS
    for index, record in enumerate(records):
        missing = _LABELED_REQUIRED_FIELDS - record.keys()
        if missing:
            raise AnnotationError(f"record {index} missing fields: {', '.join(sorted(missing))}")
        unexpected = record.keys() - allowed_fields
        if unexpected:
            raise AnnotationError(f"record {index} has unexpected fields: {', '.join(sorted(unexpected))}")
        patient_id = record["patient_id"]
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise AnnotationError(f"record {index} patient_id must be a non-empty string")
        raw_image_path = record["image_path"]
        if not isinstance(raw_image_path, str) or not raw_image_path.strip():
            raise AnnotationError(f"record {index} image_path must be a non-empty string")
        if Path(raw_image_path).is_absolute():
            raise AnnotationError(f"record {index} image_path must be relative")
        slice_index = record["slice_index"]
        if isinstance(slice_index, bool) or not isinstance(slice_index, int):
            raise AnnotationError(f"record {index} slice_index must be an integer")
        if slice_index < 0:
            raise AnnotationError(f"record {index} slice_index must be non-negative")
        if record["tumor_type"] not in TUMOR_LABELS:
            raise AnnotationError(f"record {index} tumor_type must be a canonical tumor label")
        keyframe_label = record.get("keyframe_label")
        if keyframe_label is not None and (
            type(keyframe_label) is not int or keyframe_label not in (0, 1)
        ):
            raise AnnotationError(f"record {index} keyframe_label must be integer 0 or 1")


def _validate_labeled_boundary(
    raw_records: list[dict[str, Any]],
    records: list[AnnotationRecord],
    image_root: Path,
) -> tuple[InferenceCase, int, tuple[int | None, ...]]:
    if len(raw_records) != len(records):
        raise AnnotationError("labeled annotation records could not be aligned")
    patient_ids: set[str] = set()
    labels: set[str] = set()
    identities: set[tuple[str, int]] = set()
    inference_slices: list[InferenceSlice] = []
    keyframe_labels: list[int | None] = []
    resolved_root = Path(image_root).resolve()

    for index, (raw, record) in enumerate(zip(raw_records, records, strict=True)):
        patient_id = raw["patient_id"]
        image_path = Path(record.image_path).resolve()
        try:
            image_path.relative_to(resolved_root)
        except ValueError as exc:
            raise AnnotationError(f"record {index} image_path escapes image root") from exc
        if not image_path.is_file():
            raise AnnotationError(f"record {index} image does not exist")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise AnnotationError(f"record {index} image cannot be decoded") from exc
        identity = (record.modality, record.slice_index)
        if identity in identities:
            raise AnnotationError(f"record {index} has a duplicate modality and slice_index")
        identities.add(identity)
        patient_ids.add(patient_id)
        labels.add(raw["tumor_type"])
        keyframe_label = raw.get("keyframe_label")
        keyframe_labels.append(keyframe_label)
        inference_slices.append(
            InferenceSlice(
                patient_id=patient_id,
                image_path=image_path,
                modality=record.modality,
                slice_index=record.slice_index,
            )
        )

    if len(patient_ids) != 1:
        raise AnnotationError("labeled annotations must contain exactly one patient")
    if len(labels) != 1:
        raise AnnotationError("all labeled slices must use one consistent tumor label")
    ordering = {name: index for index, name in enumerate(MODALITIES)}
    ordered_positions = sorted(
        range(len(inference_slices)),
        key=lambda position: (
            ordering[inference_slices[position].modality],
            inference_slices[position].slice_index,
        ),
    )
    ordered_slices = tuple(inference_slices[position] for position in ordered_positions)
    ordered_keyframes = tuple(keyframe_labels[position] for position in ordered_positions)
    label = TUMOR_LABELS.index(next(iter(labels)))
    return InferenceCase(next(iter(patient_ids)), ordered_slices), label, ordered_keyframes


def _load_case(
    annotations: Path,
    image_root: Path,
    labeled: bool,
) -> tuple[InferenceCase, int | None, tuple[int | None, ...]]:
    if not labeled:
        case = load_inference_case(annotations, image_root)
        return case, None, (None,) * len(case.slices)
    raw_records = _read_labeled_records(annotations)
    _validate_labeled_source(raw_records)
    records = load_annotations(annotations, image_root)
    return _validate_labeled_boundary(raw_records, records, image_root)


def _extract_store(
    case: InferenceCase,
    runtime: BrainVlmRuntime,
    checkpoint: Path,
    label: int | None,
    keyframe_labels: tuple[int | None, ...],
) -> dict[str, object]:
    logits: list[torch.Tensor] = []
    hidden: list[torch.Tensor] = []
    slice_ids: list[str] = []
    modalities: list[int] = []
    slice_metadata: list[dict[str, object]] = []
    for item, keyframe_label in zip(case.slices, keyframe_labels, strict=True):
        features = runtime.predict_slice_features(item.image_path, item)
        logits.append(features.logits)
        hidden.append(features.hidden)
        slice_ids.append(f"{case.patient_id}:{item.modality}:{item.slice_index}")
        modalities.append(MODALITY_MAP[item.modality])
        metadata: dict[str, object] = {
            "modality": item.modality,
            "slice_index": item.slice_index,
            "checksum": sha256_file(item.image_path),
        }
        if keyframe_label is not None:
            metadata["keyframe_label"] = keyframe_label
        slice_metadata.append(metadata)
    patient: dict[str, object] = {
        "slice_ids": slice_ids,
        "modalities": modalities,
        "logits": torch.stack(logits).to(dtype=torch.float32),
        "hidden": torch.stack(hidden).to(dtype=torch.float32),
        "valid_mask": [True] * len(case.slices),
        "slice_metadata": slice_metadata,
    }
    if label is not None:
        patient["label"] = label
    return {
        "metadata": {
            "schema_version": "1.0",
            "feature_kind": "hidden_and_logits",
            "logit_dimension": 5,
            "hidden_dimension": 4096,
            "modality_map": MODALITY_MAP,
            "checkpoint_sha256": sha256_file(checkpoint),
            "prompt_version": PROMPT_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        "patients": {case.patient_id: patient},
    }


def main() -> int:
    args = _parser().parse_args()
    case, label, keyframe_labels = _load_case(args.annotations, args.image_root, args.labeled)
    runtime = BrainVlmRuntime.load(
        args.llama_model,
        args.checkpoint,
        device=args.device,
        biomedclip_pretrained=args.biomedclip_pretrained,
    )
    store = _extract_store(case, runtime, args.checkpoint, label, keyframe_labels)
    save_feature_store(args.output, store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
