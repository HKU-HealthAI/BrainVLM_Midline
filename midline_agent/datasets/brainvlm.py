"""Public BrainVLM request and deterministic MRI preprocessing contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .annotations import AnnotationRecord
from .mri_dataset import evaluation_image_array
from .prompts import (
    PROMPT_VERSION,
    SUPPORTED_TASKS,
    build_task_prompt,
    label_for_record,
    validate_generated_label,
)


PREPROCESSING_VERSION = "mri-rgb-bicubic-minmax-v1"


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PreprocessedSlice:
    """Immutable image tensor plus reproducibility metadata."""

    image: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32).copy()
        if image.shape[0:1] != (3,) or image.ndim != 3:
            raise ValueError("preprocessed image must have CHW shape with three channels")
        if not bool(np.isfinite(image).all()):
            raise ValueError("preprocessed image must contain only finite values")
        image.setflags(write=False)
        metadata = dict(self.metadata)
        metadata["output_sha256"] = _sha256_array(image)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class BrainVlmRequest:
    """Validated input passed to an external BrainVLM runner."""

    task: str
    prompt: str
    image: np.ndarray
    metadata: Mapping[str, Any]
    target_label: str | None = None
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or self.task not in SUPPORTED_TASKS:
            raise ValueError("task must be one of the supported BrainVLM tasks")
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError("prompt must be a non-empty string")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError("prompt_version must match the public BrainVLM prompt version")
        image = np.asarray(self.image, dtype=np.float32).copy()
        if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] <= 0 or image.shape[2] <= 0:
            raise ValueError("BrainVLM image must have positive CHW dimensions with three channels")
        if not bool(np.isfinite(image).all()):
            raise ValueError("BrainVLM image must contain only finite values")
        image.setflags(write=False)
        metadata = dict(self.metadata)
        if "preprocessing_version" not in metadata:
            raise ValueError("preprocessing metadata must include preprocessing_version")
        if "prompt_version" in metadata and metadata["prompt_version"] != self.prompt_version:
            raise ValueError("request metadata prompt_version does not match the request")
        metadata["prompt_version"] = self.prompt_version
        if self.target_label is not None:
            try:
                target_label = validate_generated_label(self.task, self.target_label)
            except ValueError as exc:
                raise ValueError("target label is invalid for the request task") from exc
            if target_label != self.target_label:
                raise ValueError("target label must use the canonical task vocabulary")
            object.__setattr__(self, "target_label", target_label)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def preprocess_slice(image: np.ndarray, *, size: int = 224) -> PreprocessedSlice:
    """Normalize and resize one MRI slice with deterministic metadata."""
    source = np.asarray(image)
    if source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[-1] != 3):
        raise ValueError("MRI slice must be a 2D array or RGB HWC array")
    if type(size) is not int or size <= 0:
        raise ValueError("preprocessing size must be a positive integer")
    try:
        with np.errstate(over="raise", invalid="raise"):
            source_float = np.asarray(source, dtype=np.float32)
    except (FloatingPointError, TypeError, ValueError) as exc:
        raise ValueError("MRI slice cannot be represented as finite float32 values") from exc
    if not bool(np.isfinite(source_float).all()):
        raise ValueError("MRI slice must contain only finite float32 values")
    output = evaluation_image_array(source_float, size=size)
    if source_float.ndim == 2:
        normalization = "grayscale_minmax"
    elif float(source_float.max()) <= 1.0:
        normalization = "rgb_clip_0_1"
    else:
        normalization = "rgb_scale_255"
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_shape": list(source_float.shape),
        "input_dtype": str(source_float.dtype),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "normalization": normalization,
        "resize": "bicubic",
        "input_sha256": _sha256_array(source_float),
    }
    return PreprocessedSlice(output, metadata)


def build_task_request(
    record: AnnotationRecord,
    image: np.ndarray,
    task: str,
    *,
    size: int = 224,
) -> BrainVlmRequest:
    """Build a labelled training request and reject incomplete annotations."""
    processed = preprocess_slice(image, size=size)
    target_label = label_for_record(record, task)
    metadata = dict(processed.metadata)
    metadata.update({"patient_id": record.patient_id, "modality": record.modality, "slice_index": record.slice_index})
    return BrainVlmRequest(
        task=task,
        prompt=build_task_prompt(task, label=target_label),
        image=processed.image,
        metadata=metadata,
        target_label=target_label,
    )


def build_inference_request(
    task: str,
    image: np.ndarray,
    *,
    size: int = 224,
) -> BrainVlmRequest:
    """Build an unlabelled inference request with no implicit target."""
    processed = preprocess_slice(image, size=size)
    return BrainVlmRequest(
        task=task,
        prompt=build_task_prompt(task),
        image=processed.image,
        metadata=processed.metadata,
    )
