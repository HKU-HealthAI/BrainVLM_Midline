"""Minimal deterministic preprocessing for canonical MRI slice records."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .annotations import AnnotationRecord, MODALITIES


def ordered_records(records: list[AnnotationRecord]) -> list[AnnotationRecord]:
    modality_order = {modality: index for index, modality in enumerate(MODALITIES)}
    return sorted(records, key=lambda record: (modality_order[record.modality], record.slice_index))


def evaluation_image_array(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert a 2D/3D numerical image to deterministic RGB CHW float32 data."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        minimum, maximum = float(array.min()), float(array.max())
        scaled = np.zeros_like(array) if maximum == minimum else (array - minimum) / (maximum - minimum)
        array = np.repeat(scaled[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 3:
        array = np.clip(array, 0, 1) if array.max() <= 1 else array / 255.0
    else:
        raise ValueError("expected a 2D array or RGB HWC array")
    resized = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
