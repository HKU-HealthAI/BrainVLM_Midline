"""Strict collation for BrainVLM-Midline task samples."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .datasets import DatasetContractError, TaskSample


def collate_task_samples(samples: Sequence[TaskSample]) -> dict[str, object]:
    if not samples:
        raise DatasetContractError("cannot collate an empty batch")
    shape = samples[0].image.shape
    task = samples[0].task
    for sample in samples:
        if not isinstance(sample, TaskSample):
            raise DatasetContractError("batch contains an unsupported sample")
        if sample.image.shape != shape or shape != (3, 224, 224):
            raise DatasetContractError("batch images must share shape [3, 224, 224]")
        if sample.image.dtype != torch.float32 or not bool(torch.isfinite(sample.image).all()):
            raise DatasetContractError("batch images must contain finite float32 values")
        if not sample.prompt or not sample.answer or not sample.image_name:
            raise DatasetContractError("batch samples require prompt, answer, and image name")
        if sample.task != task:
            raise DatasetContractError("a batch cannot mix tumor and WHO tasks")
    return {
        "image": torch.stack([sample.image for sample in samples]),
        "instruction_input": [sample.prompt for sample in samples],
        "answer": [sample.answer for sample in samples],
        "image_name": [sample.image_name for sample in samples],
        "task": task,
    }
