"""Versioned BrainVLM task prompts and annotation label validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .annotations import AnnotationRecord


class LabelError(ValueError):
    """Raised when a task label is missing or outside the public vocabulary."""


KEYFRAME_TASK: Final = "keyframe"
TUMOR_TASK: Final = "tumor_classification"
COMBINED_TASK: Final = "tumor_who_grade"
PROMPT_VERSION: Final = "brainvlm-prompts-v1"
SUPPORTED_TASKS: Final = frozenset({KEYFRAME_TASK, TUMOR_TASK, COMBINED_TASK})

KEYFRAME_LABELS: Final = ("non_keyframe", "keyframe")
TUMOR_LABELS: Final = tuple(f"tumor_{index}" for index in range(1, 6))
WHO_LABELS: Final = ("grade_2", "grade_3", "grade_4")
ALLOWED_TUMOR_GRADES: Final = {
    "tumor_1": frozenset({"grade_4"}),
    "tumor_2": frozenset({"grade_2", "grade_3", "grade_4"}),
    "tumor_3": frozenset({"grade_4"}),
    "tumor_4": frozenset({"grade_2", "grade_3", "grade_4"}),
    "tumor_5": frozenset({"grade_2", "grade_3"}),
}


@dataclass(frozen=True)
class TaskLabel:
    """A validated target label for one BrainVLM task."""

    task: str
    value: str


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelError(f"{field} must be a non-empty string")
    return value.strip()


def validate_task_label(task: str, value: object) -> str:
    """Validate and normalize a label without inventing a missing default."""
    if task == KEYFRAME_TASK:
        if type(value) is not int or value not in (0, 1):
            raise LabelError("keyframe label must be integer 0 or 1")
        return KEYFRAME_LABELS[value]
    if task == TUMOR_TASK:
        label = _require_text(value, "tumor label")
        if label not in TUMOR_LABELS:
            raise LabelError("tumor label is outside the supported vocabulary")
        return label
    if task == COMBINED_TASK:
        combined = _require_text(value, "combined tumor/WHO label")
        parts = combined.split("|")
        if len(parts) != 2:
            raise LabelError("combined tumor/WHO label must be tumor_X|grade_Y")
        tumor_type, grade = parts
        if tumor_type not in TUMOR_LABELS or grade not in WHO_LABELS:
            raise LabelError("combined tumor/WHO label is outside the supported vocabulary")
        if grade not in ALLOWED_TUMOR_GRADES[tumor_type]:
            raise LabelError("combined tumor/WHO label is not a valid tumor-grade pair")
        return f"{tumor_type}|{grade}"
    raise LabelError(f"unsupported BrainVLM task: {task}")


def label_for_record(record: AnnotationRecord, task: str) -> str:
    """Extract a required training label from a canonical annotation record."""
    if task == KEYFRAME_TASK:
        if record.keyframe_label is None:
            raise LabelError("keyframe_label is required for the keyframe task")
        return validate_task_label(task, record.keyframe_label)
    if task == TUMOR_TASK:
        return validate_task_label(task, record.tumor_type)
    if task == COMBINED_TASK:
        if record.who_grade is None:
            raise LabelError("who_grade is required for the combined tumor/WHO task")
        return validate_task_label(task, f"{record.tumor_type}|{record.who_grade}")
    raise LabelError(f"unsupported BrainVLM task: {task}")


def build_task_prompt(task: str, *, label: object | None = None) -> str:
    """Return a deterministic instruction for a supported model task."""
    if task == KEYFRAME_TASK:
        if label is not None:
            if type(label) is int:
                validate_task_label(task, label)
            elif label not in KEYFRAME_LABELS:
                raise LabelError("keyframe label must be integer 0 or 1 or a canonical keyframe label")
        return (
            f"[{PROMPT_VERSION}] Determine whether this MRI slice is the selected keyframe. "
            "Return exactly one label: keyframe or non_keyframe."
        )
    if task == TUMOR_TASK:
        if label is not None:
            validate_task_label(task, label)
        return (
            f"[{PROMPT_VERSION}] Classify the MRI slice into one of tumor_1, tumor_2, "
            "tumor_3, tumor_4, or tumor_5. Return exactly one tumor label."
        )
    if task == COMBINED_TASK:
        if label is not None:
            validate_task_label(task, label)
        return (
            f"[{PROMPT_VERSION}] Classify the MRI slice and WHO grade. Return exactly "
            "the constrained pair <tumor_X> <grade_Y>. WHO grades are grade_2, "
            "grade_3, or grade_4; only valid tumor-grade pairs are accepted."
        )
    raise LabelError(f"unsupported BrainVLM task: {task}")


def combined_label_from_tokens(tumor_type: str, grade: str) -> str:
    """Build a validated canonical combined label from generated tokens."""
    return validate_task_label(COMBINED_TASK, f"{tumor_type}|{grade}")


def validate_generated_label(task: str, value: object) -> str:
    """Validate a model-emitted string label using the same fixed vocabulary."""
    text = _require_text(value, "generated label")
    if task == KEYFRAME_TASK:
        if text not in KEYFRAME_LABELS:
            raise LabelError("generated keyframe label is outside the supported vocabulary")
        return text
    if task == TUMOR_TASK:
        return validate_task_label(task, text)
    if task == COMBINED_TASK:
        if text.startswith("<") and "> <" in text and text.endswith(">"):
            tumor_type, grade = text[1:-1].split("> <", maxsplit=1)
            text = f"{tumor_type}|{grade}"
        return validate_task_label(task, text)
    raise LabelError(f"unsupported BrainVLM task: {task}")
