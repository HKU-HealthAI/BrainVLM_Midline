"""Canonical prompts and output labels for BrainVLM-Midline tasks."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping


TUMOR_LABELS = (
    "<tumor_1>",
    "<tumor_2>",
    "<tumor_3>",
    "<tumor_4>",
    "<tumor_5>",
)
GRADE_LABELS = ("<grade_2>", "<grade_3>", "<grade_4>")
ALLOWED_GRADES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "<tumor_1>": ("<grade_4>",),
        "<tumor_2>": ("<grade_2>", "<grade_3>", "<grade_4>"),
        "<tumor_3>": ("<grade_4>",),
        "<tumor_4>": ("<grade_2>", "<grade_3>", "<grade_4>"),
        "<tumor_5>": ("<grade_2>", "<grade_3>"),
    }
)

TUMOR_TRAINING_INSTRUCTIONS = (
    "Please predict the WHO classification for this patient based on the MRI image.",
    "Based on the MRI image, predict the WHO classification.",
    "What is the WHO classification for this patient?",
)
WHO_TRAINING_INSTRUCTIONS = (
    "Please classify both the WHO tumor type and WHO grade for this midline glioma patient based on the MRI image.",
    "Based on the MRI image, determine both the tumor classification and grade for this case.",
    "Analyze this midline glioma case and provide both the WHO tumor type and WHO grade classification.",
    "From the MRI scan, classify both the tumor type (tumor_1 to tumor_5) and the corresponding WHO grade.",
)

_TUMOR_PREFIX = (
    "You are a language vision assistant. Based on the MRI image<ImageHere>, question, and context, "
    "please give an appropriate tumor choice among WHO tumor classification labels. USER:"
)
_TUMOR_CONTEXT = (
    " Context:This is WHO tumor classification label: "
    "'H3K27M-mutated midline glioma': '<tumor_1>', "
    "'Pediatric-type diffuse glioma': '<tumor_2>', "
    "'IDH-wildtype glioblastoma': '<tumor_3>', "
    "'IDH-mutated astrocytoma': '<tumor_4>', "
    "'IDH-mutated oligodendroglioma': '<tumor_5>' ASSISTANT:"
)
_WHO_PREFIX = (
    "You are a language vision assistant. Based on the MRI image<ImageHere>, please classify both the "
    "tumor type and WHO grade. USER:"
)
_WHO_CONTEXT = (
    " Context:WHO tumor classification labels: '<tumor_1>' for H3K27M-mutated midline glioma "
    "(always <grade_4>), '<tumor_2>' for Pediatric-type diffuse glioma (can be <grade_2> or "
    "<grade_3> or <grade_4>), '<tumor_3>' for IDH-wildtype glioblastoma (always <grade_4>), "
    "'<tumor_4>' for IDH-mutated astrocytoma (can be <grade_2> or <grade_3> or <grade_4>), "
    "'<tumor_5>' for IDH-mutated oligodendroglioma (can be <grade_2> or <grade_3>). "
    "Your output format: '<tumor_X> <grade_Y>' ASSISTANT:"
)


def age_context(age: float | None) -> str:
    if age is None:
        return ""
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        raise ValueError("age must be a finite non-negative number")
    value = float(age)
    if not math.isfinite(value) or value < 0:
        raise ValueError("age must be a finite non-negative number")
    category = "child" if value <= 18 else "adult"
    return f"This is a {category} patient. "


def build_tumor_training_prompt(instruction: str, age: float | None = None) -> str:
    if instruction not in TUMOR_TRAINING_INSTRUCTIONS:
        raise ValueError("unsupported tumor instruction")
    return f"{_TUMOR_PREFIX}{age_context(age)}{instruction}{_TUMOR_CONTEXT}"


def build_tumor_inference_prompt(age: float | None = None) -> str:
    return (
        build_tumor_training_prompt(TUMOR_TRAINING_INSTRUCTIONS[0], age)
        + "<tumor_"
    )


def build_who_training_prompt(instruction: str, age: float | None = None) -> str:
    if instruction not in WHO_TRAINING_INSTRUCTIONS:
        raise ValueError("unsupported WHO instruction")
    return f"{_WHO_PREFIX}{age_context(age)}{instruction}{_WHO_CONTEXT}"


def validate_tumor_grade(tumor_label: str, grade_label: str) -> None:
    if tumor_label not in TUMOR_LABELS:
        raise ValueError("unsupported tumor label")
    if grade_label not in GRADE_LABELS:
        raise ValueError("unsupported grade label")
    if grade_label not in ALLOWED_GRADES[tumor_label]:
        raise ValueError("grade label is not valid for the tumor label")
