"""Narrow contracts used by the BrainVLM diagnostic adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..datasets.prompts import PROMPT_VERSION


@dataclass(frozen=True)
class WhoGradeRequest:
    slice_id: str
    image: np.ndarray
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class WhoGradeSliceResult:
    raw_text: str
    logits: np.ndarray
    prompt_version: str
    vocabulary_version: str = "tumor-v1"


class DiagnosticModelAdapter(Protocol):
    def predict_who_grade(self, request: WhoGradeRequest) -> WhoGradeSliceResult: ...
