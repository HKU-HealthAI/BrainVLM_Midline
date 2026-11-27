"""Diagnostic Agent components."""

from .logits_mil import LogitsMilPredictor, LogitsMilResult, align_slice_logits
from .standalone import TumorPredictionService, WhoGradePredictionService

__all__ = [
    "LogitsMilPredictor",
    "LogitsMilResult",
    "TumorPredictionService",
    "WhoGradePredictionService",
    "align_slice_logits",
]
