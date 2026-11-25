"""Prompt and schema definitions for Treatment and orchestration calls."""

from __future__ import annotations

from typing import Any


HANDOFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proceed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["proceed", "reason"],
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "recommendations", "uncertainty", "evidence"],
    "additionalProperties": False,
}


WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["title", "url", "excerpt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sources"],
    "additionalProperties": False,
}


REPORT_SYSTEM_PROMPT = (
    "You are a careful neuro-oncology report writer. Use only supplied clinical facts and evidence. "
    "Do not invent citations, trials, or patient facts. Return the requested structured object."
)


def handoff_prompt(*, diagnostic_state: str, survival_state: str) -> str:
    return (
        "The Python workflow has completed the Diagnostic and Survival steps. "
        "Confirm whether the Treatment Agent may proceed from their validated states. "
        "Do not alter either agent result.\n"
        f"Diagnostic state: {diagnostic_state}\nSurvival state: {survival_state}"
    )


def report_prompt(*, treatment_input: Any, evidence: tuple[Any, ...]) -> str:
    return (
        "Create a research decision-support report. Include only source IDs from the supplied evidence list. "
        "If no evidence is supplied, state that verified evidence was unavailable.\n"
        f"Treatment input: {treatment_input}\nEvidence: {evidence}"
    )
