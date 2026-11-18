"""Validation and rendering for structured Treatment reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import re

from ..schema import EvidenceItem


UNVERIFIED_REFERENCE_PATTERN = re.compile(
    r"(?:https?://|www\.|\bNCT\d{8}\b|\b[A-Z]{2,8}-\d{2,8}\b)", re.IGNORECASE
)


def _validate_free_text(value: str) -> None:
    if UNVERIFIED_REFERENCE_PATTERN.search(value):
        raise ValueError("report free text contains an unverified external reference")


def _plain_markdown_text(value: str) -> str:
    """Render untrusted model/provider text without Markdown or HTML control."""
    collapsed = " ".join(value.split())
    escaped = html.escape(collapsed, quote=True)
    return re.sub(r"([\\`*_\[\]()])", r"\\\1", escaped)


def validate_report(report: Mapping[str, object], evidence: Sequence[EvidenceItem]) -> dict[str, object]:
    required = ("summary", "recommendations", "uncertainty", "evidence")
    if not isinstance(report, Mapping) or any(key not in report for key in required):
        raise ValueError("structured report is missing required fields")
    summary = report["summary"]
    recommendations = report["recommendations"]
    uncertainty = report["uncertainty"]
    citations = report["evidence"]
    if not isinstance(summary, str) or not isinstance(uncertainty, str):
        raise ValueError("report summary and uncertainty must be strings")
    if not isinstance(recommendations, Sequence) or isinstance(recommendations, (str, bytes)):
        raise ValueError("report recommendations must be a sequence")
    if not all(isinstance(item, str) for item in recommendations):
        raise ValueError("report recommendations must contain strings")
    for value in (summary, uncertainty, *recommendations):
        _validate_free_text(value)
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        raise ValueError("report evidence must be a sequence")
    if not all(isinstance(item, str) for item in citations):
        raise ValueError("report evidence must contain source IDs")
    available = {item.source_id for item in evidence}
    unknown = [item for item in citations if item not in available]
    if unknown:
        raise ValueError(f"unknown evidence source: {unknown[0]}")
    return {
        "summary": summary,
        "recommendations": tuple(recommendations),
        "uncertainty": uncertainty,
        "evidence": tuple(citations),
    }


def render_structured_report(report: Mapping[str, object], evidence: Sequence[EvidenceItem]) -> str:
    validated = validate_report(report, evidence)
    lines = ["## Guideline Suggestion", _plain_markdown_text(str(validated["summary"]))]
    recommendations = validated["recommendations"]
    if recommendations:
        lines.append("\n### Recommendations")
        lines.extend(f"- {_plain_markdown_text(str(item))}" for item in recommendations)
    lines.append(f"\n### Uncertainty\n{_plain_markdown_text(str(validated['uncertainty']))}")
    citations = validated["evidence"]
    if citations:
        by_id = {item.source_id: item for item in evidence}
        for source_id in citations:
            item = by_id[source_id]
            lines.extend(
                (
                    f"\n### Evidence: {source_id}",
                    f"- Kind: {_plain_markdown_text(item.source_kind)}",
                    f"- Source: {_plain_markdown_text(item.source_name or 'Unnamed supplied source')}",
                    f"- URL: {item.url or 'Not provided'}",
                    f"- Retrieval status: {_plain_markdown_text(item.retrieval_status)}",
                    f"- Applicability: {_plain_markdown_text(item.applicability)}",
                    f"- Limitations: {_plain_markdown_text(item.limitations or 'Not specified')}",
                )
            )
    else:
        lines.append("\nNo verified evidence was available for this report.")
    return "\n".join(lines).strip() + "\n"
