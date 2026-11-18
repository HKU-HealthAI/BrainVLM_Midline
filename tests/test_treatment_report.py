from __future__ import annotations

import json

import pytest

from midline_agent.schema import EvidenceItem
from midline_agent.treatment.report import render_structured_report, validate_report


def test_report_rejects_fabricated_source_ids() -> None:
    evidence = (EvidenceItem("guideline-1", "guideline", excerpt="verified"),)
    report = {"summary": "Review", "recommendations": [], "uncertainty": "None", "evidence": ["made-up"]}

    with pytest.raises(ValueError, match="unknown evidence source"):
        validate_report(report, evidence)


def test_report_renders_only_validated_citations() -> None:
    evidence = (EvidenceItem("guideline-1", "guideline", excerpt="verified"),)
    report = {
        "summary": "Review",
        "recommendations": ["Discuss at a multidisciplinary conference."],
        "uncertainty": "Diagnosis is unavailable.",
        "evidence": ["guideline-1"],
    }

    markdown = render_structured_report(validate_report(report, evidence), evidence)

    assert "guideline-1" in markdown
    assert "made-up" not in markdown


def test_no_evidence_report_has_explicit_fallback() -> None:
    report = {"summary": "Review", "recommendations": [], "uncertainty": "Unknown", "evidence": []}

    markdown = render_structured_report(validate_report(report, ()), ())

    assert "No verified evidence was available" in markdown


def test_report_renders_separate_evidence_metadata_sections() -> None:
    evidence = (
        EvidenceItem(
            "guideline-1",
            "guideline",
            excerpt="verified",
            url="https://example.org/guideline",
            source_name="Guideline source",
            retrieval_status="verified",
            applicability="Relevant to the supplied tumor context.",
            limitations="Local policy may differ.",
        ),
    )
    report = {"summary": "Review", "recommendations": [], "uncertainty": "None", "evidence": ["guideline-1"]}

    markdown = render_structured_report(validate_report(report, evidence), evidence)

    assert "### Evidence: guideline-1" in markdown
    assert "Source: Guideline source" in markdown
    assert "URL: https://example.org/guideline" in markdown
    assert "Applicability:" in markdown
    assert "Limitations:" in markdown


def test_report_escapes_model_text_and_evidence_metadata_as_plain_markdown() -> None:
    evidence = (
        EvidenceItem(
            "source-1",
            "guideline",
            source_name="Injected\n### Heading [link](https://untrusted.invalid)",
        ),
    )
    report = {
        "summary": "Review <script>alert(1)</script>",
        "recommendations": ["Discuss **carefully**"],
        "uncertainty": "Line one\n### Injected heading",
        "evidence": ["source-1"],
    }

    markdown = render_structured_report(validate_report(report, evidence), evidence)

    assert "<script>" not in markdown
    assert "\n### Injected heading" not in markdown
    assert "\n### Heading" not in markdown
    assert r"\*\*carefully\*\*" in markdown
    assert r"\[link\]\(https://untrusted.invalid\)" in markdown


@pytest.mark.parametrize(
    "fabricated_reference",
    (
        "See https://unverified.invalid/trial for details.",
        "Consider trial NCT12345678.",
        "Review www.unverified.invalid before treatment.",
    ),
)
def test_report_rejects_unverified_references_in_free_text(fabricated_reference: str) -> None:
    report = {
        "summary": fabricated_reference,
        "recommendations": [],
        "uncertainty": "Unknown",
        "evidence": [],
    }

    with pytest.raises(ValueError, match="unverified external reference"):
        validate_report(report, ())
