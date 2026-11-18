"""Evidence-grounded Treatment Agent without bundled private sources."""

from __future__ import annotations

import json

from ..schema import StepStatus, TreatmentResult, to_jsonable
from ..state import CaseState
from .client import LlmClient, LlmRequest
from .prompts import REPORT_SCHEMA, REPORT_SYSTEM_PROMPT, report_prompt
from .providers import EvidenceCollector, EvidenceProvider, EvidenceRequest
from .report import render_structured_report, validate_report


RESEARCH_NOTICE = "This output is research decision support only and is not a treatment order."


class TreatmentAgent:
    def __init__(
        self,
        client: LlmClient,
        providers: tuple[EvidenceProvider, ...] = (),
        *,
        provider_timeout_seconds: float = 2.0,
    ) -> None:
        self.client = client
        self.collector = EvidenceCollector(providers, timeout_seconds=provider_timeout_seconds)

    def run(self, state: CaseState) -> TreatmentResult:
        treatment_input = state.to_treatment_input()
        collection = self.collector.collect(EvidenceRequest(treatment_input))
        evidence = collection.evidence
        diagnosis = treatment_input.diagnostic_result
        unavailable = diagnosis.status.state != "completed"
        diagnostic_context = (
            f"Diagnosis unavailable: {diagnosis.status.reason}. Use only the supplied clinical facts."
            if unavailable
            else f"Predicted diagnosis: {diagnosis.tumor_type}; predicted WHO grade: {diagnosis.who_grade}."
        )
        request = LlmRequest(
            system=REPORT_SYSTEM_PROMPT,
            user=(
                f"{diagnostic_context}\n"
                f"{report_prompt(treatment_input=to_jsonable(treatment_input), evidence=_safe_evidence_context(evidence))}\n"
                f"Clinical facts: {json.dumps(to_jsonable(treatment_input.clinical_facts), sort_keys=True)}"
            ),
            response_schema=REPORT_SCHEMA,
            schema_name="midline_treatment_report",
        )
        availability_notice = f"Diagnosis unavailable: {diagnosis.status.reason}\n\n" if unavailable else ""
        try:
            generated = self.client.generate(request)
            if generated.parsed is None:
                return TreatmentResult(
                    status=StepStatus("failed", "Treatment model returned no structured report"),
                    markdown=(
                        f"{RESEARCH_NOTICE}\n\n"
                        f"{availability_notice if unavailable else ''}"
                        "No verified evidence was available because the model response was not structured.\n"
                    ),
                    evidence=evidence,
                )
            validated = validate_report(generated.parsed, evidence)
            markdown = render_structured_report(validated, evidence)
        except Exception:
            return TreatmentResult(status=StepStatus("failed", "Treatment report generation failed"), evidence=evidence)
        return TreatmentResult(
            status=StepStatus("completed"),
            markdown=f"{RESEARCH_NOTICE}\n\n{availability_notice}{markdown}",
            evidence=evidence,
        )


def _safe_evidence_context(evidence: tuple) -> tuple[dict[str, str], ...]:
    """Expose only bounded metadata to the model, never URLs or raw excerpts."""
    allowed_kinds = {"guideline", "knowledge_base", "local_knowledge", "physician_skill", "similar_case", "web_search"}
    allowed_statuses = {"verified", "unverified", "unavailable", "rejected"}
    return tuple(
        {
            "source_id": item.source_id,
            "source_kind": item.source_kind if item.source_kind in allowed_kinds else "other",
            "retrieval_status": item.retrieval_status if item.retrieval_status in allowed_statuses else "unclassified",
            "applicability": "specified" if item.applicability != "unspecified" else "unspecified",
        }
        for item in evidence
    )
