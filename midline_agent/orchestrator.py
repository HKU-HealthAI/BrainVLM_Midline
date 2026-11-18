"""Hybrid Python/LLM orchestration for the public workflow."""

from __future__ import annotations

from typing import Any

from .schema import CaseInput, DiagnosticResult, StepStatus, TreatmentResult
from .state import CaseState
from .treatment.agent import TreatmentAgent
from .treatment.client import LlmClient, LlmRequest
from .treatment.prompts import HANDOFF_SCHEMA, handoff_prompt
from .treatment.providers import EvidenceProvider


class MidlineOrchestrator:
    """Enforce step order in Python while using validated LLM handoffs."""

    def __init__(
        self,
        llm_client: LlmClient,
        *,
        diagnostic_workflow: Any | None = None,
        treatment_providers: tuple[EvidenceProvider, ...] = (),
        provider_timeout_seconds: float = 2.0,
    ) -> None:
        self.llm_client = llm_client
        self.diagnostic_workflow = diagnostic_workflow
        self.treatment_agent = TreatmentAgent(
            llm_client,
            treatment_providers,
            provider_timeout_seconds=provider_timeout_seconds,
        )

    def _run_diagnostic(self, case_input: CaseInput) -> DiagnosticResult:
        workflow = self.diagnostic_workflow
        if workflow is None:
            return DiagnosticResult.unavailable("no external diagnostic adapter configured")
        try:
            if callable(workflow):
                result = workflow(case_input)
            else:
                result = workflow.run(case_input)
            if not isinstance(result, DiagnosticResult):
                raise TypeError("diagnostic workflow must return DiagnosticResult")
            return result
        except Exception:
            return DiagnosticResult(status=StepStatus("failed", "diagnostic step failed"))

    def _request_handoff(self, state: CaseState) -> bool:
        response = self.llm_client.generate(
            LlmRequest(
                system=(
                    "You are a workflow coordinator. Validate the supplied step statuses and return a "
                    "structured handoff. Never alter any diagnostic or survival result."
                ),
                user=handoff_prompt(
                    diagnostic_state=state.diagnostic.status.state,
                    survival_state=state.survival.status.state,
                ),
                response_schema=HANDOFF_SCHEMA,
                schema_name="midline_handoff",
            )
        )
        if response.parsed is None:
            # Offline compatibility clients may only return plain text. The
            # Python state machine still enforces the same order.
            return True
        if not isinstance(response.parsed, dict):
            raise ValueError("handoff must be a JSON object")
        if not isinstance(response.parsed.get("proceed"), bool) or not isinstance(
            response.parsed.get("reason"), str
        ):
            raise ValueError("handoff has an invalid schema")
        return bool(response.parsed["proceed"])

    def run(
        self,
        case_input: CaseInput,
    ) -> CaseState:
        state = CaseState.from_input(case_input)
        state = state.with_diagnostic(self._run_diagnostic(case_input))
        if state.diagnostic.status.state == "failed":
            return state.with_treatment(
                TreatmentResult(
                    status=StepStatus(
                        "failed",
                        "Treatment was skipped because the Diagnostic step failed",
                    )
                )
            )
        try:
            if not self._request_handoff(state):
                return state.with_treatment(
                    TreatmentResult(
                        status=StepStatus(
                            "unavailable",
                            "Treatment step was not authorized by the workflow handoff",
                        ),
                        provenance={"handoff_state": "declined"},
                    )
                )
            treatment_result = self.treatment_agent.run(state)
        except Exception:
            treatment_result = TreatmentResult(status=StepStatus("failed", "Treatment orchestration failed"))
        return state.with_treatment(treatment_result)
