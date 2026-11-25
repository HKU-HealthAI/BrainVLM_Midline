import inspect

from midline_agent.orchestrator import MidlineOrchestrator
from midline_agent.schema import (
    CaseInput,
    DiagnosticResult,
    StepStatus,
    SurvivalResult,
)


class SuccessfulClient:
    def generate(self, request):
        if request.schema_name == "midline_handoff":
            parsed = {"proceed": True, "reason": "workflow state is valid"}
        else:
            parsed = {
                "summary": "Review the case.",
                "recommendations": [],
                "uncertainty": "No verified evidence was supplied.",
                "evidence": [],
            }
        return type("Response", (), {"text": "", "parsed": parsed})()


def test_orchestrator_run_has_no_diagnostic_bypass_parameters() -> None:
    parameters = inspect.signature(MidlineOrchestrator.run).parameters

    assert "diagnostic_result" not in parameters
    assert "use_mock_diagnostic" not in parameters


def test_orchestrator_continues_with_unconfigured_survival_fallback() -> None:
    workflow = lambda case: DiagnosticResult.unavailable("external checkpoint is not configured")
    state = MidlineOrchestrator(
        SuccessfulClient(),
        diagnostic_workflow=workflow,
    ).run(
        CaseInput(case_id="synthetic-1")
    )

    assert state.diagnostic.status.state == "unavailable"
    assert state.survival.status.state == "unavailable"
    assert state.treatment.status.state == "completed"


def test_orchestrator_runs_injected_diagnostic_workflow() -> None:
    diagnostic = DiagnosticResult(status=StepStatus("completed"), tumor_type="tumor_4", who_grade="grade_3")
    calls = []

    def workflow(case_input):
        calls.append(case_input.case_id)
        return diagnostic

    state = MidlineOrchestrator(
        SuccessfulClient(),
        diagnostic_workflow=workflow,
    ).run(
        CaseInput(case_id="synthetic-1")
    )

    assert calls == ["synthetic-1"]
    assert state.diagnostic is diagnostic


def test_orchestrator_runs_survival_between_diagnostic_and_treatment() -> None:
    calls = []
    diagnostic = DiagnosticResult(
        status=StepStatus("completed"),
        tumor_type="tumor_4",
        who_grade="grade_3",
    )
    survival = SurvivalResult(
        status=StepStatus("completed"),
        risk_score=1.25,
        median_survival_time=18.0,
        time_points=(6.0, 12.0, 24.0),
        survival_probabilities=(0.9, 0.7, 0.4),
    )

    def diagnostic_workflow(case_input):
        calls.append(("diagnostic", case_input.case_id))
        return diagnostic

    def survival_workflow(case_input, diagnostic_result):
        calls.append(("survival", case_input.case_id, diagnostic_result.tumor_type))
        return survival

    state = MidlineOrchestrator(
        SuccessfulClient(),
        diagnostic_workflow=diagnostic_workflow,
        survival_workflow=survival_workflow,
    ).run(CaseInput(case_id="synthetic-1"))

    assert calls == [
        ("diagnostic", "synthetic-1"),
        ("survival", "synthetic-1", "tumor_4"),
    ]
    assert state.survival is survival
    assert state.treatment.status.state == "completed"
    assert state.to_treatment_input().survival_result.median_survival_time == 18.0
    assert state.to_trace().steps[1].result["survival_probabilities"] == (0.9, 0.7, 0.4)


def test_orchestrator_stops_before_treatment_when_survival_fails() -> None:
    class Client:
        def generate(self, request):
            del request
            raise AssertionError("LLM must not run after a failed Survival step")

    def failed_survival(case_input, diagnostic_result):
        del case_input, diagnostic_result
        raise RuntimeError("survival checkpoint failed")

    state = MidlineOrchestrator(
        Client(),
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
        survival_workflow=failed_survival,
    ).run(CaseInput(case_id="opaque-case"))

    assert state.survival.status.state == "failed"
    assert state.treatment.status.state == "failed"
    assert state.to_trace().steps[1].status.state == "failed"


def test_orchestrator_rejects_pending_survival_result_before_treatment() -> None:
    class Client:
        def generate(self, request):
            del request
            raise AssertionError("LLM must not run with a pending Survival result")

    state = MidlineOrchestrator(
        Client(),
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
        survival_workflow=lambda case, diagnostic: SurvivalResult.pending(),
    ).run(CaseInput(case_id="opaque-case"))

    assert state.survival.status.state == "failed"
    assert state.treatment.status.state == "failed"


def test_orchestrator_requests_structured_handoff_before_treatment() -> None:
    class RecordingClient(SuccessfulClient):
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    client = RecordingClient()
    state = MidlineOrchestrator(
        client,
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
    ).run(CaseInput(case_id="opaque-case"))

    assert state.treatment.status.state == "completed"
    assert len(client.requests) == 2
    assert client.requests[0].schema_name == "midline_handoff"
    assert client.requests[0].response_schema is not None


def test_orchestrator_returns_failed_partial_state_on_client_error() -> None:
    class FailingClient:
        def generate(self, request):
            del request
            raise RuntimeError("API unavailable")

    state = MidlineOrchestrator(
        FailingClient(),
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
    ).run(
        CaseInput(case_id="opaque-case")
    )

    assert state.diagnostic.status.state == "unavailable"
    assert state.survival.status.state == "unavailable"
    assert state.treatment.status.state == "failed"
    assert state.to_trace().steps[-1].status.state == "failed"


def test_orchestrator_honors_declined_structured_handoff() -> None:
    class DecliningClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            assert request.schema_name == "midline_handoff"
            return type("Response", (), {"text": "", "parsed": {"proceed": False, "reason": "hold"}})()

    client = DecliningClient()
    state = MidlineOrchestrator(
        client,
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
    ).run(CaseInput(case_id="opaque-case"))

    assert client.calls == 1
    assert state.treatment.status.state == "unavailable"
    assert state.to_trace().steps[-1].result["provenance"]["handoff_state"] == "declined"


def test_orchestrator_stops_before_treatment_when_diagnostic_fails() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            raise AssertionError("LLM must not run after a failed diagnostic")

    failed = DiagnosticResult(status=StepStatus("failed", "adapter failed"))
    client = Client()
    state = MidlineOrchestrator(client, diagnostic_workflow=lambda case: failed).run(
        CaseInput(case_id="opaque-case")
    )

    assert client.calls == 0
    assert state.diagnostic.status.state == "failed"
    assert state.treatment.status.state == "failed"
    assert state.to_trace().steps[-1].status.state == "failed"
