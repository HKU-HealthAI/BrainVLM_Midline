import inspect

from midline_agent.orchestrator import MidlineOrchestrator
from midline_agent.schema import CaseInput, DiagnosticResult, StepStatus


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


def test_orchestrator_continues_after_survival_placeholder() -> None:
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
