from midline_agent.schema import CaseInput, DiagnosticResult
from midline_agent.state import CaseState
from midline_agent.treatment.agent import TreatmentAgent


class StructuredReportClient:
    def generate(self, request):
        del request
        return type(
            "Response",
            (),
            {
                "text": "",
                "parsed": {
                    "summary": "Use multidisciplinary review.",
                    "recommendations": [],
                    "uncertainty": "Diagnosis is unavailable.",
                    "evidence": [],
                },
            },
        )()


def test_treatment_does_not_infer_missing_diagnosis() -> None:
    state = CaseState.from_input(CaseInput(case_id="synthetic-1", clinical_facts={"age_years": 44})).with_diagnostic(
        DiagnosticResult.unavailable("no compatible checkpoint")
    )

    result = TreatmentAgent(StructuredReportClient()).run(state)

    assert "Diagnosis unavailable" in result.markdown
    assert "research decision support" in result.markdown.lower()


def test_treatment_rejects_unstructured_model_text() -> None:
    class UnstructuredClient:
        def generate(self, request):
            del request
            return type("Response", (), {"text": "unstructured output", "parsed": None})()

    result = TreatmentAgent(UnstructuredClient()).run(
        CaseState.from_input(CaseInput(case_id="opaque-case"))
    )

    assert result.status.state in {"failed", "unavailable"}
    assert "unstructured output" not in result.markdown


def test_treatment_prompt_does_not_send_raw_evidence_url_or_excerpt() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.request = None

        def generate(self, request):
            self.request = request
            return type(
                "Response",
                (),
                {
                    "text": "",
                    "parsed": {
                        "summary": "Review",
                        "recommendations": [],
                        "uncertainty": "None",
                        "evidence": ["guideline-1"],
                    },
                },
            )()

    from midline_agent.schema import EvidenceItem
    from midline_agent.treatment.providers import StaticEvidenceProvider

    client = RecordingClient()
    TreatmentAgent(
        client,
        (StaticEvidenceProvider(
            "skills",
            (EvidenceItem("guideline-1", "guideline", "RAW_EXCERPT_SENTINEL", "https://example.org/source"),),
        ),),
    ).run(CaseState.from_input(CaseInput(case_id="opaque-case")))

    assert "RAW_EXCERPT_SENTINEL" not in client.request.user
    assert "https://example.org/source" not in client.request.user
