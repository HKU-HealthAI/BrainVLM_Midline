from __future__ import annotations

import time

from midline_agent.schema import CaseInput
from midline_agent.state import CaseState
from midline_agent.treatment.providers import (
    EvidenceRequest,
    EvidenceCollector,
    OpenAiWebSearchProvider,
    StaticEvidenceProvider,
)
from midline_agent.schema import EvidenceItem


def _request() -> EvidenceRequest:
    state = CaseState.from_input(CaseInput(case_id="opaque-case"))
    return EvidenceRequest(treatment_input=state.to_treatment_input())


def test_provider_failure_is_isolated_and_source_ids_are_unique() -> None:
    class FailingProvider:
        name = "failing"

        def collect(self, request: EvidenceRequest) -> tuple[EvidenceItem, ...]:
            del request
            raise RuntimeError("private provider unavailable")

    collector = EvidenceCollector(
        (FailingProvider(), StaticEvidenceProvider("skills", (EvidenceItem("skill-1", "physician_skill"),)))
    )

    result = collector.collect(_request())

    assert result.evidence == (EvidenceItem("skill-1", "physician_skill"),)
    assert result.provider_errors == ("failing",)


def test_provider_deadline_does_not_block_other_providers() -> None:
    class SlowProvider:
        name = "slow"

        def collect(self, request: EvidenceRequest) -> tuple[EvidenceItem, ...]:
            del request
            time.sleep(0.05)
            return ()

    started = time.monotonic()
    result = EvidenceCollector((SlowProvider(),), timeout_seconds=0.001).collect(_request())

    assert time.monotonic() - started < 0.04
    assert result.provider_errors == ("slow",)


def test_provider_timeout_uses_daemon_worker() -> None:
    class SlowProvider:
        name = "slow"

        def collect(self, request: EvidenceRequest):
            del request
            time.sleep(0.1)
            return ()

    result = EvidenceCollector((SlowProvider(),), timeout_seconds=0.001).collect(_request())

    assert result.provider_errors == ("slow",)


def test_collector_rejects_lazy_provider_iterables_without_iterating_them() -> None:
    iteration_attempted = False

    class LazyProvider:
        name = "lazy"

        def collect(self, request: EvidenceRequest):
            del request

            def values():
                nonlocal iteration_attempted
                iteration_attempted = True
                yield EvidenceItem("lazy-1", "guideline")

            return values()

    result = EvidenceCollector((LazyProvider(),)).collect(_request())

    assert iteration_attempted is False
    assert result.evidence == ()
    assert result.provider_errors == ("lazy",)


def test_web_search_provider_uses_structured_client_results() -> None:
    class Client:
        def generate(self, request):
            assert request.tools == ({"type": "web_search"},)
            return type(
                "Response",
                (),
                {
                    "text": "",
                    "parsed": {
                        "sources": [
                            {
                                "title": "Guideline",
                                "url": "https://example.org/guideline",
                                "excerpt": "A short verified excerpt.",
                            }
                        ]
                    },
                },
            )()

    evidence = OpenAiWebSearchProvider(Client()).collect(_request())

    assert evidence[0].source_kind == "web_search"
    assert evidence[0].source_id.startswith("web-")
    assert evidence[0].url == "https://example.org/guideline"


def test_web_search_provider_rejects_urls_with_query_or_userinfo() -> None:
    class Client:
        def generate(self, request):
            del request
            return type(
                "Response",
                (),
                {
                    "text": "",
                    "parsed": {
                        "sources": [
                            {
                                "title": "Unsafe",
                                "url": "https://user:secret@example.org/path?token=secret#fragment",
                                "excerpt": "secret excerpt",
                            }
                        ]
                    },
                },
            )()

    assert OpenAiWebSearchProvider(Client()).collect(_request()) == ()


def test_collector_rejects_empty_source_ids() -> None:
    class MalformedProvider:
        name = "malformed"

        def collect(self, request: EvidenceRequest):
            del request
            return (EvidenceItem("", "guideline"),)

    import pytest

    with pytest.raises(ValueError, match="source_id"):
        EvidenceCollector((MalformedProvider(),)).collect(_request())
