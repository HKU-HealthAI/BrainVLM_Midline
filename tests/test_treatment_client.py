import json
import sys
import types

import pytest

from midline_agent.treatment.client import LlmRequest, OpenAiResponsesClient


def test_openai_client_requires_credentials() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAiResponsesClient(api_key="", model="gpt-test")


def test_openai_client_does_not_accept_custom_endpoint() -> None:
    removed_argument = "base" + "_url"
    with pytest.raises(TypeError):
        OpenAiResponsesClient(
            api_key="<test-key>",
            model="gpt-test",
            **{removed_argument: "https://endpoint.invalid"},
        )


def test_openai_client_uses_responses_structured_output_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        output_text = json.dumps({"proceed": True, "reason": "ready"})

    class FakeResponses:
        def create(self, **kwargs: object) -> FakeResponse:
            calls.append(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {"api_key": "<test-key>", "timeout": 12.0, "max_retries": 0}
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    response = OpenAiResponsesClient(api_key="<test-key>", model="gpt-test", timeout=12, retries=1).generate(
        LlmRequest(
            system="system",
            user="user",
            response_schema={
                "type": "object",
                "properties": {"proceed": {"type": "boolean"}, "reason": {"type": "string"}},
                "required": ["proceed", "reason"],
                "additionalProperties": False,
            },
            schema_name="handoff",
        )
    )

    assert response.parsed == {"proceed": True, "reason": "ready"}
    assert calls[0]["store"] is False
    assert calls[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "handoff",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"proceed": {"type": "boolean"}, "reason": {"type": "string"}},
                "required": ["proceed", "reason"],
                "additionalProperties": False,
            },
        },
    }


def test_openai_client_retries_within_one_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock_values = iter((0.0, 0.0, 0.4, 1.1))
    sleeps: list[float] = []
    attempts = 0

    class FakeResponses:
        def create(self, **kwargs: object):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("temporary")

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAiResponsesClient(
        api_key="<test-key>", model="gpt-test", timeout=1.0, retries=5, backoff=0.2,
        clock=lambda: next(clock_values), sleeper=sleeps.append,
    )

    with pytest.raises(RuntimeError, match="failed"):
        client.generate(LlmRequest(system="system", user="user"))

    assert attempts == 2
    assert sleeps == [0.2]
