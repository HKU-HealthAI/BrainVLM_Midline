"""Offline-testable clients for the official OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class LlmRequest:
    system: str
    user: str
    response_schema: Mapping[str, Any] | None = None
    schema_name: str = "midline_response"
    tools: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class LlmResponse:
    text: str
    parsed: Mapping[str, Any] | None = None


class LlmClient(Protocol):
    def generate(self, request: LlmRequest) -> LlmResponse: ...


class OpenAiResponsesClient:
    """Synchronous client restricted to the official OpenAI Responses API.

    The SDK is imported lazily so the repository remains usable for offline
    feature extraction and model training without the treatment extra.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        retries: int = 2,
        backoff: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if backoff < 0:
            raise ValueError("backoff must be non-negative")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.clock = clock
        self.sleeper = sleeper

    def generate(self, request: LlmRequest) -> LlmResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the treatment extra: pip install -e '.[treatment]'") from exc

        client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "store": False,
        }
        if request.tools:
            kwargs["tools"] = [dict(tool) for tool in request.tools]
        if request.response_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "strict": True,
                    "schema": dict(request.response_schema),
                }
            }

        deadline = self.clock() + self.timeout
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt > 0 and self.clock() >= deadline:
                raise RuntimeError("OpenAI Responses request failed") from last_error
            try:
                response = client.responses.create(**kwargs)
                text = str(getattr(response, "output_text", "") or "")
                parsed: Mapping[str, Any] | None = None
                if request.response_schema is not None:
                    candidate = json.loads(text)
                    if not isinstance(candidate, dict):
                        raise ValueError("structured Responses output must be a JSON object")
                    parsed = candidate
                return LlmResponse(text=text, parsed=parsed)
            except Exception as exc:  # SDK exceptions vary across supported versions.
                last_error = exc
                remaining = deadline - self.clock()
                if attempt >= self.retries or remaining <= 0:
                    raise RuntimeError("OpenAI Responses request failed") from exc
                delay = min(self.backoff * (2**attempt), remaining)
                if delay > 0:
                    self.sleeper(delay)
        raise RuntimeError("OpenAI Responses request failed") from last_error
