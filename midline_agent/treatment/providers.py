"""Injected evidence-provider protocols with isolation and bounded waits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from queue import Queue
from threading import Thread
from typing import Any, Protocol
from urllib.parse import urlparse

from ..schema import EvidenceItem, TreatmentInput
from .client import LlmClient, LlmRequest
from .prompts import WEB_SEARCH_SCHEMA


@dataclass(frozen=True)
class EvidenceRequest:
    treatment_input: TreatmentInput


class EvidenceProvider(Protocol):
    name: str

    def collect(self, request: EvidenceRequest) -> tuple[EvidenceItem, ...]: ...


@dataclass(frozen=True)
class EvidenceCollection:
    evidence: tuple[EvidenceItem, ...]
    provider_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticEvidenceProvider:
    """Deployment-owned provider for physician skills or local knowledge.

    The public package ships no records; callers inject their own evidence.
    """

    name: str
    items: tuple[EvidenceItem, ...]

    def collect(self, request: EvidenceRequest) -> tuple[EvidenceItem, ...]:
        del request
        return self.items


class PhysicianSkillProvider(StaticEvidenceProvider):
    """Adapter for physician-authored skills supplied by a deployment."""

    name = "physician_skill"

    def __init__(self, items: tuple[EvidenceItem, ...]) -> None:
        super().__init__(self.name, tuple(items))


class LocalKnowledgeProvider(StaticEvidenceProvider):
    """Adapter for a user-owned local knowledge source."""

    name = "local_knowledge"

    def __init__(self, items: tuple[EvidenceItem, ...]) -> None:
        super().__init__(self.name, tuple(items))


class SimilarCaseProvider(StaticEvidenceProvider):
    """Adapter for user-owned similar-case summaries."""

    name = "similar_case"

    def __init__(self, items: tuple[EvidenceItem, ...]) -> None:
        super().__init__(self.name, tuple(items))


class EvidenceCollector:
    def __init__(self, providers: tuple[EvidenceProvider, ...] = (), *, timeout_seconds: float = 2.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.providers = tuple(providers)
        self.timeout_seconds = float(timeout_seconds)

    def collect(self, request: EvidenceRequest) -> EvidenceCollection:
        evidence: list[EvidenceItem] = []
        errors: list[str] = []
        for provider in self.providers:
            name = str(getattr(provider, "name", provider.__class__.__name__))
            outcome: Queue[tuple[bool, object]] = Queue(maxsize=1)

            def invoke() -> None:
                try:
                    outcome.put((True, provider.collect(request)))
                except BaseException as exc:
                    outcome.put((False, exc))

            worker = Thread(target=invoke, name=f"midline-evidence-{name}", daemon=True)
            worker.start()
            worker.join(self.timeout_seconds)
            if worker.is_alive():
                errors.append(name)
                continue
            succeeded, payload = outcome.get_nowait()
            if not succeeded:
                if isinstance(payload, ValueError):
                    raise payload
                errors.append(name)
                continue
            if not isinstance(payload, tuple):
                errors.append(name)
                continue
            try:
                for item in payload:
                    if not isinstance(item, EvidenceItem):
                        raise TypeError("providers must return EvidenceItem values")
                    if not item.source_id.strip() or not item.source_kind.strip():
                        raise ValueError("provider returned an empty evidence source_id or source_kind")
                    evidence.append(item)
            except ValueError:
                raise
            except Exception:
                errors.append(name)
        unique: dict[str, EvidenceItem] = {}
        for item in evidence:
            if item.source_id and item.source_id not in unique:
                unique[item.source_id] = item
        return EvidenceCollection(tuple(unique.values()), tuple(errors))


class OpenAiWebSearchProvider:
    """Deployment-time web search provider using the official OpenAI tool."""

    name = "official_web_search"

    def __init__(self, client: LlmClient) -> None:
        self.client = client

    def collect(self, request: EvidenceRequest) -> tuple[EvidenceItem, ...]:
        response = self.client.generate(
            LlmRequest(
                system="Find current authoritative sources relevant to the supplied treatment context.",
                user=(
                    "Return only sources that can be cited. Context: "
                    f"{request.treatment_input.clinical_facts}"
                ),
                response_schema=WEB_SEARCH_SCHEMA,
                schema_name="midline_web_sources",
                tools=({"type": "web_search"},),
            )
        )
        if not isinstance(response.parsed, dict):
            return ()
        result: list[EvidenceItem] = []
        for index, source in enumerate(response.parsed.get("sources", ()), start=1):
            if not isinstance(source, dict):
                continue
            url = source.get("url")
            title = source.get("title", "")
            excerpt = source.get("excerpt", "")
            parsed_url = urlparse(url) if isinstance(url, str) else None
            if (
                parsed_url is None
                or parsed_url.scheme != "https"
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or bool(parsed_url.query)
                or bool(parsed_url.fragment)
                or not isinstance(title, str)
                or not isinstance(excerpt, str)
            ):
                continue
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            result.append(
                EvidenceItem(
                    f"web-{index}-{digest}",
                    "web_search",
                    excerpt,
                    url,
                    source_name=title,
                    retrieval_status="verified",
                    applicability="Relevant to the treatment context returned by official web search.",
                    limitations="Search results require clinician review before use.",
                )
            )
        return tuple(result)
