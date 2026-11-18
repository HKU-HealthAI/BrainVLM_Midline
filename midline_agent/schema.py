"""Versioned, immutable data contracts for the public workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit


StepState = Literal["pending", "completed", "unavailable", "failed"]
StepName = Literal["diagnostic", "survival", "treatment"]
STEP_STATES = frozenset({"pending", "completed", "unavailable", "failed"})
TUMOR_TYPES = tuple(f"tumor_{index}" for index in range(1, 6))
WHO_GRADES = frozenset(f"grade_{index}" for index in range(2, 5))
OPAQUE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def freeze_json(value: Any) -> Any:
    """Copy a JSON-compatible value into recursively immutable containers."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract floats must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("contract mapping keys must be strings")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


@dataclass(frozen=True)
class StepStatus:
    state: StepState
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in STEP_STATES:
            raise ValueError(f"invalid step state: {self.state!r}")
        if not isinstance(self.reason, str):
            raise TypeError("step reason must be a string")


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    clinical_facts: Mapping[str, Any] = field(default_factory=dict)
    mri_manifest: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty opaque string")
        if not OPAQUE_ID_PATTERN.fullmatch(self.case_id):
            raise ValueError("case_id must be a bounded opaque identifier")
        if not isinstance(self.clinical_facts, Mapping):
            raise TypeError("clinical_facts must be a mapping")
        object.__setattr__(self, "clinical_facts", freeze_json(self.clinical_facts))
        frozen_manifest = freeze_json(self.mri_manifest)
        if not all(isinstance(item, Mapping) for item in frozen_manifest):
            raise TypeError("mri_manifest entries must be mappings")
        object.__setattr__(self, "mri_manifest", frozen_manifest)


@dataclass(frozen=True)
class DiagnosticResult:
    status: StepStatus
    tumor_type: str | None = None
    who_grade: str | None = None
    probabilities: tuple[float, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, StepStatus):
            raise TypeError("diagnostic status must be a StepStatus")
        if self.tumor_type is not None and not isinstance(self.tumor_type, str):
            raise TypeError("tumor_type must be a string or None")
        if self.who_grade is not None and not isinstance(self.who_grade, str):
            raise TypeError("who_grade must be a string or None")
        probabilities = tuple(float(value) for value in self.probabilities)
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("diagnostic probabilities must be finite")
        if self.status.state == "completed":
            if self.tumor_type not in TUMOR_TYPES or self.who_grade not in WHO_GRADES:
                raise ValueError(
                    "completed diagnostic requires valid tumor_type and who_grade"
                )
            if probabilities:
                if (
                    len(probabilities) != len(TUMOR_TYPES)
                    or any(value < 0.0 or value > 1.0 for value in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6)
                    or TUMOR_TYPES[max(range(len(probabilities)), key=probabilities.__getitem__)]
                    != self.tumor_type
                ):
                    raise ValueError(
                        "completed diagnostic probabilities must be normalized and match tumor_type"
                    )
        if not isinstance(self.provenance, Mapping):
            raise TypeError("diagnostic provenance must be a mapping")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "provenance", freeze_json(self.provenance))

    @classmethod
    def pending(cls) -> "DiagnosticResult":
        return cls(status=StepStatus("pending"))

    @classmethod
    def unavailable(cls, reason: str) -> "DiagnosticResult":
        return cls(status=StepStatus("unavailable", reason))


@dataclass(frozen=True)
class SurvivalResult:
    status: StepStatus

    def __post_init__(self) -> None:
        if not isinstance(self.status, StepStatus):
            raise TypeError("survival status must be a StepStatus")

    @classmethod
    def unavailable(cls, reason: str = "Survival Agent is not included in this release") -> "SurvivalResult":
        return cls(status=StepStatus("unavailable", reason))


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    source_kind: str
    excerpt: str = ""
    url: str | None = None
    source_name: str = ""
    retrieval_status: str = "verified"
    applicability: str = "unspecified"
    limitations: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TypeError("evidence source_id must be a string")
        if not OPAQUE_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError("evidence source_id must be a bounded safe identifier")
        if not isinstance(self.source_kind, str):
            raise TypeError("evidence source_kind must be a string")
        if not self.source_kind.strip():
            raise ValueError("evidence source_kind must be a non-empty string")
        if not isinstance(self.excerpt, str):
            raise TypeError("evidence excerpt must be a string")
        if self.url is not None and not isinstance(self.url, str):
            raise TypeError("evidence url must be a string or None")
        if self.url is not None and self.url:
            parsed_url = urlsplit(self.url)
            if (
                parsed_url.scheme != "https"
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or parsed_url.query
                or parsed_url.fragment
            ):
                raise ValueError("evidence url must be a query-free HTTPS URL without userinfo")
        if not isinstance(self.source_name, str):
            raise TypeError("evidence source_name must be a string")
        if not isinstance(self.retrieval_status, str) or not self.retrieval_status.strip():
            raise ValueError("evidence retrieval_status must be a non-empty string")
        if not isinstance(self.applicability, str) or not self.applicability.strip():
            raise ValueError("evidence applicability must be a non-empty string")
        if not isinstance(self.limitations, str):
            raise TypeError("evidence limitations must be a string")


@dataclass(frozen=True)
class TreatmentResult:
    status: StepStatus
    markdown: str = ""
    evidence: tuple[EvidenceItem, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, StepStatus):
            raise TypeError("treatment status must be a StepStatus")
        if not isinstance(self.markdown, str):
            raise TypeError("treatment markdown must be a string")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("treatment evidence entries must be EvidenceItem values")
        object.__setattr__(self, "evidence", evidence)
        if not isinstance(self.provenance, Mapping):
            raise TypeError("treatment provenance must be a mapping")
        object.__setattr__(self, "provenance", freeze_json(self.provenance))

    @classmethod
    def pending(cls) -> "TreatmentResult":
        return cls(status=StepStatus("pending"))

    @classmethod
    def empty(cls) -> "TreatmentResult":
        return cls(status=StepStatus("completed"))


@dataclass(frozen=True)
class TreatmentInput:
    clinical_facts: Mapping[str, Any]
    diagnostic_status: StepStatus
    diagnostic_result: DiagnosticResult
    survival_status: StepStatus
    survival_result: SurvivalResult

    def __post_init__(self) -> None:
        if not isinstance(self.clinical_facts, Mapping):
            raise TypeError("treatment clinical_facts must be a mapping")
        object.__setattr__(self, "clinical_facts", freeze_json(self.clinical_facts))
        if not isinstance(self.diagnostic_status, StepStatus):
            raise TypeError("diagnostic_status must be a StepStatus")
        if not isinstance(self.diagnostic_result, DiagnosticResult):
            raise TypeError("diagnostic_result must be a DiagnosticResult")
        if not isinstance(self.survival_status, StepStatus):
            raise TypeError("survival_status must be a StepStatus")
        if not isinstance(self.survival_result, SurvivalResult):
            raise TypeError("survival_result must be a SurvivalResult")
        if self.diagnostic_status != self.diagnostic_result.status:
            raise ValueError("diagnostic status must match diagnostic result")
        if self.survival_status != self.survival_result.status:
            raise ValueError("survival status must match survival result")


@dataclass(frozen=True)
class StepRecord:
    step: StepName
    status: StepStatus
    result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step not in ("diagnostic", "survival", "treatment"):
            raise ValueError(f"invalid workflow step: {self.step!r}")
        if not isinstance(self.status, StepStatus):
            raise TypeError("step record status must be a StepStatus")
        if not isinstance(self.result, Mapping):
            raise TypeError("step record result must be a mapping")
        object.__setattr__(self, "result", freeze_json(self.result))

    @property
    def name(self) -> StepName:
        return self.step


@dataclass(frozen=True)
class CaseTrace:
    schema_version: str
    case_id: str
    steps: tuple[StepRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("trace schema_version must be a non-empty string")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("trace case_id must be a non-empty opaque string")
        steps = tuple(self.steps)
        if not all(isinstance(item, StepRecord) for item in steps):
            raise TypeError("trace steps must be StepRecord values")
        if tuple(item.step for item in steps) != ("diagnostic", "survival", "treatment"):
            raise ValueError("trace steps must follow diagnostic, survival, treatment workflow order")
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_jsonable(self))


def to_jsonable(value: Any) -> Any:
    """Return a detached JSON-compatible representation of a contract value."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")
