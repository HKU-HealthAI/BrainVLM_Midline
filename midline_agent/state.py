"""Immutable workflow-state transitions and privacy-minimized projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import re
from typing import Any, cast

from .schema import (
    CaseInput,
    CaseTrace,
    DiagnosticResult,
    StepRecord,
    StepStatus,
    SurvivalResult,
    TUMOR_TYPES,
    TreatmentInput,
    TreatmentResult,
    WHO_GRADES,
    to_jsonable,
)


TRACE_PROVENANCE_FIELDS = frozenset(
    {
        "adapter_version",
        "checkpoint_identifier",
        "class_vocabulary_version",
        "feature_schema_version",
        "model_identifier",
        "preprocessing_version",
        "prompt_version",
        "rejected_vote_count",
        "rule_set_version",
        "vote_counts",
    }
)
TRACE_IDENTIFIER_FIELDS = TRACE_PROVENANCE_FIELDS - {"rejected_vote_count", "vote_counts"}
VOTE_LABEL_PATTERN = re.compile(r"(?:grade_[234]|tumor_[1-5])\Z")
EVIDENCE_SOURCE_KINDS = frozenset(
    {"guideline", "knowledge_base", "local_knowledge", "physician_skill", "similar_case", "web_search"}
)
CLINICAL_STORAGE_TOKENS = frozenset(
    {"dir", "directory", "file", "filename", "folder", "manifest", "path", "uri", "url"}
)
CLINICAL_DIRECT_IDENTIFIER_TOKENS = frozenset(
    {"address", "email", "passport", "phone", "ssn", "telephone"}
)
CLINICAL_ID_QUALIFIERS = frozenset(
    {"case", "hospital", "medical", "national", "patient", "person", "provider", "record", "subject", "user"}
)
CLINICAL_NAME_QUALIFIERS = frozenset(
    {"doctor", "family", "first", "full", "given", "last", "patient", "person", "physician", "provider", "subject"}
)
CLINICAL_SECRET_TOKENS = frozenset(
    {"auth", "authorization", "credential", "credentials", "passwd", "password", "secret", "token"}
)
CLINICAL_KEY_QUALIFIERS = frozenset(
    {"access", "api", "encryption", "openai", "private", "signing"}
)
CLINICAL_DIRECT_IDENTIFIER_ALIASES = frozenset(
    {
        "caseid",
        "dateofbirth",
        "dob",
        "familyname",
        "firstname",
        "fullname",
        "givenname",
        "id",
        "identifier",
        "lastname",
        "medicalrecordnumber",
        "mrn",
        "name",
        "patientid",
        "patientname",
        "subjectid",
    }
)
STORAGE_VALUE_SUFFIXES = (
    ".bmp",
    ".csv",
    ".dcm",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".jsonl",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
    ".tsv",
)
CALENDAR_DATE_VALUE_PATTERN = re.compile(r"\d{4}/\d{1,2}/\d{1,2}\Z")
DROP_CLINICAL_VALUE = object()


def _trace_status(status: StepStatus) -> StepStatus:
    reason = ""
    if status.state == "failed":
        reason = "step failed"
    elif status.state == "unavailable":
        reason = "step unavailable"
    return StepStatus(status.state, reason)


def _treatment_status(status: StepStatus, step: str) -> StepStatus:
    reason = ""
    if status.state == "failed":
        reason = f"{step} step failed"
    elif status.state == "unavailable":
        reason = f"{step} step unavailable"
    return StepStatus(status.state, reason)


def _identifier_fingerprint(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _trace_case_id(value: str) -> str:
    return _identifier_fingerprint(value) or "sha256:invalid"


def _safe_vote_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    safe_counts = {
        key: count
        for key, count in value.items()
        if isinstance(key, str)
        and VOTE_LABEL_PATTERN.fullmatch(key)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }
    return safe_counts or None


def _key_tokens(key: str) -> frozenset[str]:
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    return frozenset(token for token in re.split(r"[^A-Za-z0-9]+", separated.lower()) if token)


def _is_sensitive_clinical_key(key: str) -> bool:
    tokens = _key_tokens(key)
    compact = re.sub(r"[^A-Za-z0-9]+", "", key).lower()
    if tokens.intersection(CLINICAL_STORAGE_TOKENS):
        return True
    if compact in {"imagemanifest", "mrimanifest", "mristudy"} or compact.endswith(
        ("directory", "filename", "filepath", "imagepath", "mripath")
    ):
        return True
    if tokens.intersection(CLINICAL_DIRECT_IDENTIFIER_TOKENS):
        return True
    if tokens.intersection({"dob", "mrn"}) or {"date", "birth"}.issubset(tokens):
        return True
    if tokens.intersection({"no", "number"}) and tokens.intersection(
        {"account", "hospital", "medical", "patient", "record", "subject"}
    ):
        return True
    if compact in CLINICAL_DIRECT_IDENTIFIER_ALIASES:
        return True
    if tokens.intersection({"id", "identifier"}) and tokens.intersection(CLINICAL_ID_QUALIFIERS):
        return True
    if "name" in tokens and tokens.intersection(CLINICAL_NAME_QUALIFIERS):
        return True
    if tokens.intersection(CLINICAL_SECRET_TOKENS):
        return True
    if compact.endswith(("accesskey", "apikey", "encryptionkey", "privatekey", "signingkey")):
        return True
    return "key" in tokens and bool(
        tokens.intersection(CLINICAL_KEY_QUALIFIERS) or {"open", "ai"}.issubset(tokens)
    )


def _is_storage_value(value: str) -> bool:
    candidate = value.strip()
    lowered = candidate.lower()
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", candidate):
        return True
    if candidate.startswith(("/", "~/", "./", "../", "\\\\")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return True
    if lowered.endswith(STORAGE_VALUE_SUFFIXES):
        return True
    if CALENDAR_DATE_VALUE_PATTERN.fullmatch(candidate):
        return False
    components = candidate.replace("\\", "/").split("/")
    return (
        len(components) >= 3
        and all(components)
        and not any(any(character.isspace() for character in part) for part in components)
    )


def _sanitize_clinical_facts(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if _is_sensitive_clinical_key(key):
                continue
            clean_item = _sanitize_clinical_facts(item)
            if clean_item is not DROP_CLINICAL_VALUE:
                sanitized[key] = clean_item
        return sanitized
    if isinstance(value, (tuple, list)):
        sanitized_items = tuple(_sanitize_clinical_facts(item) for item in value)
        return tuple(item for item in sanitized_items if item is not DROP_CLINICAL_VALUE)
    if isinstance(value, str) and _is_storage_value(value):
        return DROP_CLINICAL_VALUE
    return value


def _trace_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in sorted(TRACE_IDENTIFIER_FIELDS.intersection(provenance)):
        fingerprint = _identifier_fingerprint(provenance[key])
        if fingerprint is not None:
            safe[key] = fingerprint
    rejected_count = provenance.get("rejected_vote_count")
    if isinstance(rejected_count, int) and not isinstance(rejected_count, bool) and rejected_count >= 0:
        safe["rejected_vote_count"] = rejected_count
    vote_counts = _safe_vote_counts(provenance.get("vote_counts"))
    if vote_counts is not None:
        safe["vote_counts"] = vote_counts
    return safe


def _diagnostic_trace_result(result: DiagnosticResult) -> dict[str, Any]:
    return {
        "tumor_type": result.tumor_type if result.tumor_type in TUMOR_TYPES else None,
        "who_grade": result.who_grade if result.who_grade in WHO_GRADES else None,
        "probabilities": result.probabilities,
        "provenance": _trace_provenance(result.provenance),
    }


def _treatment_trace_result(result: TreatmentResult) -> dict[str, Any]:
    evidence = []
    for item in result.evidence:
        if item.source_kind in EVIDENCE_SOURCE_KINDS:
            evidence.append(
                {
                    "source_id": f"source-{len(evidence) + 1}",
                    "source_kind": item.source_kind,
                    "retrieval_status": (
                        item.retrieval_status
                        if item.retrieval_status in {"verified", "unverified", "unavailable", "rejected"}
                        else "unclassified"
                    ),
                    "applicability": "specified" if item.applicability != "unspecified" else "unspecified",
                }
            )
    return {
        "evidence": tuple(evidence),
        "provenance": {
            key: value
            for key, value in result.provenance.items()
            if key in {"handoff_state"} and isinstance(value, str)
        },
    }


@dataclass(frozen=True)
class CaseState:
    schema_version: str
    case_input: CaseInput
    diagnostic: DiagnosticResult
    survival: SurvivalResult
    treatment: TreatmentResult

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.case_input, CaseInput):
            raise TypeError("case_input must be a CaseInput")
        if not isinstance(self.diagnostic, DiagnosticResult):
            raise TypeError("diagnostic must be a DiagnosticResult")
        if not isinstance(self.survival, SurvivalResult):
            raise TypeError("survival must be a SurvivalResult")
        if not isinstance(self.treatment, TreatmentResult):
            raise TypeError("treatment must be a TreatmentResult")
        if self.treatment.status.state == "completed" and self.diagnostic.status.state not in {
            "completed",
            "unavailable",
        }:
            raise ValueError(
                "completed treatment requires a completed or unavailable diagnostic result"
            )
        if self.treatment.status.state != "pending" and self.diagnostic.status.state == "pending":
            raise ValueError("terminal treatment requires diagnostic completion")

    @classmethod
    def from_input(cls, case_input: CaseInput) -> "CaseState":
        return cls(
            schema_version="1.0",
            case_input=case_input,
            diagnostic=DiagnosticResult.pending(),
            survival=SurvivalResult.unavailable(),
            treatment=TreatmentResult.pending(),
        )

    @property
    def step_records(self) -> tuple[StepRecord, ...]:
        return (
            StepRecord("diagnostic", self.diagnostic.status),
            StepRecord("survival", self.survival.status),
            StepRecord("treatment", self.treatment.status),
        )

    def with_diagnostic(self, result: DiagnosticResult) -> "CaseState":
        if not isinstance(result, DiagnosticResult):
            raise TypeError("result must be a DiagnosticResult")
        if self.treatment.status.state != "pending":
            raise ValueError("cannot change diagnosis after treatment has started")
        return replace(self, diagnostic=result)

    def with_treatment(self, result: TreatmentResult) -> "CaseState":
        if not isinstance(result, TreatmentResult):
            raise TypeError("result must be a TreatmentResult")
        if self.treatment.status.state != "pending":
            raise ValueError("cannot change treatment after it has finished")
        return replace(self, treatment=result)

    def to_treatment_input(self) -> TreatmentInput:
        diagnostic_status = _treatment_status(self.diagnostic.status, "diagnostic")
        survival_status = _treatment_status(self.survival.status, "survival")
        return TreatmentInput(
            clinical_facts=_sanitize_clinical_facts(self.case_input.clinical_facts),
            diagnostic_status=diagnostic_status,
            diagnostic_result=DiagnosticResult(
                status=diagnostic_status,
                tumor_type=(
                    self.diagnostic.tumor_type if self.diagnostic.tumor_type in TUMOR_TYPES else None
                ),
                who_grade=self.diagnostic.who_grade if self.diagnostic.who_grade in WHO_GRADES else None,
                probabilities=self.diagnostic.probabilities,
            ),
            survival_status=survival_status,
            survival_result=SurvivalResult(status=survival_status),
        )

    def to_trace(self) -> CaseTrace:
        return CaseTrace(
            schema_version=self.schema_version,
            case_id=_trace_case_id(self.case_input.case_id),
            steps=(
                StepRecord(
                    "diagnostic",
                    _trace_status(self.diagnostic.status),
                    _diagnostic_trace_result(self.diagnostic),
                ),
                StepRecord("survival", _trace_status(self.survival.status)),
                StepRecord(
                    "treatment",
                    _trace_status(self.treatment.status),
                    _treatment_trace_result(self.treatment),
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete in-memory state for compatibility, not persistence."""
        return cast(dict[str, Any], to_jsonable(self))
