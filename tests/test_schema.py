import json

import pytest

from midline_agent.schema import (
    CaseInput,
    CaseTrace,
    DiagnosticResult,
    EvidenceItem,
    StepRecord,
    StepStatus,
    SurvivalResult,
    TreatmentInput,
    TreatmentResult,
    to_jsonable,
)
from midline_agent.state import CaseState


def test_step_status_rejects_unknown_state() -> None:
    with pytest.raises(ValueError, match="invalid step state"):
        StepStatus("running")


def test_case_state_starts_with_pending_steps_in_workflow_order() -> None:
    state = CaseState.from_input(CaseInput(case_id="synthetic-1"))

    assert state.diagnostic.status.state == "pending"
    assert state.survival.status.state == "unavailable"
    assert state.treatment.status.state == "pending"
    assert [record.step for record in state.step_records] == ["diagnostic", "survival", "treatment"]


def test_contracts_deep_freeze_nested_input_provenance_and_evidence() -> None:
    clinical_facts = {"measurements": {"age_years": 44}, "symptoms": ["headache"]}
    mri_record = {"path": "/synthetic-storage/study/t1.npy", "metadata": {"sequence": "T1"}}
    provenance = {"vote_counts": {"grade_3": 2}, "rejected_votes": ["raw response"]}
    evidence = [EvidenceItem(source_id="guideline-1", source_kind="guideline", excerpt="source text")]

    case_input = CaseInput(case_id="synthetic-1", clinical_facts=clinical_facts, mri_manifest=(mri_record,))
    diagnostic = DiagnosticResult(
        status=StepStatus("completed"),
        tumor_type="tumor_4",
        who_grade="grade_3",
        provenance=provenance,
    )
    treatment = TreatmentResult(status=StepStatus("completed"), evidence=evidence)

    clinical_facts["measurements"]["age_years"] = 99
    clinical_facts["symptoms"].append("copied mutation")
    mri_record["metadata"]["sequence"] = "changed"
    provenance["vote_counts"]["grade_3"] = 0
    evidence.append(EvidenceItem(source_id="late", source_kind="guideline"))

    assert case_input.clinical_facts["measurements"]["age_years"] == 44
    assert case_input.clinical_facts["symptoms"] == ("headache",)
    assert case_input.mri_manifest[0]["metadata"]["sequence"] == "T1"
    assert diagnostic.provenance["vote_counts"]["grade_3"] == 2
    assert [item.source_id for item in treatment.evidence] == ["guideline-1"]
    with pytest.raises(TypeError):
        case_input.clinical_facts["measurements"]["age_years"] = 50
    with pytest.raises(TypeError):
        diagnostic.provenance["vote_counts"]["grade_3"] = 3
    json.dumps(to_jsonable(case_input))
    json.dumps(to_jsonable(diagnostic))
    json.dumps(to_jsonable(treatment))


def test_contracts_reject_mutable_scalars_and_nonfinite_probabilities() -> None:
    with pytest.raises(TypeError, match="source_id"):
        EvidenceItem(source_id={"mutable": "value"}, source_kind="guideline")
    with pytest.raises(ValueError, match="finite"):
        DiagnosticResult(status=StepStatus("completed"), probabilities=(float("nan"),))
    with pytest.raises(ValueError, match="finite"):
        CaseInput(case_id="synthetic-1", clinical_facts={"score": float("inf")})


@pytest.mark.parametrize(
    ("tumor_type", "who_grade"),
    [
        (None, "grade_3"),
        ("", "grade_3"),
        ("   ", "grade_3"),
        ("unknown", "grade_3"),
        ("tumor_4", None),
        ("tumor_4", ""),
        ("tumor_4", "grade_9"),
    ],
)
def test_completed_diagnostic_requires_valid_labels(
    tumor_type: str | None, who_grade: str | None
) -> None:
    with pytest.raises(ValueError, match="completed diagnostic"):
        DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type=tumor_type,
            who_grade=who_grade,
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.5, 0.5),
        (0.1, 0.2, 0.3, 0.4, 0.1),
        (-0.1, 0.2, 0.3, 0.4, 0.2),
        (1.1, 0.0, 0.0, 0.0, -0.1),
        (0.6, 0.1, 0.1, 0.1, 0.1),
    ],
)
def test_completed_diagnostic_rejects_incoherent_probabilities(
    probabilities: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type="tumor_4",
            who_grade="grade_3",
            probabilities=probabilities,
        )


def test_completed_diagnostic_accepts_categorical_or_coherent_probabilities() -> None:
    categorical = DiagnosticResult(
        status=StepStatus("completed"), tumor_type="tumor_4", who_grade="grade_3"
    )
    probabilistic = DiagnosticResult(
        status=StepStatus("completed"),
        tumor_type="tumor_4",
        who_grade="grade_3",
        probabilities=(0.1, 0.1, 0.1, 0.6, 0.1),
    )

    assert categorical.probabilities == ()
    assert sum(probabilistic.probabilities) == pytest.approx(1.0)


def test_treatment_input_rejects_non_contract_results() -> None:
    status = StepStatus("pending")

    with pytest.raises(TypeError, match="diagnostic_result"):
        TreatmentInput(
            clinical_facts={},
            diagnostic_status=status,
            diagnostic_result=object(),
            survival_status=StepStatus("unavailable"),
            survival_result=SurvivalResult.unavailable(),
        )


def test_treatment_projection_has_upstream_results_but_no_mri_manifest() -> None:
    state = CaseState.from_input(
        CaseInput(
            case_id="synthetic-1",
            clinical_facts={"age_years": 44},
            mri_manifest=({"path": "/synthetic-storage/study/t1.npy"},),
        )
    ).with_diagnostic(
        DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type="tumor_4",
            who_grade="grade_3",
            provenance={"input_path": "/synthetic-storage/TREATMENT_MRI_PATH_SENTINEL/t1.npy"},
        )
    )

    projection = state.to_treatment_input()
    payload = to_jsonable(projection)
    encoded = json.dumps(payload, sort_keys=True)

    assert projection.clinical_facts["age_years"] == 44
    assert projection.diagnostic_status.state == "completed"
    assert projection.diagnostic_result.tumor_type == "tumor_4"
    assert projection.survival_status.state == "unavailable"
    assert projection.survival_result.status == projection.survival_status
    assert set(payload) == {
        "clinical_facts",
        "diagnostic_status",
        "diagnostic_result",
        "survival_status",
        "survival_result",
    }
    assert "mri_manifest" not in encoded
    assert "/synthetic-storage/study/t1.npy" not in encoded
    assert "TREATMENT_MRI_PATH_SENTINEL" not in encoded


def test_treatment_projection_sanitizes_nested_clinical_facts() -> None:
    api_key_field = "_".join(("api", "key"))
    camel_api_key_field = "".join(("API", "Key"))
    openai_key_field = "".join(("open", "AI", "Key"))
    openai_api_key_field = "".join(("OpenAI", "API", "Key"))
    access_token_field = "_".join(("access", "token"))
    provider_secret_field = "_".join(("provider", "secret"))
    state = CaseState.from_input(
        CaseInput(
            case_id="opaque-case-2",
            clinical_facts={
                "age_years": 44,
                "kps": 80,
                "tumor_location": "thalamus",
                "symptoms": ["headache", "diplopia"],
                "assessment": {
                    "location": "midline",
                    "measurements_mm": [12.0, 9.5],
                    "mri_path": "/synthetic-storage/NESTED_MRI_PATH_SENTINEL/t1.npy",
                    "image_path": "/synthetic-storage/NESTED_IMAGE_PATH_SENTINEL/slice.png",
                    "source_file": "/synthetic-storage/NESTED_FILE_SENTINEL/report.txt",
                    "filepath": "/synthetic-storage/COMPACT_FILEPATH_SENTINEL/report.txt",
                    "storage_directory": "/synthetic-storage/NESTED_DIRECTORY_SENTINEL",
                    "MRIStudy": "MRI_ACRONYM_SENTINEL",
                    "name": "DIRECT_BARE_NAME_SENTINEL",
                    "id": "DIRECT_BARE_ID_SENTINEL",
                    "patient_name": "DIRECT_NAME_SENTINEL",
                    "patientId": "DIRECT_PATIENT_ID_SENTINEL",
                    "patient_mrn": "QUALIFIED_MRN_SENTINEL",
                    "subjectDOB": "QUALIFIED_DOB_SENTINEL",
                    "patient_record_number": "QUALIFIED_RECORD_NUMBER_SENTINEL",
                    "medical_record_number": "DIRECT_MRN_SENTINEL",
                    "dateOfBirth": "DIRECT_DOB_SENTINEL",
                    api_key_field: "API_KEY_SENTINEL",
                    camel_api_key_field: "CAMEL_API_KEY_SENTINEL",
                    openai_key_field: "OPENAI_KEY_SENTINEL",
                    openai_api_key_field: "OPENAI_API_KEY_SENTINEL",
                    access_token_field: "ACCESS_TOKEN_SENTINEL",
                },
                "history": [
                    {
                        "symptom": "nausea",
                        "scan_filename": "NESTED_FILENAME_SENTINEL.nii.gz",
                        "email_address": "DIRECT_EMAIL_SENTINEL@example.invalid",
                        provider_secret_field: "PROVIDER_SECRET_SENTINEL",
                    }
                ],
            },
        )
    )

    projection = state.to_treatment_input()
    encoded = json.dumps(to_jsonable(projection.clinical_facts), sort_keys=True)

    assert projection.clinical_facts["age_years"] == 44
    assert projection.clinical_facts["kps"] == 80
    assert projection.clinical_facts["tumor_location"] == "thalamus"
    assert projection.clinical_facts["symptoms"] == ("headache", "diplopia")
    assert projection.clinical_facts["assessment"]["location"] == "midline"
    assert projection.clinical_facts["assessment"]["measurements_mm"] == (12.0, 9.5)
    assert projection.clinical_facts["history"][0]["symptom"] == "nausea"
    for sensitive_value in (
        "NESTED_MRI_PATH_SENTINEL",
        "NESTED_IMAGE_PATH_SENTINEL",
        "NESTED_FILE_SENTINEL",
        "COMPACT_FILEPATH_SENTINEL",
        "NESTED_DIRECTORY_SENTINEL",
        "NESTED_FILENAME_SENTINEL",
        "MRI_ACRONYM_SENTINEL",
        "DIRECT_BARE_NAME_SENTINEL",
        "DIRECT_BARE_ID_SENTINEL",
        "DIRECT_NAME_SENTINEL",
        "DIRECT_PATIENT_ID_SENTINEL",
        "QUALIFIED_MRN_SENTINEL",
        "QUALIFIED_DOB_SENTINEL",
        "QUALIFIED_RECORD_NUMBER_SENTINEL",
        "DIRECT_MRN_SENTINEL",
        "DIRECT_DOB_SENTINEL",
        "DIRECT_EMAIL_SENTINEL",
        "API_KEY_SENTINEL",
        "CAMEL_API_KEY_SENTINEL",
        "OPENAI_KEY_SENTINEL",
        "OPENAI_API_KEY_SENTINEL",
        "ACCESS_TOKEN_SENTINEL",
        "PROVIDER_SECRET_SENTINEL",
    ):
        assert sensitive_value not in encoded
    with pytest.raises(TypeError):
        projection.clinical_facts["assessment"]["location"] = "changed"


def test_treatment_projection_sanitizes_path_values_but_preserves_imaging_facts() -> None:
    state = CaseState.from_input(
        CaseInput(
            case_id="opaque-case-3",
            clinical_facts={
                "mri_findings": "T2 hyperintensity without enhancement",
                "imaging_features": {
                    "signal": "T2/FLAIR hyperintensity",
                    "enhancement": "none",
                    "modalities": "T2/FLAIR",
                },
                "scan_date_relative_days": -2,
                "clinical_summary": "/synthetic-storage/VALUE_PATH_SENTINEL/t1.nii.gz",
                "observations": {
                    "primary": "s3://synthetic-bucket/VALUE_URI_SENTINEL/t1.nii.gz",
                    "items": [
                        "headache",
                        "synthetic-series/VALUE_RELATIVE_PATH_SENTINEL/slice.dcm",
                        "VALUE_FILENAME_SENTINEL.nii.gz",
                        "synthetic-series/VALUE_EXTENSIONLESS_PATH_SENTINEL/series",
                        "synthetic-series\\VALUE_WINDOWS_PATH_SENTINEL\\series",
                    ],
                },
            },
        )
    )

    projection = state.to_treatment_input()
    encoded = json.dumps(to_jsonable(projection.clinical_facts), sort_keys=True)

    assert projection.clinical_facts["mri_findings"] == "T2 hyperintensity without enhancement"
    assert projection.clinical_facts["imaging_features"]["signal"] == "T2/FLAIR hyperintensity"
    assert projection.clinical_facts["imaging_features"]["enhancement"] == "none"
    assert projection.clinical_facts["imaging_features"]["modalities"] == "T2/FLAIR"
    assert projection.clinical_facts["scan_date_relative_days"] == -2
    assert projection.clinical_facts["observations"]["items"] == ("headache",)
    for sensitive_value in (
        "VALUE_PATH_SENTINEL",
        "VALUE_URI_SENTINEL",
        "VALUE_RELATIVE_PATH_SENTINEL",
        "VALUE_FILENAME_SENTINEL",
        "VALUE_EXTENSIONLESS_PATH_SENTINEL",
        "VALUE_WINDOWS_PATH_SENTINEL",
    ):
        assert sensitive_value not in encoded
    with pytest.raises(TypeError):
        projection.clinical_facts["imaging_features"]["signal"] = "changed"


def test_treatment_projection_distinguishes_clinical_slashes_from_storage_paths() -> None:
    clinical_values = {
        "motor_strength": "3/5",
        "radiotherapy": "60 Gy/30 fractions",
        "dose": "75 mg/m2",
        "assessment_date": "2026/11/18",
        "symptom_summary": "headache/nausea",
    }
    state = CaseState.from_input(
        CaseInput(
            case_id="opaque-case-4",
            clinical_facts={
                **clinical_values,
                "observations": {
                    "absolute": "/synthetic-storage/case-1/series-2",
                    "drive": "C:\\synthetic-storage\\case-1\\series-2",
                    "unc": "\\\\synthetic-server\\synthetic-share\\case-1",
                    "current_relative": "./synthetic-storage/case-1",
                    "parent_relative": "../synthetic-storage/case-1",
                    "known_filename": "VALUE_IMAGE_FILENAME_SENTINEL.nii.gz",
                    "extensionless": (
                        "synthetic-storage/VALUE_EXTENSIONLESS_SENTINEL/series-2"
                    ),
                },
            },
        )
    )

    projection = state.to_treatment_input()
    encoded = json.dumps(to_jsonable(projection.clinical_facts), sort_keys=True)

    for key, value in clinical_values.items():
        assert projection.clinical_facts[key] == value
    for sensitive_value in (
        "synthetic-storage",
        "synthetic-server",
        "VALUE_IMAGE_FILENAME_SENTINEL",
        "VALUE_EXTENSIONLESS_SENTINEL",
    ):
        assert sensitive_value not in encoded


def _valid_completed_diagnostic() -> DiagnosticResult:
    return DiagnosticResult(
        status=StepStatus("completed"), tumor_type="tumor_4", who_grade="grade_3"
    )


@pytest.mark.parametrize("member", ["case_input", "diagnostic", "survival", "treatment"])
def test_case_state_validates_member_types(member: str) -> None:
    values = {
        "schema_version": "1.0",
        "case_input": CaseInput(case_id="synthetic-1"),
        "diagnostic": DiagnosticResult.pending(),
        "survival": SurvivalResult.unavailable(),
        "treatment": TreatmentResult.pending(),
    }
    values[member] = object()

    with pytest.raises(TypeError, match=member):
        CaseState(**values)


def test_case_state_rejects_completed_treatment_without_usable_diagnosis() -> None:
    for diagnostic in (
        DiagnosticResult.pending(),
        DiagnosticResult(status=StepStatus("failed", "inference failed")),
    ):
        with pytest.raises(ValueError, match="completed treatment"):
            CaseState(
                schema_version="1.0",
                case_input=CaseInput(case_id="synthetic-1"),
                diagnostic=diagnostic,
                survival=SurvivalResult.unavailable(),
                treatment=TreatmentResult.empty(),
            )


def test_case_state_transitions_diagnostic_before_treatment_monotonically() -> None:
    initial = CaseState.from_input(CaseInput(case_id="synthetic-1"))
    diagnosed = initial.with_diagnostic(_valid_completed_diagnostic())
    treated = diagnosed.with_treatment(TreatmentResult.empty())

    assert treated.diagnostic.status.state == "completed"
    assert treated.survival.status.state == "unavailable"
    assert treated.treatment.status.state == "completed"
    with pytest.raises(ValueError, match="after treatment"):
        treated.with_diagnostic(DiagnosticResult.unavailable("replacement"))


def test_completed_treatment_allows_intentionally_unavailable_diagnosis_and_survival() -> None:
    state = CaseState.from_input(CaseInput(case_id="synthetic-1")).with_diagnostic(
        DiagnosticResult.unavailable("adapter not configured")
    )

    treated = state.with_treatment(TreatmentResult.empty())

    assert treated.diagnostic.status.state == "unavailable"
    assert treated.survival.status.state == "unavailable"
    assert treated.treatment.status.state == "completed"


def test_with_treatment_rejects_completed_result_after_failed_diagnosis() -> None:
    state = CaseState.from_input(CaseInput(case_id="synthetic-1")).with_diagnostic(
        DiagnosticResult(status=StepStatus("failed", "inference failed"))
    )

    with pytest.raises(ValueError, match="completed treatment"):
        state.with_treatment(TreatmentResult.empty())


def test_completed_diagnostic_rejects_mri_path_in_labels() -> None:
    with pytest.raises(ValueError, match="completed diagnostic"):
        DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type="/synthetic-storage/LABEL_MRI_PATH_SENTINEL/t1.npy",
            who_grade="grade_3",
        )


def test_case_trace_rejects_missing_duplicate_or_out_of_order_steps() -> None:
    status = StepStatus("pending")

    with pytest.raises(ValueError, match="workflow order"):
        CaseTrace(
            schema_version="1.0",
            case_id="opaque-case-1",
            steps=(
                StepRecord("treatment", status),
                StepRecord("survival", status),
                StepRecord("diagnostic", status),
            ),
        )


def test_case_trace_is_ordered_serializable_and_minimized() -> None:
    api_key_field = "_".join(("api", "key"))
    state = CaseState.from_input(
        CaseInput(
            case_id="opaque-case-7",
            clinical_facts={
                "patient_name": "DIRECT_IDENTIFIER_SENTINEL",
                "clinical_note": "RAW_CLINICAL_TEXT_SENTINEL",
                api_key_field: "SECRET_SENTINEL",
            },
            mri_manifest=({"path": "/synthetic-storage/MRI_PATH_SENTINEL/t1.npy"},),
        )
    ).with_diagnostic(
        DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type="tumor_4",
            who_grade="grade_3",
            probabilities=(0.1, 0.2, 0.3, 0.4, 0.0),
            provenance={
                "model_identifier": "public-model-v1",
                "checkpoint_identifier": "DIRECT_EMAIL_SENTINEL@example.invalid",
                "adapter_version": "/synthetic-storage/ALLOWLIST_PATH_SENTINEL/adapter.py",
                "prompt_version": "ALLOWLIST_SECRET_SENTINEL",
                "vote_counts": {"grade_3": 2},
                "raw_generations": ["RAW_PROVIDER_PAYLOAD_SENTINEL"],
                api_key_field: "PROVENANCE_SECRET_SENTINEL",
                "input_path": "/synthetic-storage/PROVENANCE_PATH_SENTINEL",
                "patient_name": "PROVENANCE_IDENTIFIER_SENTINEL",
            },
        )
    ).with_treatment(
        TreatmentResult(
            status=StepStatus("failed", "provider unavailable"),
            markdown="",
            evidence=(
                EvidenceItem(
                    source_id="source-1",
                    source_kind="guideline",
                    excerpt="RAW_EVIDENCE_PAYLOAD_SENTINEL",
                    url="https://example.invalid/source",
                ),
            ),
        )
    )

    trace = state.to_trace()
    payload = trace.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["case_id"].startswith("sha256:")
    assert payload["case_id"] != "opaque-case-7"
    assert [record.step for record in trace.steps] == ["diagnostic", "survival", "treatment"]
    assert trace.steps[0].result["tumor_type"] == "tumor_4"
    assert trace.steps[0].result["provenance"]["model_identifier"].startswith("sha256:")
    assert trace.steps[2].status.state == "failed"
    assert trace.steps[2].result["evidence"][0]["source_id"] == "source-1"
    for sensitive_value in (
        "DIRECT_IDENTIFIER_SENTINEL",
        "RAW_CLINICAL_TEXT_SENTINEL",
        "SECRET_SENTINEL",
        "MRI_PATH_SENTINEL",
        "RAW_PROVIDER_PAYLOAD_SENTINEL",
        "PROVENANCE_SECRET_SENTINEL",
        "PROVENANCE_PATH_SENTINEL",
        "PROVENANCE_IDENTIFIER_SENTINEL",
        "RAW_EVIDENCE_PAYLOAD_SENTINEL",
        "URL_SECRET_SENTINEL",
        "ALLOWLIST_PATH_SENTINEL",
        "ALLOWLIST_SECRET_SENTINEL",
        "SIGNED_SOURCE_SENTINEL",
        "DIRECT_EMAIL_SENTINEL",
        "public-model-v1",
    ):
        assert sensitive_value not in encoded
    with pytest.raises(TypeError):
        trace.steps[0].result["tumor_type"] = "changed"


@pytest.mark.parametrize(
    ("source_id", "url"),
    (
        ("https://source.invalid/path", None),
        ("source with spaces", None),
        ("source\nnewline", None),
        ("source-1", "https://user:secret@example.invalid/path"),
        ("source-1", "https://example.invalid/path?token=secret"),
        ("source-1", "https://example.invalid/path#fragment"),
        ("source-1", "https://"),
    ),
)
def test_evidence_contract_rejects_unsafe_identifiers_and_urls(
    source_id: str, url: str | None
) -> None:
    with pytest.raises(ValueError):
        EvidenceItem(source_id=source_id, source_kind="guideline", url=url)


def test_treatment_cannot_mutate_diagnostic_result() -> None:
    diagnosed = CaseState.from_input(CaseInput(case_id="synthetic-1")).with_diagnostic(
        DiagnosticResult.unavailable("missing checkpoint")
    )

    treated = diagnosed.with_treatment(TreatmentResult.empty())

    assert treated.diagnostic is diagnosed.diagnostic
    assert treated.to_dict()["schema_version"] == "1.0"
    assert treated.treatment.status.state == "completed"
