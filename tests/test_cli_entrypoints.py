import json
import subprocess
import sys
from pathlib import Path

from midline_agent.schema import CaseInput, DiagnosticResult, StepStatus, TreatmentResult
from midline_agent.state import CaseState


def test_pipeline_runner_exposes_no_custom_openai_endpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_pipeline.py").read_text(encoding="utf-8")
    endpoint_environment_name = "OPENAI_BASE" + "_URL"
    endpoint_argument_name = "base" + "_url"

    assert endpoint_environment_name not in source
    assert endpoint_argument_name not in source
    assert "user-supplied LLM endpoint" not in source
    assert "--mock-diagnostic" not in source


def test_diagnostic_cli_entrypoints_require_explicit_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    extraction = subprocess.run(
        [sys.executable, "scripts/extract_features.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    training = subprocess.run(
        [sys.executable, "scripts/train_attention_mil.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert extraction.returncode == 0, extraction.stderr
    assert "--output" in extraction.stdout
    assert "--annotations" in extraction.stdout
    assert training.returncode == 0, training.stderr
    assert "--feature-file" in training.stdout
    assert "--train-split" in training.stdout


def test_pipeline_persists_minimized_trace_for_failed_downstream_step(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_pipeline as run_pipeline

    case_path = tmp_path / "case.json"
    output_path = tmp_path / "trace.json"
    case_path.write_text(
        json.dumps(
            {
                "case_id": "opaque-case-9",
                "clinical_facts": {"note": "CLI_RAW_TEXT_SENTINEL"},
                "mri_manifest": [{"path": "/synthetic-storage/CLI_MRI_PATH_SENTINEL/t1.npy"}],
            }
        ),
        encoding="utf-8",
    )
    failed_state = (
        CaseState.from_input(CaseInput(case_id="opaque-case-9"))
        .with_diagnostic(
            DiagnosticResult(
                status=StepStatus("completed"), tumor_type="tumor_2", who_grade="grade_3"
            )
        )
        .with_treatment(TreatmentResult(status=StepStatus("failed", "API request failed")))
    )

    class StubOrchestrator:
        def __init__(self, client, **kwargs) -> None:
            pass

        def run(self, case_input) -> CaseState:
            return failed_state

    monkeypatch.setattr(run_pipeline, "MidlineOrchestrator", StubOrchestrator)
    monkeypatch.setattr(run_pipeline, "OpenAiResponsesClient", lambda **kwargs: object())
    monkeypatch.setattr(run_pipeline, "_build_diagnostic_workflow", lambda args: object())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--case-json",
            str(case_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = run_pipeline.main()

    persisted = output_path.read_text(encoding="utf-8")
    payload = json.loads(persisted)
    assert payload["case_id"].startswith("sha256:")
    assert payload["case_id"] != "opaque-case-9"
    assert payload["steps"][2]["status"]["state"] == "failed"
    assert "case_input" not in payload
    assert "CLI_RAW_TEXT_SENTINEL" not in persisted
    assert "CLI_MRI_PATH_SENTINEL" not in persisted
    assert exit_code == 1


def test_pipeline_writes_failed_trace_when_openai_configuration_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_pipeline as run_pipeline

    case_path = tmp_path / "case.json"
    output_path = tmp_path / "trace.json"
    case_path.write_text(json.dumps({"case_id": "opaque-case"}), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--case-json", str(case_path), "--output", str(output_path)],
    )

    assert run_pipeline.main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["steps"][2]["status"]["state"] == "failed"
