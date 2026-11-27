import json
import subprocess
import sys
from pathlib import Path

import pytest

from midline_agent.schema import (
    CaseInput,
    DiagnosticResult,
    StepStatus,
    SurvivalResult,
    TreatmentResult,
)
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


def test_standalone_diagnostic_cli_help_exposes_all_required_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = {
        "scripts/predict_tumor.py": (
            "--annotations",
            "--image-root",
            "--brainvlm-source",
            "--llama-model",
            "--checkpoint",
            "--mil-checkpoint",
            "--output",
            "--device",
        ),
        "scripts/predict_who_grade.py": (
            "--annotations",
            "--image-root",
            "--brainvlm-source",
            "--llama-model",
            "--checkpoint",
            "--tumor-result",
            "--tumor-type",
            "--output",
            "--device",
        ),
    }
    for script, expected in commands.items():
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert all(flag in completed.stdout for flag in expected)


class _FakeResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def test_tumor_cli_wires_runtime_service_and_writes_json_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.predict_tumor as command

    paths = {name: tmp_path / name for name in ("annotations", "images", "source", "llama", "model", "mil")}
    output = tmp_path / "nested" / "tumor.json"
    case = object()
    runtime = object()
    predictor = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(command, "load_inference_case", lambda annotations, image_root: case)
    monkeypatch.setattr(command.BrainVlmRuntime, "load", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(command, "LogitsMilPredictor", lambda path: predictor)

    class FakeService:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)

        def predict(self, value):
            assert value is case
            return _FakeResult({"patient_id": "patient-a", "tumor_type": "tumor_4"})

    monkeypatch.setattr(command, "TumorPredictionService", FakeService)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_tumor.py",
            "--annotations",
            str(paths["annotations"]),
            "--image-root",
            str(paths["images"]),
            "--brainvlm-source",
            str(paths["source"]),
            "--llama-model",
            str(paths["llama"]),
            "--checkpoint",
            str(paths["model"]),
            "--mil-checkpoint",
            str(paths["mil"]),
            "--output",
            str(output),
            "--device",
            "cuda:0",
        ],
    )

    assert command.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["tumor_type"] == "tumor_4"
    assert observed["runtime"] is runtime
    assert observed["mil_predictor"] is predictor
    assert not list(output.parent.glob("*.tmp"))


def test_who_cli_requires_one_tumor_source_and_uses_result_patient_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.predict_who_grade as command

    tumor_result = tmp_path / "tumor.json"
    tumor_result.write_text(
        json.dumps({"patient_id": "patient-a", "tumor_type": "tumor_4"}),
        encoding="utf-8",
    )
    output = tmp_path / "who.json"
    case = type("Case", (), {"patient_id": "patient-a"})()
    observed: dict[str, object] = {}
    monkeypatch.setattr(command, "load_inference_case", lambda annotations, image_root: case)

    class FakeService:
        def __init__(self, runtime_factory, model_checkpoint) -> None:
            observed["factory"] = runtime_factory
            observed["checkpoint"] = model_checkpoint

        def predict(self, value, tumor_type):
            assert value is case
            observed["tumor_type"] = tumor_type
            return _FakeResult({"patient_id": "patient-a", "who_grade": "grade_3"})

    monkeypatch.setattr(command, "WhoGradePredictionService", FakeService)
    required = [
        "--annotations",
        str(tmp_path / "annotations.json"),
        "--image-root",
        str(tmp_path / "images"),
        "--brainvlm-source",
        str(tmp_path / "source"),
        "--llama-model",
        str(tmp_path / "llama"),
        "--checkpoint",
        str(tmp_path / "who.pth"),
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    monkeypatch.setattr(sys, "argv", ["predict_who_grade.py", *required, "--tumor-result", str(tumor_result)])

    assert command.main() == 0
    assert observed["tumor_type"] == "tumor_4"
    assert json.loads(output.read_text(encoding="utf-8"))["who_grade"] == "grade_3"

    monkeypatch.setattr(sys, "argv", ["predict_who_grade.py", *required])
    with pytest.raises(SystemExit):
        command.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict_who_grade.py", *required, "--tumor-result", str(tumor_result), "--tumor-type", "tumor_4"],
    )
    with pytest.raises(SystemExit):
        command.main()


def test_who_cli_rejects_tumor_result_for_another_patient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.predict_who_grade as command

    tumor_result = tmp_path / "tumor.json"
    tumor_result.write_text(
        json.dumps({"patient_id": "patient-b", "tumor_type": "tumor_4"}),
        encoding="utf-8",
    )
    case = type("Case", (), {"patient_id": "patient-a"})()
    monkeypatch.setattr(command, "load_inference_case", lambda annotations, image_root: case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_who_grade.py",
            "--annotations",
            str(tmp_path / "annotations.json"),
            "--image-root",
            str(tmp_path / "images"),
            "--brainvlm-source",
            str(tmp_path / "source"),
            "--llama-model",
            str(tmp_path / "llama"),
            "--checkpoint",
            str(tmp_path / "who.pth"),
            "--tumor-result",
            str(tumor_result),
            "--output",
            str(tmp_path / "who.json"),
            "--device",
            "cuda:0",
        ],
    )

    with pytest.raises(ValueError, match="patient_id"):
        command.main()


def test_standalone_commands_do_not_read_openai_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("predict_tumor.py", "predict_who_grade.py"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "OPENAI" not in source


def test_readme_documents_reproducible_checkpoint_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="ascii")

    required = (
        "https://github.com/HKU-HealthAI/BrainVLM",
        "168Bp-JgH9CheV9QYj0MLoolSG_ONFuDD",
        "1eWBI0DTWuwhQv5kErNfcDzMCmdSplwKL",
        "15fch44MAuqOE4g4vuPegwAaFZn2EDk-M",
        "Llama 3.1-8B Instruct",
        "BiomedCLIP",
        "test_data/utsw_glioma_bt0323/annotations.json",
        "scripts/run_tumor_prediction.sh",
        "scripts/run_who_grading.sh",
        "scripts/run_tumor_prediction.sbatch",
        "scripts/run_who_grading.sbatch",
        "tumor_4",
        "grade_3",
        "No generated prediction is bundled",
        "https://doi.org/10.7937/DFAE-1B86",
    )

    assert all(item in readme for item in required)
    assert "Python 3.9 or newer" in readme
    assert "The public CLI does not expose a separate WHO-checkpoint argument" not in readme
    assert "no patient data or example cases are bundled" not in readme


def test_data_schema_separates_label_free_inference_from_training_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = (root / "docs/data-schema.md").read_text(encoding="ascii")

    assert "Label-free inference records" in schema
    assert "Training annotation records" in schema
    assert "tumor_type" in schema
    assert "not accepted by the inference loader" in schema


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
        .with_survival(
            SurvivalResult.unavailable("survival checkpoint unavailable")
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
