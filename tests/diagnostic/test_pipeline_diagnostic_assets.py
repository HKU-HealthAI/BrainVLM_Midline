import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "annotations": tmp_path / "annotations.json",
        "image_root": tmp_path / "images",
        "llama_model": tmp_path / "llama",
        "biomedclip_pretrained": "local-biomedclip",
        "tumor_checkpoint": tmp_path / "tumor.pth",
        "mil_checkpoint": tmp_path / "mil.pth",
        "who_checkpoint": tmp_path / "who.pth",
        "device": "cuda:3",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pipeline_help_exposes_shared_diagnostic_assets_and_no_legacy_factory() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--case-json",
        "--output",
        "--annotations",
        "--image-root",
        "--llama-model",
        "--biomedclip-pretrained",
        "--tumor-checkpoint",
        "--mil-checkpoint",
        "--who-checkpoint",
        "--device",
    ):
        assert flag in completed.stdout
    for removed in (
        "--adapter",
        "--source-root",
        "--checkpoint ",
        "--attention-mil-checkpoint",
        "--model-factory",
    ):
        assert removed not in completed.stdout


def test_pipeline_source_has_no_adapter_attention_mil_or_path_injection() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_pipeline.py").read_text(encoding="ascii")

    for removed in (
        "sys.path",
        "load_brainvlm_adapter",
        "AttentionMilPredictor",
        "BrainVlmAttentionMilWorkflow",
        "source_root",
        "model_factory",
    ):
        assert removed not in source


def test_build_diagnostic_workflow_constructs_exact_shared_services_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.predict_tumor as tumor_command
    import scripts.run_pipeline as command

    inference_case = object()
    tumor_runtime = object()
    who_runtime = object()
    mil_predictor = object()
    observed: dict[str, object] = {"runtime_calls": []}

    assert command._load_brainvlm_runtime is tumor_command._load_brainvlm_runtime
    monkeypatch.setattr(command, "load_inference_case", lambda annotations, image_root: inference_case)

    def load_runtime(llama_model, checkpoint, *, device, biomedclip_pretrained):
        observed["runtime_calls"].append(
            (llama_model, checkpoint, device, biomedclip_pretrained)
        )
        return tumor_runtime if checkpoint.name == "tumor.pth" else who_runtime

    monkeypatch.setattr(command, "_load_brainvlm_runtime", load_runtime)

    def load_mil(path):
        observed["mil_checkpoint"] = path
        return mil_predictor

    monkeypatch.setattr(command, "LogitsMilPredictor", load_mil)

    class FakeTumorService:
        def __init__(self, **kwargs) -> None:
            observed["tumor_service"] = kwargs

    class FakeWhoService:
        def __init__(self, **kwargs) -> None:
            observed["who_service"] = kwargs

    class FakeWorkflow:
        def __init__(self, case, tumor_service, who_service) -> None:
            observed["workflow"] = (case, tumor_service, who_service)

    monkeypatch.setattr(command, "TumorPredictionService", FakeTumorService)
    monkeypatch.setattr(command, "WhoGradePredictionService", FakeWhoService)
    monkeypatch.setattr(command, "DiagnosticWorkflow", FakeWorkflow)
    args = _args(tmp_path)

    workflow = command._build_diagnostic_workflow(args)

    assert isinstance(workflow, FakeWorkflow)
    assert observed["runtime_calls"] == [
        (
            args.llama_model,
            args.tumor_checkpoint,
            "cuda:3",
            "local-biomedclip",
        )
    ]
    assert observed["mil_checkpoint"] == args.mil_checkpoint
    assert observed["tumor_service"] == {
        "runtime": tumor_runtime,
        "mil_predictor": mil_predictor,
        "model_checkpoint": args.tumor_checkpoint,
        "mil_checkpoint": args.mil_checkpoint,
    }
    who_service = observed["who_service"]
    assert who_service["model_checkpoint"] == args.who_checkpoint
    assert who_service["runtime_factory"]() is who_runtime
    assert observed["runtime_calls"][-1] == (
        args.llama_model,
        args.who_checkpoint,
        "cuda:3",
        "local-biomedclip",
    )


def test_build_diagnostic_workflow_accepts_no_assets_as_unconfigured(tmp_path: Path) -> None:
    import scripts.run_pipeline as command

    args = _args(
        tmp_path,
        annotations=None,
        image_root=None,
        llama_model=None,
        biomedclip_pretrained=None,
        tumor_checkpoint=None,
        mil_checkpoint=None,
        who_checkpoint=None,
        device=None,
    )

    assert command._build_diagnostic_workflow(args) is None


@pytest.mark.parametrize(
    "provided",
    [
        "annotations",
        "image_root",
        "llama_model",
        "biomedclip_pretrained",
        "tumor_checkpoint",
        "mil_checkpoint",
        "who_checkpoint",
        "device",
    ],
)
def test_build_diagnostic_workflow_rejects_every_partial_asset_configuration(
    tmp_path: Path,
    provided: str,
) -> None:
    import scripts.run_pipeline as command

    full = _args(tmp_path)
    values = {name: None for name in vars(full)}
    values[provided] = getattr(full, provided)

    with pytest.raises(ValueError, match="diagnostic workflow requires"):
        command._build_diagnostic_workflow(argparse.Namespace(**values))


def test_pipeline_turns_partial_diagnostic_assets_into_parser_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_pipeline as command

    case_path = tmp_path / "case.json"
    case_path.write_text('{"case_id": "patient-a"}\n', encoding="ascii")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--case-json",
            str(case_path),
            "--output",
            str(tmp_path / "trace.json"),
            "--annotations",
            str(tmp_path / "annotations.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        command.main()

    assert exc_info.value.code == 2


def test_pipeline_executes_configured_diagnostic_without_openai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_pipeline as command

    case_path = tmp_path / "case.json"
    output = tmp_path / "trace.json"
    case_path.write_text('{"case_id": "patient-a"}\n', encoding="ascii")
    calls: list[str] = []

    def workflow(case_input):
        calls.append(case_input.case_id)
        return DiagnosticResult(
            status=StepStatus("completed"),
            tumor_type="tumor_4",
            who_grade="grade_3",
            probabilities=(0.05, 0.05, 0.05, 0.8, 0.05),
        )

    monkeypatch.setattr(command, "_build_diagnostic_workflow", lambda args: workflow)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--case-json", str(case_path), "--output", str(output)],
    )

    assert command.main() == 0
    assert calls == ["patient-a"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [step["status"]["state"] for step in payload["steps"]] == [
        "completed",
        "unavailable",
        "unavailable",
    ]
    assert payload["steps"][0]["result"]["tumor_type"] == "tumor_4"


def test_pipeline_trace_write_is_atomic(tmp_path: Path) -> None:
    import scripts.run_pipeline as command

    state = CaseState.from_input(CaseInput(case_id="patient-a"))
    state = state.with_diagnostic(DiagnosticResult.unavailable("not configured"))
    state = state.with_survival(SurvivalResult.unavailable())
    state = state.with_treatment(
        TreatmentResult(status=StepStatus("unavailable", "OpenAI is not configured"))
    )
    output = tmp_path / "nested" / "trace.json"

    command._write_trace(output, state)

    assert json.loads(output.read_text(encoding="utf-8"))["steps"][0]["step"] == "diagnostic"
    assert not list(output.parent.glob("*.tmp"))


def test_public_case_json_is_label_free_and_contains_no_paths_or_private_fields() -> None:
    path = PROJECT_ROOT / "test_data" / "utsw_glioma_bt0156" / "case.json"
    payload = json.loads(path.read_text(encoding="ascii"))

    assert payload == {"case_id": "BT0156", "clinical_facts": {}, "mri_manifest": []}
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        "tumor",
        "grade",
        "label",
        "prediction",
        "result",
        "path",
        "url",
        "private",
    ):
        assert forbidden not in serialized


def test_installed_diagnostic_acceptance_runner_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tests.acceptance.run_installed_diagnostic", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--annotations",
        "--images",
        "--case-json",
        "--llama-model",
        "--biomedclip-pretrained",
        "--tumor-checkpoint",
        "--mil-checkpoint",
        "--who-checkpoint",
        "--device",
        "--output",
        "--expected-tumor",
        "--expected-grade",
    ):
        assert flag in completed.stdout
