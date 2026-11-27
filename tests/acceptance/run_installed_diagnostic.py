#!/usr/bin/env python3
"""Run checkpoint-backed Diagnostic orchestration from an installed package."""

from __future__ import annotations

import argparse
import json
import os
import re
import site
import sys
import sysconfig
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHA256_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
TRACE_PATH_ARGUMENTS = (
    "annotations",
    "image_root",
    "case_json",
    "llama_model",
    "biomedclip_pretrained",
    "tumor_checkpoint",
    "mil_checkpoint",
    "who_checkpoint",
    "output",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--biomedclip-pretrained")
    parser.add_argument("--tumor-checkpoint", type=Path, required=True)
    parser.add_argument("--mil-checkpoint", type=Path, required=True)
    parser.add_argument("--who-checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-tumor",
        choices=tuple(f"tumor_{index}" for index in range(1, 6)),
        required=True,
    )
    parser.add_argument(
        "--expected-grade",
        choices=("grade_2", "grade_3", "grade_4"),
        required=True,
    )
    return parser


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _active_site_package_roots() -> tuple[Path, ...]:
    prefix = Path(sys.prefix).resolve()
    candidates = {
        Path(value).resolve()
        for value in (
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
            *site.getsitepackages(),
        )
        if value
    }
    roots = tuple(sorted((path for path in candidates if _is_beneath(path, prefix)), key=str))
    if not roots:
        raise AssertionError("active environment has no site-packages directory")
    return roots


def _assert_installed_module_origins() -> None:
    roots = _active_site_package_roots()
    checked = 0
    for name, module in tuple(sys.modules.items()):
        if name != "midline_agent" and not name.startswith("midline_agent."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        checked += 1
        resolved = Path(origin).resolve()
        if not any(_is_beneath(resolved, root) for root in roots):
            raise AssertionError(f"{name} was not imported from the active environment")
    if checked == 0:
        raise AssertionError("no midline_agent modules were loaded")


def _json_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        strings = tuple(str(key) for key in value)
        return strings + tuple(
            item
            for child in value.values()
            for item in _json_strings(child)
        )
    if isinstance(value, (list, tuple)):
        return tuple(item for child in value for item in _json_strings(child))
    return ()


def _argument_variants(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = os.fspath(value)
    variants = {raw, str(Path(raw).expanduser().resolve())}
    return tuple(sorted((item for item in variants if item), key=len, reverse=True))


def _assert_trace_privacy(
    serialized: str,
    args: argparse.Namespace,
    raw_case_id: str,
) -> None:
    try:
        trace_strings = _json_strings(json.loads(serialized))
    except (TypeError, json.JSONDecodeError) as exc:
        raise AssertionError("trace is not valid privacy-minimized JSON") from exc
    inspectable_strings = tuple(
        value
        for value in trace_strings
        if SHA256_FINGERPRINT_PATTERN.fullmatch(value) is None
    )
    forbidden = {"raw_generation", "https://", raw_case_id}
    for field in TRACE_PATH_ARGUMENTS:
        forbidden.update(_argument_variants(getattr(args, field, None)))
    for candidate in forbidden:
        if not candidate:
            continue
        if any(
            value == candidate or (len(candidate) >= 4 and candidate in value)
            for value in inspectable_strings
        ):
            raise AssertionError("trace is not privacy-minimized")


def _write_json_atomic(output: Path, payload: object) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if _is_beneath(output, REPOSITORY_ROOT.resolve()):
        raise ValueError("acceptance output must be outside the repository")

    from midline_agent.datasets.inference_annotations import load_inference_case
    from midline_agent.diagnostic.brainvlm_runtime import BrainVlmRuntime
    from midline_agent.diagnostic.logits_mil import LogitsMilPredictor
    from midline_agent.diagnostic.standalone import (
        TumorPredictionService,
        WhoGradePredictionService,
    )
    from midline_agent.diagnostic.workflow import DiagnosticWorkflow
    from midline_agent.orchestrator import MidlineOrchestrator
    from midline_agent.schema import CaseInput, StepStatus, SurvivalResult
    from midline_agent.treatment.client import LlmResponse

    payload = json.loads(args.case_json.read_text(encoding="ascii"))
    if payload != {"case_id": "BT0156", "clinical_facts": {}, "mri_manifest": []}:
        raise ValueError("case JSON must be the label-free public Diagnostic case")
    case_input = CaseInput(
        case_id=payload["case_id"],
        clinical_facts=payload["clinical_facts"],
        mri_manifest=tuple(payload["mri_manifest"]),
    )
    inference_case = load_inference_case(args.annotations, args.image_root)
    tumor_runtime = BrainVlmRuntime.load(
        args.llama_model,
        args.tumor_checkpoint,
        device=args.device,
        biomedclip_pretrained=args.biomedclip_pretrained,
    )
    tumor_service = TumorPredictionService(
        runtime=tumor_runtime,
        mil_predictor=LogitsMilPredictor(args.mil_checkpoint),
        model_checkpoint=args.tumor_checkpoint,
        mil_checkpoint=args.mil_checkpoint,
    )

    def who_runtime() -> BrainVlmRuntime:
        return BrainVlmRuntime.load(
            args.llama_model,
            args.who_checkpoint,
            device=args.device,
            biomedclip_pretrained=args.biomedclip_pretrained,
        )

    who_service = WhoGradePredictionService(who_runtime, args.who_checkpoint)

    class DeterministicClient:
        def generate(self, request):
            if request.schema_name == "midline_handoff":
                return LlmResponse(
                    "",
                    {"proceed": True, "reason": "deterministic acceptance"},
                )
            return LlmResponse(
                "",
                {
                    "summary": "No network acceptance report.",
                    "recommendations": [],
                    "uncertainty": "No treatment evidence was supplied.",
                    "evidence": [],
                },
            )

    def survival_workflow(case, diagnostic):
        del case, diagnostic
        return SurvivalResult(
            status=StepStatus("completed"),
            risk_score=1.0,
            median_survival_time=12.0,
            time_points=(6.0, 12.0),
            survival_probabilities=(0.8, 0.6),
            provenance={"model_identifier": "deterministic-acceptance"},
        )

    state = MidlineOrchestrator(
        DeterministicClient(),
        diagnostic_workflow=DiagnosticWorkflow(inference_case, tumor_service, who_service),
        survival_workflow=survival_workflow,
    ).run(case_input)
    if state.diagnostic.tumor_type != args.expected_tumor:
        raise AssertionError("unexpected tumor prediction")
    if state.diagnostic.who_grade != args.expected_grade:
        raise AssertionError("unexpected WHO grade prediction")
    trace = state.to_trace()
    trace_payload = trace.to_dict()
    if [step.status.state for step in trace.steps] != ["completed", "completed", "completed"]:
        raise AssertionError("orchestrator did not complete all injected acceptance stages")
    serialized = json.dumps(trace_payload, sort_keys=True, allow_nan=False)
    _assert_trace_privacy(serialized, args, payload["case_id"])
    _assert_installed_module_origins()
    _write_json_atomic(output, trace_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
