#!/usr/bin/env python3
"""Run the public Diagnostic, Survival, and Treatment workflow."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from midline_agent.datasets.inference_annotations import load_inference_case
from midline_agent.diagnostic.brainvlm_runtime import BrainVlmRuntime
from midline_agent.diagnostic.logits_mil import LogitsMilPredictor
from midline_agent.diagnostic.standalone import (
    TumorPredictionService,
    WhoGradePredictionService,
)
from midline_agent.diagnostic.workflow import DiagnosticWorkflow
from midline_agent.orchestrator import MidlineOrchestrator
from midline_agent.schema import CaseInput, CaseTrace
from midline_agent.state import CaseState
from midline_agent.treatment.client import OpenAiResponsesClient
from scripts.predict_tumor import _load_brainvlm_runtime


_DIAGNOSTIC_ASSETS = (
    ("annotations", "annotations"),
    ("image_root", "image root"),
    ("llama_model", "Llama model"),
    ("tumor_checkpoint", "tumor checkpoint"),
    ("mil_checkpoint", "MIL checkpoint"),
    ("who_checkpoint", "WHO checkpoint"),
    ("device", "device"),
)


def _build_diagnostic_workflow(args: argparse.Namespace) -> DiagnosticWorkflow | None:
    values = {
        attribute: getattr(args, attribute, None)
        for attribute, _ in _DIAGNOSTIC_ASSETS
    }
    optional_biomedclip = getattr(args, "biomedclip_pretrained", None)
    configured = any(value is not None for value in values.values()) or optional_biomedclip is not None
    if not configured:
        return None
    missing = [
        label
        for attribute, label in _DIAGNOSTIC_ASSETS
        if values[attribute] is None
    ]
    if missing:
        raise ValueError("diagnostic workflow requires " + ", ".join(missing))

    inference_case = load_inference_case(args.annotations, args.image_root)
    tumor_runtime = _load_brainvlm_runtime(
        args.llama_model,
        args.tumor_checkpoint,
        device=args.device,
        biomedclip_pretrained=optional_biomedclip,
    )
    tumor_service = TumorPredictionService(
        runtime=tumor_runtime,
        mil_predictor=LogitsMilPredictor(args.mil_checkpoint),
        model_checkpoint=args.tumor_checkpoint,
        mil_checkpoint=args.mil_checkpoint,
    )

    def who_runtime_factory() -> BrainVlmRuntime:
        return _load_brainvlm_runtime(
            args.llama_model,
            args.who_checkpoint,
            device=args.device,
            biomedclip_pretrained=optional_biomedclip,
        )

    who_service = WhoGradePredictionService(
        runtime_factory=who_runtime_factory,
        model_checkpoint=args.who_checkpoint,
    )
    return DiagnosticWorkflow(inference_case, tumor_service, who_service)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="Label-free single-patient JSON")
    parser.add_argument("--image-root", type=Path, help="Root for annotation image paths")
    parser.add_argument("--llama-model", type=Path, help="Llama 3.1-8B Instruct directory")
    parser.add_argument(
        "--biomedclip-pretrained",
        help="Optional local BiomedCLIP pretrained file or open_clip identifier",
    )
    parser.add_argument(
        "--tumor-checkpoint",
        type=Path,
        help="BrainVLM-Midline tumor checkpoint",
    )
    parser.add_argument(
        "--mil-checkpoint",
        type=Path,
        help="Released learnable-logits MIL checkpoint",
    )
    parser.add_argument(
        "--who-checkpoint",
        type=Path,
        help="BrainVLM-Midline WHO checkpoint",
    )
    parser.add_argument("--device", help="Indexed CUDA device, for example cuda:0")
    return parser


def _load_case_input(path: Path) -> CaseInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("case JSON must contain an object")
    return CaseInput(
        case_id=str(payload["case_id"]),
        clinical_facts=dict(payload.get("clinical_facts", {})),
        mri_manifest=tuple(payload.get("mri_manifest", [])),
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        case_input = _load_case_input(args.case_json)
        diagnostic_workflow = _build_diagnostic_workflow(args)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    client = (
        OpenAiResponsesClient(api_key=api_key, model=model)
        if api_key and model
        else None
    )
    state = MidlineOrchestrator(
        client,
        diagnostic_workflow=diagnostic_workflow,
    ).run(case_input)
    trace = _write_trace(args.output, state)
    return int(any(record.status.state == "failed" for record in trace.steps))


def _write_trace(output: Path, state: CaseState) -> CaseTrace:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = state.to_trace()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(trace.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return trace


if __name__ == "__main__":
    raise SystemExit(main())
