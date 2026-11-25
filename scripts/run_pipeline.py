#!/usr/bin/env python3
"""Run the public workflow with the official OpenAI Responses API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midline_agent.orchestrator import MidlineOrchestrator
from midline_agent.schema import (
    CaseInput,
    CaseTrace,
    DiagnosticResult,
    StepStatus,
    SurvivalResult,
    TreatmentResult,
)
from midline_agent.state import CaseState
from midline_agent.treatment.client import OpenAiResponsesClient
from midline_agent.datasets.annotations import load_annotations
from midline_agent.diagnostic.brainvlm_adapter import load_brainvlm_adapter
from midline_agent.diagnostic.checkpoints import AttentionMilPredictor
from midline_agent.diagnostic.workflow import BrainVlmAttentionMilWorkflow


def _build_diagnostic_workflow(args: argparse.Namespace) -> BrainVlmAttentionMilWorkflow:
    required = {
        "annotations": args.annotations,
        "image root": args.image_root,
        "external adapter": args.adapter,
        "source root": args.source_root,
        "BrainVLM checkpoint": args.checkpoint,
        "AttentionMIL checkpoint": args.attention_mil_checkpoint,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("diagnostic workflow requires " + ", ".join(missing))
    return BrainVlmAttentionMilWorkflow(
        records=tuple(load_annotations(args.annotations, args.image_root)),
        adapter=load_brainvlm_adapter(args.source_root, args.checkpoint, args.adapter),
        predictor=AttentionMilPredictor(args.attention_mil_checkpoint),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--adapter", help="External factory import path, formatted module:callable")
    parser.add_argument("--source-root", type=Path, help="External BrainVLM source checkout")
    parser.add_argument("--checkpoint", type=Path, help="External BrainVLM checkpoint path")
    parser.add_argument("--attention-mil-checkpoint", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    payload = json.loads(args.case_json.read_text(encoding="utf-8"))
    case_input = CaseInput(
        case_id=str(payload["case_id"]),
        clinical_facts=dict(payload.get("clinical_facts", {})),
        mri_manifest=tuple(payload.get("mri_manifest", [])),
    )
    if not api_key or not model:
        state = CaseState.from_input(case_input).with_diagnostic(
            DiagnosticResult.unavailable("OpenAI configuration is missing")
        ).with_survival(
            SurvivalResult.unavailable("OpenAI configuration is missing")
        ).with_treatment(
            TreatmentResult(status=StepStatus("failed", "OpenAI configuration is missing"))
        )
        _write_trace(args.output, state)
        return 1
    client = OpenAiResponsesClient(api_key=api_key, model=model)
    try:
        diagnostic_workflow = _build_diagnostic_workflow(args)
    except ValueError as exc:
        parser.error(str(exc))
    state = MidlineOrchestrator(
        client,
        diagnostic_workflow=diagnostic_workflow,
    ).run(case_input)
    trace = _write_trace(args.output, state)
    return int(any(record.status.state == "failed" for record in trace.steps))


def _write_trace(output: Path, state: CaseState) -> CaseTrace:
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = state.to_trace()
    output.write_text(
        json.dumps(trace.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return trace


if __name__ == "__main__":
    raise SystemExit(main())
