#!/usr/bin/env python3
"""Run standalone tumor-conditioned WHO grade prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from midline_agent.datasets.inference_annotations import load_inference_case
from midline_agent.diagnostic.brainvlm_runtime import BrainVlmRuntime
from midline_agent.diagnostic.logits_mil import TUMOR_TYPES
from midline_agent.diagnostic.standalone import WhoGradePredictionService
from scripts.predict_tumor import _load_brainvlm_runtime, _write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Label-free single-patient JSON")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for annotation image paths")
    parser.add_argument("--llama-model", type=Path, required=True, help="Llama 3.1-8B Instruct directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="WHO Grading checkpoint")
    tumor_source = parser.add_mutually_exclusive_group(required=True)
    tumor_source.add_argument("--tumor-result", type=Path, help="JSON produced by midline-predict-tumor")
    tumor_source.add_argument("--tumor-type", choices=TUMOR_TYPES, help="Explicit canonical patient tumor type")
    parser.add_argument("--output", type=Path, required=True, help="Destination patient result JSON")
    parser.add_argument("--device", required=True, help="Indexed CUDA device, for example cuda:0")
    parser.add_argument(
        "--biomedclip-pretrained",
        help="Optional local BiomedCLIP pretrained file or open_clip identifier",
    )
    return parser


def _tumor_type_from_result(path: Path, patient_id: str) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read tumor result JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("tumor result JSON must contain an object")
    if payload.get("patient_id") != patient_id:
        raise ValueError("tumor result patient_id does not match the inference case")
    tumor_type = payload.get("tumor_type")
    if tumor_type not in TUMOR_TYPES:
        raise ValueError("tumor result contains an unsupported tumor type")
    return tumor_type


def main() -> int:
    args = _parser().parse_args()
    case = load_inference_case(args.annotations, args.image_root)
    tumor_type = args.tumor_type or _tumor_type_from_result(args.tumor_result, case.patient_id)

    def runtime_factory() -> BrainVlmRuntime:
        return _load_brainvlm_runtime(
            args.llama_model,
            args.checkpoint,
            device=args.device,
            biomedclip_pretrained=args.biomedclip_pretrained,
        )

    result = WhoGradePredictionService(
        runtime_factory=runtime_factory,
        model_checkpoint=args.checkpoint,
    ).predict(case, tumor_type)
    _write_json_atomic(args.output, result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
