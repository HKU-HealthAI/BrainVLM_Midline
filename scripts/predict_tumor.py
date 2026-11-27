#!/usr/bin/env python3
"""Run standalone BrainVLM-Midline and MIL tumor prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midline_agent.datasets.inference_annotations import load_inference_case
from midline_agent.diagnostic.brainvlm_runtime import BrainVlmRuntime
from midline_agent.diagnostic.logits_mil import LogitsMilPredictor
from midline_agent.diagnostic.standalone import TumorPredictionService


def _write_json_atomic(output: Path, payload: dict[str, object]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Label-free single-patient JSON")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for annotation image paths")
    parser.add_argument("--brainvlm-source", type=Path, required=True, help="External BrainVLM checkout")
    parser.add_argument("--llama-model", type=Path, required=True, help="Llama 3.1-8B Instruct directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="BrainVLM-Midline checkpoint")
    parser.add_argument("--mil-checkpoint", type=Path, required=True, help="Released learnable-logits MIL checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination patient result JSON")
    parser.add_argument("--device", required=True, help="Indexed CUDA device, for example cuda:0")
    parser.add_argument(
        "--biomedclip-pretrained",
        help="Optional local BiomedCLIP pretrained file or open_clip identifier",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    case = load_inference_case(args.annotations, args.image_root)
    runtime = BrainVlmRuntime.load(
        args.brainvlm_source,
        args.llama_model,
        args.checkpoint,
        device=args.device,
        biomedclip_pretrained=args.biomedclip_pretrained,
    )
    predictor = LogitsMilPredictor(args.mil_checkpoint)
    result = TumorPredictionService(
        runtime=runtime,
        mil_predictor=predictor,
        model_checkpoint=args.checkpoint,
        mil_checkpoint=args.mil_checkpoint,
    ).predict(case)
    _write_json_atomic(args.output, result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
