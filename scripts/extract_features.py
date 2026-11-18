#!/usr/bin/env python3
"""Extract patient-level slice logits from a user-supplied BrainVLM adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midline_agent.datasets.annotations import load_annotations
from midline_agent.diagnostic.brainvlm_adapter import load_brainvlm_adapter
from midline_agent.diagnostic.feature_store import save_feature_store
from midline_agent.diagnostic.workflow import build_feature_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="External factory import path, formatted module:callable")
    parser.add_argument("--source-root", type=Path, required=True, help="External BrainVLM source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="External checkpoint path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_annotations(args.annotations, args.image_root)
    adapter = load_brainvlm_adapter(args.source_root, args.checkpoint, args.adapter)
    store = build_feature_store(records, adapter)
    save_feature_store(args.output, store)


if __name__ == "__main__":
    main()
