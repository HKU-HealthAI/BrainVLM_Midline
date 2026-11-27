#!/usr/bin/env python3
"""Train AttentionMIL from a user-owned extracted feature store."""

from __future__ import annotations

import argparse
from pathlib import Path

from midline_agent.diagnostic.feature_store import load_feature_store
from midline_agent.diagnostic.mil_training import train_attention_mil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    store = load_feature_store(args.feature_file)
    train_attention_mil(
        store=store,
        train_split_path=args.train_split,
        validation_split_path=args.validation_split,
        checkpoint_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
