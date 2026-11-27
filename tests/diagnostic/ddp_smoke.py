#!/usr/bin/env python3
"""Two-rank CPU smoke test for the public BrainVLM training command."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

import scripts.train_brainvlm_midline as command
from midline_agent.diagnostic.brainvlm_midline.training import (
    BrainVlmTrainingConfig,
    configure_trainable_parameters,
)


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.epoch = 0

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": torch.tensor([float(index + 1), 2.0])}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TinyDdpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.llama_proj_mlps = nn.Sequential(
            nn.Linear(2, 2),
            nn.GELU(),
            nn.Linear(2, 2),
        )

    def forward(self, batch, reduction="mean"):
        value = self.llama_proj_mlps(batch["image"])
        loss = value.square().mean(dim=1)
        return {"loss": loss.mean() if reduction == "mean" else loss}


def _manifest(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
        if name.startswith("llama_proj_mlps.")
    }


def _config(output: Path) -> BrainVlmTrainingConfig:
    return BrainVlmTrainingConfig.from_mapping(
        {
            "task": "tumor",
            "annotations": ["unused.json"],
            "image_root": "unused-images",
            "llama_model": "unused-llama",
            "biomedclip_pretrained": "unused-biomedclip",
            "output_dir": str(output),
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "min_learning_rate": 0.0,
            "warmup_learning_rate": 0.0,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "iterations_per_epoch": 1,
            "gradient_accumulation_steps": 1,
            "seed": 5,
            "num_workers": 0,
            "use_age": False,
            "upsample_ratio": 0.0,
            "amp": False,
            "device": "cuda:0",
            "distributed": True,
            "world_size": 2,
            "dist_url": "env://",
        }
    )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            mode="w",
            encoding="ascii",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _verify(output: Path) -> None:
    deadline = time.monotonic() + 10.0
    markers: list[Path] = []
    while time.monotonic() < deadline:
        markers = sorted(output.glob("rank-*.json"))
        if len(markers) == 2:
            break
        time.sleep(0.05)
    assert [path.name for path in markers] == ["rank-0.json", "rank-1.json"]
    values = [json.loads(path.read_text(encoding="ascii")) for path in markers]
    assert {value["rank"] for value in values} == {0, 1}
    assert {value["world_size"] for value in values} == {2}
    assert {value["optimizer_steps"] for value in values} == {1}
    assert values[0]["mean_loss"] == values[1]["mean_loss"]
    checkpoints = sorted(output.glob("*.pth"))
    assert [path.name for path in checkpoints] == ["checkpoint_last.pth"]
    assert not list(output.glob("*rank-1*.pth"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = TinyDdpModel()
    configure_trainable_parameters(model)
    command._build_dataset = lambda config: TinyDataset()
    command._build_model = lambda config: model
    command._resolve_device = lambda *args, **kwargs: torch.device("cpu")
    command.collate_task_samples = lambda samples: {
        "image": torch.stack([sample["image"] for sample in samples])
    }
    command.CHECKPOINT_MANIFEST = _manifest(model)

    result = command.run_training(_config(args.output))
    _atomic_json(
        args.output / f"rank-{result.rank}.json",
        {
            "rank": result.rank,
            "world_size": result.world_size,
            "optimizer_steps": result.optimizer_steps,
            "mean_loss": result.mean_loss,
        },
    )
    if result.rank == 0:
        _verify(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
