#!/usr/bin/env python3
"""Two-rank CPU smoke test for the public BrainVLM training core."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from midline_agent.diagnostic.brainvlm_midline.training import (
    WarmupCosineScheduler,
    build_optimizer,
    configure_trainable_parameters,
    save_training_checkpoint,
    train_epoch,
)


class TinyDdpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.llama_proj_mlps = nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.Linear(2, 2))

    def forward(self, batch, reduction="mean"):
        value = self.llama_proj_mlps(batch["image"])
        loss = value.square().mean(dim=1)
        return {"loss": loss.mean() if reduction == "mean" else loss}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    torch.manual_seed(5)
    model = TinyDdpModel()
    configure_trainable_parameters(model)
    wrapped = nn.parallel.DistributedDataParallel(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=1,
        minimum_learning_rate=0.0,
    )
    result = train_epoch(
        wrapped,
        [{"image": torch.tensor([[1.0, 2.0]])}],
        optimizer,
        scheduler,
        scaler=None,
        gradient_accumulation_steps=1,
        device=torch.device("cpu"),
        amp=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "optimizer_steps": result.optimizer_steps,
                "mean_loss": result.mean_loss,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    if rank == 0:
        manifest = {
            name: tuple(value.shape)
            for name, value in model.state_dict().items()
            if name.startswith("llama_proj_mlps.")
        }
        save_training_checkpoint(
            args.output / "checkpoint.pth",
            model,
            optimizer,
            scaler=None,
            epoch=0,
            global_step=1,
            config={"task": "tumor"},
            manifest=manifest,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
