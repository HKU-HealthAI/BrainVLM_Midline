from pathlib import Path

import pytest
import torch
from torch import nn

from midline_agent.diagnostic.checkpoints import (
    BrainVlmCheckpointError,
    load_brainvlm_checkpoint,
)


class TinyStateModel(nn.Module):
    def __init__(self, modern_peft: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        if modern_peft:
            self.register_parameter("mapped", nn.Parameter(torch.zeros(1)))

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if "mapped" in state:
            state["adapter.lora_A.default.weight"] = state.pop("mapped")
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        translated = dict(state_dict)
        if "adapter.lora_A.default.weight" in translated:
            translated["mapped"] = translated.pop("adapter.lora_A.default.weight")
        return super().load_state_dict(translated, strict=strict, assign=assign)


def _save(path: Path, state: dict[str, object]) -> None:
    torch.save({"model": state, "epoch": 4}, path)


def test_checkpoint_loader_requires_exact_manifest_and_loads_values(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    _save(path, {"weight": torch.full((2, 3), 2.0)})
    model = TinyStateModel()

    loaded = load_brainvlm_checkpoint(path, model, manifest={"weight": (2, 3)})

    assert loaded == ("weight",)
    torch.testing.assert_close(model.weight, torch.full((2, 3), 2.0))


def test_checkpoint_loader_maps_legacy_lora_names_to_modern_peft(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    _save(
        path,
        {
            "weight": torch.ones(2, 3),
            "adapter.lora_A.weight": torch.tensor([3.0]),
        },
    )
    model = TinyStateModel(modern_peft=True)
    manifest = {"weight": (2, 3), "adapter.lora_A.weight": (1,)}

    loaded = load_brainvlm_checkpoint(path, model, manifest=manifest)

    assert loaded == ("adapter.lora_A.weight", "weight")
    torch.testing.assert_close(model.mapped, torch.tensor([3.0]))


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "missing"),
        ({"weight": torch.ones(2, 3), "other": torch.ones(1)}, "unexpected"),
        ({"weight": torch.ones(3, 2)}, "shape"),
        ({"weight": "bad"}, "tensor"),
        ({"weight": torch.full((2, 3), float("nan"))}, "finite"),
    ],
)
def test_checkpoint_loader_rejects_invalid_trainable_state(
    tmp_path: Path,
    state: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "model.pth"
    _save(path, state)

    with pytest.raises(BrainVlmCheckpointError, match=message):
        load_brainvlm_checkpoint(path, TinyStateModel(), manifest={"weight": (2, 3)})


def test_checkpoint_loader_rejects_missing_model_envelope(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    torch.save({"weight": torch.ones(2, 3)}, path)

    with pytest.raises(BrainVlmCheckpointError, match="model"):
        load_brainvlm_checkpoint(path, TinyStateModel(), manifest={"weight": (2, 3)})
