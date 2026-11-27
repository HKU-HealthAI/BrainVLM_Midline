from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from midline_agent.diagnostic.brainvlm_midline.config import BIOMEDCLIP_MODEL_NAME
from midline_agent.diagnostic.brainvlm_midline.vision import (
    BIOMEDCLIP_MEAN,
    BIOMEDCLIP_STD,
    VisionEncodingError,
    create_biomedclip_vision,
    encode_low_resolution,
    preprocess_image,
)


class TinyVisionTrunk(nn.Module):
    def __init__(self, *, tokens: int = 197, width: int = 768) -> None:
        super().__init__()
        self.tokens = tokens
        self.width = width
        self.stem = nn.Linear(3, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        values = torch.arange(
            batch * self.tokens * self.width,
            dtype=image.dtype,
            device=image.device,
        ).reshape(batch, self.tokens, self.width)
        tokens = self.norm(values)
        return tokens[:, 0]


def _write_rgb(path: Path, color: tuple[int, int, int] = (0, 64, 128)) -> None:
    Image.new("RGB", (32, 16), color=color).save(path)


def test_preprocess_image_uses_bicubic_rgb_and_biomedclip_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slice.png"
    _write_rgb(path)

    tensor = preprocess_image(path)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert bool(torch.isfinite(tensor).all())
    expected = (
        torch.tensor([0.0, 64 / 255, 128 / 255])
        - torch.tensor(BIOMEDCLIP_MEAN)
    ) / torch.tensor(BIOMEDCLIP_STD)
    torch.testing.assert_close(tensor[:, 0, 0], expected)


def test_preprocess_image_rejects_an_undecodable_file(tmp_path: Path) -> None:
    path = tmp_path / "slice.png"
    path.write_text("not an image", encoding="ascii")

    with pytest.raises(VisionEncodingError, match="decode"):
        preprocess_image(path)


def test_create_biomedclip_vision_freezes_everything_except_final_norm() -> None:
    trunk = TinyVisionTrunk()
    calls: list[tuple[str, str]] = []

    def factory(model_name: str, *, pretrained: str):
        calls.append((model_name, pretrained))
        return SimpleNamespace(visual=SimpleNamespace(trunk=trunk)), None, None

    result = create_biomedclip_vision("/models/biomedclip.bin", factory=factory)

    assert result is trunk
    assert calls == [(BIOMEDCLIP_MODEL_NAME, "/models/biomedclip.bin")]
    assert trunk.norm.weight.requires_grad
    assert trunk.norm.bias.requires_grad
    assert not trunk.stem.weight.requires_grad
    assert not trunk.stem.bias.requires_grad


def test_create_biomedclip_vision_rejects_an_incompatible_openclip_model() -> None:
    def factory(model_name: str, *, pretrained: str):
        del model_name, pretrained
        return object(), None, None

    with pytest.raises(VisionEncodingError, match="vision trunk"):
        create_biomedclip_vision("asset.bin", factory=factory)


def test_low_resolution_encoding_uses_final_norm_patches_without_cls() -> None:
    trunk = TinyVisionTrunk()
    image = torch.zeros(2, 3, 224, 224)
    captured: dict[str, torch.Tensor] = {}

    def capture(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        del module, inputs
        captured["tokens"] = output.detach().clone()

    handle = trunk.norm.register_forward_hook(capture)
    trunk(image)
    handle.remove()
    expected = captured["tokens"][:, 1:, :].mean(dim=1)

    observed = encode_low_resolution(trunk, image)

    assert observed.shape == (2, 768)
    torch.testing.assert_close(observed, expected)
    assert not trunk.norm._forward_hooks


@pytest.mark.parametrize(
    ("tokens", "width", "message"),
    [(196, 768, "197"), (197, 4, "768")],
)
def test_low_resolution_encoding_rejects_incompatible_hook_shapes(
    tokens: int,
    width: int,
    message: str,
) -> None:
    trunk = TinyVisionTrunk(tokens=tokens, width=width)

    with pytest.raises(VisionEncodingError, match=message):
        encode_low_resolution(trunk, torch.zeros(1, 3, 224, 224))

    assert not trunk.norm._forward_hooks


def test_low_resolution_encoding_rejects_nonfinite_tokens() -> None:
    class NonFiniteTrunk(TinyVisionTrunk):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            batch = image.shape[0]
            values = torch.full(
                (batch, 197, 768),
                float("nan"),
                device=image.device,
            )
            tokens = self.norm(values)
            return tokens[:, 0]

    with pytest.raises(VisionEncodingError, match="finite"):
        encode_low_resolution(
            NonFiniteTrunk(),
            torch.zeros(1, 3, 224, 224),
        )
