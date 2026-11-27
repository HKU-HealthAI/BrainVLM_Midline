"""BiomedCLIP preprocessing and low-resolution feature extraction."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from ...datasets.brainvlm import preprocess_slice
from .config import BIOMEDCLIP_MODEL_NAME


BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_EXPECTED_TOKENS = 197
_EXPECTED_WIDTH = 768


class VisionEncodingError(RuntimeError):
    """Raised when a vision asset or encoded feature violates the contract."""


def preprocess_image(path: Path | str, *, image_size: int = 224) -> torch.Tensor:
    """Decode one RGB image and apply the checkpoint's BiomedCLIP transform."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise VisionEncodingError(f"could not decode image: {image_path}") from exc

    try:
        processed = preprocess_slice(image, size=image_size).image
    except ValueError as exc:
        raise VisionEncodingError(f"could not preprocess image: {image_path}") from exc

    tensor = torch.from_numpy(processed.copy())
    mean = tensor.new_tensor(BIOMEDCLIP_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(BIOMEDCLIP_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise VisionEncodingError("preprocessed image must contain finite float32 values")
    return tensor


def create_biomedclip_vision(
    pretrained: str,
    *,
    factory: Callable[..., Any] | None = None,
) -> nn.Module:
    """Create the checkpoint-compatible BiomedCLIP trunk and set trainability."""
    if factory is None:
        try:
            from open_clip import create_model_and_transforms
        except ImportError as exc:
            raise VisionEncodingError(
                "open_clip_torch is required to create the BiomedCLIP vision trunk"
            ) from exc
        factory = create_model_and_transforms

    try:
        created = factory(BIOMEDCLIP_MODEL_NAME, pretrained=pretrained)
        model = created[0] if isinstance(created, tuple) else created
        trunk = model.visual.trunk
    except (AttributeError, IndexError, TypeError) as exc:
        raise VisionEncodingError(
            "BiomedCLIP factory did not return a compatible vision trunk"
        ) from exc

    if not isinstance(trunk, nn.Module) or not isinstance(
        getattr(trunk, "norm", None), nn.Module
    ):
        raise VisionEncodingError("BiomedCLIP model has no compatible vision trunk")
    norm = trunk.norm
    if not isinstance(getattr(norm, "weight", None), nn.Parameter) or not isinstance(
        getattr(norm, "bias", None), nn.Parameter
    ):
        raise VisionEncodingError("BiomedCLIP vision trunk has no trainable final norm")

    trunk.requires_grad_(False)
    norm.weight.requires_grad_(True)
    norm.bias.requires_grad_(True)
    return trunk


def encode_low_resolution(trunk: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Mean-pool final-normalized patch tokens while excluding the CLS token."""
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, 224, 224):
        raise VisionEncodingError("image batch must have shape [B, 3, 224, 224]")
    if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
        raise VisionEncodingError("image batch must contain finite floating-point values")

    norm = getattr(trunk, "norm", None)
    if not isinstance(norm, nn.Module):
        raise VisionEncodingError("vision trunk has no final norm module")
    captured: list[torch.Tensor] = []

    def capture_tokens(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        captured.append(output)

    handle = norm.register_forward_hook(capture_tokens)
    try:
        trunk(image)
    finally:
        handle.remove()

    if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
        raise VisionEncodingError("vision final norm did not emit one token tensor")
    tokens = captured[0]
    if tokens.ndim != 3 or tokens.shape[1] != _EXPECTED_TOKENS:
        raise VisionEncodingError(
            f"vision final norm must emit {_EXPECTED_TOKENS} tokens"
        )
    if tokens.shape[2] != _EXPECTED_WIDTH:
        raise VisionEncodingError(
            f"vision final norm token width must be {_EXPECTED_WIDTH}"
        )
    if tokens.shape[0] != image.shape[0]:
        raise VisionEncodingError("vision token batch size does not match the image batch")
    if not bool(torch.isfinite(tokens).all()):
        raise VisionEncodingError("vision tokens must contain only finite values")
    return tokens[:, 1:, :].mean(dim=1)
