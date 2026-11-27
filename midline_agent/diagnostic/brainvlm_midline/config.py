"""Validated constants and asset configuration for BrainVLM-Midline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


BIOMEDCLIP_MODEL_NAME = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


@dataclass(frozen=True)
class ModelSettings:
    image_size: int = 224
    vision_width: int = 768
    llama_width: int = 4096
    compatibility_width: int = 1408
    llama_layers: int = 32
    key_value_width: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    special_tokens: tuple[str, ...] = ("<box>", "<Img>", "</Img>", "<t>")
    pad_token: str = "$$"


@dataclass(frozen=True)
class FoundationAssets:
    llama_model: Path
    checkpoint: Path
    biomedclip_pretrained: str

    @classmethod
    def from_paths(
        cls,
        llama_model: Path,
        checkpoint: Path,
        biomedclip_pretrained: str | None = None,
    ) -> "FoundationAssets":
        llama_path = Path(llama_model).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not llama_path.is_dir():
            raise ValueError("llama model must be an existing directory")
        if not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing file")

        pretrained = biomedclip_pretrained or BIOMEDCLIP_MODEL_NAME
        if pretrained != BIOMEDCLIP_MODEL_NAME:
            pretrained_path = Path(pretrained).expanduser().resolve()
            if not pretrained_path.is_file():
                raise ValueError("biomedclip pretrained asset must be an existing file")
            pretrained = str(pretrained_path)
        return cls(
            llama_model=llama_path,
            checkpoint=checkpoint_path,
            biomedclip_pretrained=pretrained,
        )


def _build_checkpoint_manifest(settings: ModelSettings) -> dict[str, tuple[int, ...]]:
    manifest: dict[str, tuple[int, ...]] = {}
    prefix = "llama_model.base_model.model.model.layers"
    for layer in range(settings.llama_layers):
        attention = f"{prefix}.{layer}.self_attn"
        manifest[f"{attention}.q_proj.lora_A.weight"] = (
            settings.lora_rank,
            settings.llama_width,
        )
        manifest[f"{attention}.q_proj.lora_B.weight"] = (
            settings.llama_width,
            settings.lora_rank,
        )
        manifest[f"{attention}.v_proj.lora_A.weight"] = (
            settings.lora_rank,
            settings.llama_width,
        )
        manifest[f"{attention}.v_proj.lora_B.weight"] = (
            settings.key_value_width,
            settings.lora_rank,
        )

    manifest.update(
        {
            "visual_encoder.norm.weight": (settings.vision_width,),
            "visual_encoder.norm.bias": (settings.vision_width,),
            "llama_proj_mlps.0.weight": (
                settings.llama_width,
                settings.vision_width,
            ),
            "llama_proj_mlps.0.bias": (settings.llama_width,),
            "llama_proj_mlps.2.weight": (
                settings.llama_width,
                settings.llama_width,
            ),
            "llama_proj_mlps.2.bias": (settings.llama_width,),
            "v_q_project.weight": (
                settings.compatibility_width,
                settings.vision_width,
            ),
            "v_q_project.bias": (settings.compatibility_width,),
        }
    )
    for name in ("Wq", "Wk", "Wv", "output_linear"):
        manifest[f"cross_attention.{name}.weight"] = (
            settings.vision_width,
            settings.vision_width,
        )
        manifest[f"cross_attention.{name}.bias"] = (settings.vision_width,)
    manifest["cross_attention.norm.weight"] = (settings.vision_width,)
    manifest["cross_attention.norm.bias"] = (settings.vision_width,)
    return manifest


_DEFAULT_SETTINGS = ModelSettings()
CHECKPOINT_MANIFEST: Mapping[str, tuple[int, ...]] = MappingProxyType(
    _build_checkpoint_manifest(_DEFAULT_SETTINGS)
)


def checkpoint_manifest(
    settings: ModelSettings,
) -> Mapping[str, tuple[int, ...]]:
    if settings == _DEFAULT_SETTINGS:
        return CHECKPOINT_MANIFEST
    return MappingProxyType(_build_checkpoint_manifest(settings))
