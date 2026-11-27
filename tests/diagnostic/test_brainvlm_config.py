from pathlib import Path

import pytest

from midline_agent.diagnostic.brainvlm_midline.config import (
    BIOMEDCLIP_MODEL_NAME,
    CHECKPOINT_MANIFEST,
    FoundationAssets,
    ModelSettings,
    checkpoint_manifest,
)


def _write_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "config.json").write_text("{}\n", encoding="ascii")
    biomedclip = tmp_path / "biomedclip.bin"
    biomedclip.write_bytes(b"weights")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    return llama, biomedclip, checkpoint


def test_foundation_assets_resolve_existing_local_assets(tmp_path: Path) -> None:
    llama, biomedclip, checkpoint = _write_assets(tmp_path)

    assets = FoundationAssets.from_paths(
        llama_model=llama,
        checkpoint=checkpoint,
        biomedclip_pretrained=str(biomedclip),
    )

    assert assets.llama_model == llama.resolve()
    assert assets.checkpoint == checkpoint.resolve()
    assert assets.biomedclip_pretrained == str(biomedclip.resolve())


def test_foundation_assets_allow_the_packaged_biomedclip_identifier(tmp_path: Path) -> None:
    llama, _, checkpoint = _write_assets(tmp_path)

    assets = FoundationAssets.from_paths(llama, checkpoint, BIOMEDCLIP_MODEL_NAME)

    assert assets.biomedclip_pretrained == BIOMEDCLIP_MODEL_NAME


@pytest.mark.parametrize("field", ["llama", "checkpoint", "biomedclip"])
def test_foundation_assets_reject_missing_or_wrong_asset_types(
    tmp_path: Path,
    field: str,
) -> None:
    llama, biomedclip, checkpoint = _write_assets(tmp_path)
    if field == "llama":
        llama = tmp_path / "missing-llama"
    elif field == "checkpoint":
        checkpoint = tmp_path / "missing-checkpoint"
    else:
        biomedclip = tmp_path / "missing-biomedclip"

    with pytest.raises(ValueError, match=field):
        FoundationAssets.from_paths(llama, checkpoint, str(biomedclip))


def test_model_settings_match_the_released_checkpoint_architecture() -> None:
    settings = ModelSettings()

    assert settings.image_size == 224
    assert settings.vision_width == 768
    assert settings.llama_width == 4096
    assert settings.compatibility_width == 1408
    assert settings.llama_layers == 32
    assert settings.key_value_width == 1024
    assert settings.lora_rank == 8
    assert settings.lora_alpha == 32
    assert settings.lora_dropout == pytest.approx(0.05)
    assert settings.lora_targets == ("q_proj", "v_proj")
    assert settings.special_tokens == ("<box>", "<Img>", "</Img>", "<t>")
    assert settings.pad_token == "$$"


def test_checkpoint_manifest_contains_all_released_trainable_tensors() -> None:
    manifest = checkpoint_manifest(ModelSettings())

    assert manifest is CHECKPOINT_MANIFEST
    assert len(manifest) == 146
    assert manifest["visual_encoder.norm.weight"] == (768,)
    assert manifest["visual_encoder.norm.bias"] == (768,)
    assert manifest["llama_proj_mlps.0.weight"] == (4096, 768)
    assert manifest["llama_proj_mlps.2.weight"] == (4096, 4096)
    assert manifest["v_q_project.weight"] == (1408, 768)
    assert manifest["cross_attention.Wq.weight"] == (768, 768)
    assert manifest["cross_attention.norm.bias"] == (768,)
    assert (
        manifest[
            "llama_model.base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight"
        ]
        == (8, 4096)
    )
    assert (
        manifest[
            "llama_model.base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
        ]
        == (1024, 8)
    )


def test_checkpoint_manifest_is_immutable() -> None:
    with pytest.raises(TypeError):
        CHECKPOINT_MANIFEST["new.weight"] = (1,)  # type: ignore[index]
