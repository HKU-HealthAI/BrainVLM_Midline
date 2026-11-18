"""Strict safetensors checkpoints for the public AttentionMIL contract."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import SafetensorError

from .attention_mil import AttentionMIL, MilOutput
from .artifact_io import (
    ARTIFACT_ID_PATTERN,
    SHA256_PATTERN,
    ArtifactBindingError,
    load_bound_safetensors,
    write_bound_safetensors,
)


CHECKPOINT_FORMAT = "midline-agent-attention-mil-checkpoint"
CHECKPOINT_FORMAT_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "1.0"
FEATURE_KINDS = frozenset({"logits", "hidden", "hidden_and_logits"})
MODEL_CONFIG_KEYS = frozenset(
    {
        "logit_dimension",
        "input_hidden_dimension",
        "hidden_dimension",
        "num_modalities",
        "modality_embedding_dimension",
        "num_classes",
        "dropout",
    }
)
CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "format",
        "format_version",
        "model_config",
        "feature_schema_version",
        "feature_kind",
        "logit_dimension",
        "hidden_dimension",
        "modality_map",
        "train_split_sha256",
        "validation_split_sha256",
        "best_epoch",
        "validation_loss",
        "artifact_id",
        "tensor_sha256",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CheckpointError(ValueError):
    pass


def _require_path(path: Path) -> None:
    if path.suffix != ".safetensors":
        raise CheckpointError("checkpoint path must use the .safetensors extension")


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        raise CheckpointError("checkpoint metadata must be a mapping")
    missing = CHECKPOINT_METADATA_KEYS - metadata.keys()
    if missing:
        raise CheckpointError(f"checkpoint metadata missing: {', '.join(sorted(missing))}")
    unexpected = metadata.keys() - CHECKPOINT_METADATA_KEYS
    if unexpected:
        raise CheckpointError(f"checkpoint metadata contains unexpected fields: {', '.join(sorted(unexpected))}")
    if metadata["format"] != CHECKPOINT_FORMAT or metadata["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("unsupported checkpoint format")
    if not isinstance(metadata["artifact_id"], str) or ARTIFACT_ID_PATTERN.fullmatch(metadata["artifact_id"]) is None:
        raise CheckpointError("checkpoint artifact_id is invalid")
    if not isinstance(metadata["tensor_sha256"], str) or SHA256_PATTERN.fullmatch(metadata["tensor_sha256"]) is None:
        raise CheckpointError("checkpoint tensor_sha256 is invalid")
    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise CheckpointError("unsupported feature schema version")
    feature_kind = metadata["feature_kind"]
    if feature_kind not in FEATURE_KINDS:
        raise CheckpointError("unsupported feature_kind")
    model_config = metadata["model_config"]
    if not isinstance(model_config, dict) or set(model_config) != MODEL_CONFIG_KEYS:
        raise CheckpointError("model_config must be a mapping")
    for key in ("logit_dimension", "input_hidden_dimension"):
        if type(model_config[key]) is not int or model_config[key] < 0:
            raise CheckpointError(f"model_config {key} must be a non-negative integer")
    if model_config["logit_dimension"] + model_config["input_hidden_dimension"] == 0:
        raise CheckpointError("model_config requires logits, hidden features, or both")
    for key in ("hidden_dimension", "num_modalities", "modality_embedding_dimension", "num_classes"):
        if type(model_config[key]) is not int or model_config[key] <= 0:
            raise CheckpointError(f"model_config {key} must be a positive integer")
    dropout = model_config["dropout"]
    if type(dropout) not in {int, float} or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise CheckpointError("model_config dropout must be finite and in the interval [0, 1)")
    for key in ("logit_dimension", "hidden_dimension"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise CheckpointError(f"checkpoint {key} must be a non-negative integer")
    expects_logits = feature_kind in {"logits", "hidden_and_logits"}
    expects_hidden = feature_kind in {"hidden", "hidden_and_logits"}
    if expects_logits != (metadata["logit_dimension"] > 0) or expects_hidden != (metadata["hidden_dimension"] > 0):
        raise CheckpointError("feature_kind is inconsistent with feature dimensions")
    if model_config["logit_dimension"] != metadata["logit_dimension"]:
        raise CheckpointError("checkpoint logit dimensions are inconsistent")
    if model_config["input_hidden_dimension"] != metadata["hidden_dimension"]:
        raise CheckpointError("checkpoint hidden dimensions are inconsistent")
    modality_map = metadata["modality_map"]
    if (
        not isinstance(modality_map, dict)
        or not modality_map
        or any(not isinstance(name, str) or not name for name in modality_map)
        or any(type(value) is not int or value < 0 for value in modality_map.values())
        or set(modality_map.values()) != set(range(len(modality_map)))
        or model_config["num_modalities"] != len(modality_map)
    ):
        raise CheckpointError("checkpoint modality metadata are inconsistent")
    for key in ("train_split_sha256", "validation_split_sha256"):
        value = metadata[key]
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise CheckpointError(f"checkpoint {key} must be a SHA-256 hash")
    if type(metadata["best_epoch"]) is not int or metadata["best_epoch"] < 0:
        raise CheckpointError("checkpoint best_epoch must be a non-negative integer")
    validation_loss = metadata["validation_loss"]
    if type(validation_loss) not in {int, float} or not math.isfinite(validation_loss) or validation_loss < 0:
        raise CheckpointError("checkpoint validation_loss must be finite and non-negative")


def _validate_state_tensors(model: AttentionMIL, state: Mapping[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    if set(state) != set(expected):
        raise CheckpointError("checkpoint tensor names do not match model_config")
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if tensor.shape != expected_tensor.shape:
            raise CheckpointError(f"checkpoint tensor {name} has an incompatible shape")
        if tensor.dtype != expected_tensor.dtype or (expected_tensor.is_floating_point() and not tensor.is_floating_point()):
            raise CheckpointError(f"checkpoint tensor {name} has an incompatible dtype")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise CheckpointError(f"checkpoint tensor {name} contains non-finite values")


def save_attention_mil_checkpoint(
    path: Path,
    model: AttentionMIL,
    metadata: Mapping[str, Any],
) -> None:
    path = Path(path)
    _require_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = dict(metadata)
    sidecar.update(
        {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": model.configuration(),
            "artifact_id": "0" * 32,
            "tensor_sha256": "0" * 64,
        }
    )
    _validate_metadata(sidecar)
    try:
        encoded = json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint metadata must be JSON serializable") from exc
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    _validate_state_tensors(model, state)

    def build_sidecar(artifact_id: str, tensor_sha256: str) -> dict[str, Any]:
        bound_sidecar = dict(sidecar)
        bound_sidecar["artifact_id"] = artifact_id
        bound_sidecar["tensor_sha256"] = tensor_sha256
        _validate_metadata(bound_sidecar)
        return bound_sidecar

    try:
        write_bound_safetensors(
            path,
            _sidecar_path(path),
            state,
            artifact_format=CHECKPOINT_FORMAT,
            format_version=CHECKPOINT_FORMAT_VERSION,
            build_sidecar=build_sidecar,
        )
    except (ArtifactBindingError, OSError, TypeError, ValueError) as exc:
        raise CheckpointError("unable to write AttentionMIL checkpoint") from exc


def load_attention_mil_checkpoint(path: Path) -> tuple[AttentionMIL, dict[str, Any]]:
    path = Path(path)
    _require_path(path)
    try:
        metadata = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
        _validate_metadata(metadata)
        state = load_bound_safetensors(
            path,
            metadata,
            artifact_format=CHECKPOINT_FORMAT,
            format_version=CHECKPOINT_FORMAT_VERSION,
        )
    except ArtifactBindingError as exc:
        raise CheckpointError("checkpoint artifact binding failed") from exc
    except CheckpointError as exc:
        raise CheckpointError(f"unable to read AttentionMIL checkpoint: {exc}") from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise CheckpointError("unable to read AttentionMIL checkpoint") from exc
    try:
        model = AttentionMIL(**metadata["model_config"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid model_config") from exc
    _validate_state_tensors(model, state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError("checkpoint tensors do not match model_config") from exc
    model.eval()
    return model, metadata


class AttentionMilPredictor:
    def __init__(self, checkpoint_path: Path) -> None:
        self.model, self.metadata = load_attention_mil_checkpoint(checkpoint_path)

    def _validate_feature_contract(self, feature_metadata: Mapping[str, Any]) -> None:
        for key in (
            "schema_version",
            "feature_kind",
            "logit_dimension",
            "hidden_dimension",
            "modality_map",
        ):
            checkpoint_key = "feature_schema_version" if key == "schema_version" else key
            if feature_metadata.get(key) != self.metadata.get(checkpoint_key):
                raise ValueError(f"feature {key} does not match checkpoint")

    def predict_patient(self, patient: Mapping[str, Any], feature_metadata: Mapping[str, Any]) -> MilOutput:
        self._validate_feature_contract(feature_metadata)
        logits = patient.get("logits")
        hidden = patient.get("hidden")
        if logits is not None:
            logits = logits.unsqueeze(0)
        if hidden is not None:
            hidden = hidden.unsqueeze(0)
        modalities = torch.as_tensor(patient["modalities"], dtype=torch.long).unsqueeze(0)
        mask = torch.as_tensor(patient["valid_mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            return self.model(logits=logits, hidden=hidden, modalities=modalities, mask=mask)
