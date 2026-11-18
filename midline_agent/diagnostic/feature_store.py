"""Versioned, pickle-free storage for extracted slice features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import SafetensorError

from .artifact_io import ArtifactBindingError, load_bound_safetensors, write_bound_safetensors


FEATURE_STORE_FORMAT = "midline-agent-feature-store"
FEATURE_STORE_FORMAT_VERSION = "1.0"
FEATURE_KINDS = frozenset({"logits", "hidden", "hidden_and_logits"})
TUMOR_CLASS_COUNT = 5
FEATURE_SIDECAR_KEYS = frozenset(
    {"format", "format_version", "artifact_id", "tensor_sha256", "metadata", "patients"}
)


class FeatureSchemaError(ValueError):
    pass


def _require_safetensors_path(path: Path) -> None:
    if path.suffix != ".safetensors":
        raise FeatureSchemaError("feature store path must use the .safetensors extension")


def _validate_metadata(metadata: Any) -> tuple[set[int], tuple[str, ...]]:
    if not isinstance(metadata, Mapping):
        raise FeatureSchemaError("metadata must be a mapping")
    required = {"schema_version", "feature_kind", "logit_dimension", "hidden_dimension", "modality_map"}
    missing = required - metadata.keys()
    if missing:
        raise FeatureSchemaError(f"metadata missing: {', '.join(sorted(missing))}")
    if str(metadata["schema_version"]).split(".", 1)[0] != "1":
        raise FeatureSchemaError("unsupported feature schema major version")
    kind = metadata["feature_kind"]
    if kind not in FEATURE_KINDS:
        raise FeatureSchemaError("unsupported feature_kind")
    for key in ("logit_dimension", "hidden_dimension"):
        value = metadata[key]
        if type(value) is not int or value < 0:
            raise FeatureSchemaError(f"{key} must be a non-negative integer")
    expects_logits = kind in {"logits", "hidden_and_logits"}
    expects_hidden = kind in {"hidden", "hidden_and_logits"}
    if expects_logits != (metadata["logit_dimension"] > 0):
        raise FeatureSchemaError("feature_kind and logit_dimension are inconsistent")
    if expects_hidden != (metadata["hidden_dimension"] > 0):
        raise FeatureSchemaError("feature_kind and hidden_dimension are inconsistent")
    modality_map = metadata["modality_map"]
    if not isinstance(modality_map, Mapping) or not modality_map:
        raise FeatureSchemaError("modality_map must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in modality_map):
        raise FeatureSchemaError("modality_map names must be non-empty strings")
    modality_values = tuple(modality_map.values())
    if any(type(value) is not int or value < 0 for value in modality_values):
        raise FeatureSchemaError("modality_map values must be non-negative integers")
    if set(modality_values) != set(range(len(modality_values))):
        raise FeatureSchemaError("modality_map values must be unique and contiguous")
    tensor_names = tuple(name for name in ("logits", "hidden") if name in kind)
    return set(modality_values), tensor_names


def _validate_tensor(patient_id: str, name: str, tensor: Any, rows: int, dimension: int) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise FeatureSchemaError(f"patient {patient_id} {name} must be a rank-2 tensor")
    if tensor.shape != (rows, dimension):
        raise FeatureSchemaError(f"patient {patient_id} {name} has wrong shape")
    if tensor.dtype != torch.float32:
        raise FeatureSchemaError(f"patient {patient_id} {name} must use float32")
    if not bool(torch.isfinite(tensor).all()):
        raise FeatureSchemaError(f"patient {patient_id} {name} must contain only finite values")


def validate_feature_store(store: Any) -> None:
    if not isinstance(store, Mapping):
        raise FeatureSchemaError("feature store must be a mapping")
    metadata = store.get("metadata")
    modality_values, tensor_names = _validate_metadata(metadata)
    patients = store.get("patients")
    if not isinstance(patients, Mapping) or not patients:
        raise FeatureSchemaError("feature store must contain at least one patient")
    for patient_id, patient in patients.items():
        if not isinstance(patient_id, str) or not patient_id:
            raise FeatureSchemaError("patient IDs must be non-empty strings")
        if not isinstance(patient, Mapping):
            raise FeatureSchemaError(f"patient {patient_id} must be a mapping")
        if "label" in patient and (
            type(patient["label"]) is not int or not 0 <= patient["label"] < TUMOR_CLASS_COUNT
        ):
            raise FeatureSchemaError(f"patient {patient_id} label must be an integer in the supported class range")
        required = {"slice_ids", "modalities", "valid_mask", "slice_metadata", *tensor_names}
        if required - patient.keys():
            raise FeatureSchemaError(f"patient {patient_id} missing required fields")
        unexpected_tensors = ({"logits", "hidden"} & patient.keys()) - set(tensor_names)
        if unexpected_tensors:
            raise FeatureSchemaError(f"patient {patient_id} contains tensors not declared by feature_kind")
        slice_ids = patient["slice_ids"]
        modalities = patient["modalities"]
        valid_mask = patient["valid_mask"]
        slice_metadata = patient["slice_metadata"]
        if not all(isinstance(value, list) for value in (slice_ids, modalities, valid_mask, slice_metadata)):
            raise FeatureSchemaError(f"patient {patient_id} aligned fields must be lists")
        rows = len(slice_ids)
        if rows == 0:
            raise FeatureSchemaError(f"patient {patient_id} must contain a non-empty bag")
        if len({rows, len(modalities), len(valid_mask), len(slice_metadata)}) != 1:
            raise FeatureSchemaError(f"patient {patient_id} aligned slice fields have inconsistent lengths")
        if any(type(value) is not bool for value in valid_mask):
            raise FeatureSchemaError(f"patient {patient_id} valid_mask must contain boolean values")
        if not any(valid_mask):
            raise FeatureSchemaError(f"patient {patient_id} must contain at least one valid slice")
        if any(type(value) is not int or value not in modality_values | {-1} for value in modalities):
            raise FeatureSchemaError(f"patient {patient_id} contains an unsupported modality")
        if any(valid and modality not in modality_values for valid, modality in zip(valid_mask, modalities, strict=True)):
            raise FeatureSchemaError(f"patient {patient_id} contains an invalid active modality")
        if len(set(slice_ids)) != rows or any(not isinstance(value, str) or not value for value in slice_ids):
            raise FeatureSchemaError(f"patient {patient_id} slice_ids must be unique non-empty strings")
        for name in tensor_names:
            dimension = metadata["logit_dimension" if name == "logits" else "hidden_dimension"]
            _validate_tensor(patient_id, name, patient[name], rows, dimension)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_feature_store(path: Path, store: dict[str, Any]) -> None:
    path = Path(path)
    _require_safetensors_path(path)
    validate_feature_store(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    patients: dict[str, dict[str, Any]] = {}
    for index, (patient_id, patient) in enumerate(sorted(store["patients"].items())):
        tensor_keys: dict[str, str] = {}
        serializable = {key: value for key, value in patient.items() if not isinstance(value, torch.Tensor)}
        for name in ("logits", "hidden"):
            if name in patient:
                tensor_key = f"patient.{index}.{name}"
                tensors[tensor_key] = patient[name].detach().cpu().contiguous()
                tensor_keys[name] = tensor_key
        serializable["tensor_keys"] = tensor_keys
        patients[patient_id] = serializable
    def build_sidecar(artifact_id: str, tensor_sha256: str) -> dict[str, Any]:
        return {
            "format": FEATURE_STORE_FORMAT,
            "format_version": FEATURE_STORE_FORMAT_VERSION,
            "artifact_id": artifact_id,
            "tensor_sha256": tensor_sha256,
            "metadata": dict(store["metadata"]),
            "patients": patients,
        }

    try:
        write_bound_safetensors(
            path,
            _sidecar_path(path),
            tensors,
            artifact_format=FEATURE_STORE_FORMAT,
            format_version=FEATURE_STORE_FORMAT_VERSION,
            build_sidecar=build_sidecar,
        )
    except (ArtifactBindingError, OSError, TypeError, ValueError) as exc:
        raise FeatureSchemaError("unable to write feature store") from exc


def load_feature_store(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require_safetensors_path(path)
    try:
        sidecar = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict) or set(sidecar) != FEATURE_SIDECAR_KEYS:
            raise FeatureSchemaError("invalid feature store sidecar fields")
        tensors = load_bound_safetensors(
            path,
            sidecar,
            artifact_format=FEATURE_STORE_FORMAT,
            format_version=FEATURE_STORE_FORMAT_VERSION,
        )
    except ArtifactBindingError as exc:
        raise FeatureSchemaError("unable to read feature store: artifact binding failed") from exc
    except FeatureSchemaError:
        raise
    except (OSError, SafetensorError, ValueError) as exc:
        raise FeatureSchemaError("unable to read feature store") from exc
    if sidecar.get("format") != FEATURE_STORE_FORMAT or sidecar.get("format_version") != FEATURE_STORE_FORMAT_VERSION:
        raise FeatureSchemaError("unsupported feature store format")
    patients: dict[str, dict[str, Any]] = {}
    used_tensor_keys: set[str] = set()
    for patient_id, stored_patient in sidecar.get("patients", {}).items():
        patient = {key: value for key, value in stored_patient.items() if key != "tensor_keys"}
        tensor_keys = stored_patient.get("tensor_keys", {})
        if not isinstance(tensor_keys, dict):
            raise FeatureSchemaError(f"patient {patient_id} has invalid tensor metadata")
        for name, tensor_key in tensor_keys.items():
            if name not in {"logits", "hidden"} or tensor_key not in tensors or tensor_key in used_tensor_keys:
                raise FeatureSchemaError(f"patient {patient_id} references an invalid tensor")
            patient[name] = tensors[tensor_key]
            used_tensor_keys.add(tensor_key)
        patients[patient_id] = patient
    if used_tensor_keys != set(tensors):
        raise FeatureSchemaError("feature store contains unreferenced tensors")
    store = {"metadata": sidecar.get("metadata"), "patients": patients}
    validate_feature_store(store)
    return store
