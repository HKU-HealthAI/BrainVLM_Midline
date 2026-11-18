"""Atomic, cryptographically bound safetensors and JSON artifact pairs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ARTIFACT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ArtifactBindingError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bound_safetensors(
    tensor_path: Path,
    sidecar_path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    artifact_format: str,
    format_version: str,
    build_sidecar: Callable[[str, str], Mapping[str, Any]],
) -> None:
    tensor_path = Path(tensor_path)
    sidecar_path = Path(sidecar_path)
    if tensor_path.parent != sidecar_path.parent:
        raise ArtifactBindingError("artifact components must share a directory")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_id = secrets.token_hex(16)
    tensor_fd, tensor_temp_name = tempfile.mkstemp(
        dir=tensor_path.parent,
        prefix=f".{tensor_path.name}.",
        suffix=".tmp",
    )
    sidecar_fd, sidecar_temp_name = tempfile.mkstemp(
        dir=sidecar_path.parent,
        prefix=f".{sidecar_path.name}.",
        suffix=".tmp",
    )
    os.close(tensor_fd)
    os.close(sidecar_fd)
    tensor_temp = Path(tensor_temp_name)
    sidecar_temp = Path(sidecar_temp_name)
    try:
        save_file(
            dict(tensors),
            str(tensor_temp),
            metadata={
                "artifact_id": artifact_id,
                "format": artifact_format,
                "format_version": format_version,
            },
        )
        tensor_sha256 = sha256_file(tensor_temp)
        sidecar = build_sidecar(artifact_id, tensor_sha256)
        encoded = json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n"
        sidecar_temp.write_text(encoded, encoding="utf-8")
        os.replace(tensor_temp, tensor_path)
        os.replace(sidecar_temp, sidecar_path)
    finally:
        tensor_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)


def load_bound_safetensors(
    tensor_path: Path,
    sidecar: Mapping[str, Any],
    *,
    artifact_format: str,
    format_version: str,
) -> dict[str, torch.Tensor]:
    artifact_id = sidecar.get("artifact_id")
    tensor_sha256 = sidecar.get("tensor_sha256")
    if not isinstance(artifact_id, str) or ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
        raise ArtifactBindingError("artifact binding has an invalid artifact ID")
    if not isinstance(tensor_sha256, str) or SHA256_PATTERN.fullmatch(tensor_sha256) is None:
        raise ArtifactBindingError("artifact binding has an invalid tensor hash")
    actual_sha256 = sha256_file(tensor_path)
    if not secrets.compare_digest(actual_sha256, tensor_sha256):
        raise ArtifactBindingError("artifact binding tensor hash mismatch")
    with safe_open(tensor_path, framework="pt", device="cpu") as artifact:
        tensor_metadata = artifact.metadata() or {}
    expected_metadata = {
        "artifact_id": artifact_id,
        "format": artifact_format,
        "format_version": format_version,
    }
    if tensor_metadata != expected_metadata:
        raise ArtifactBindingError("artifact binding metadata mismatch")
    return load_file(tensor_path, device="cpu")
