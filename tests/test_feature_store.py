import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from midline_agent.diagnostic.feature_store import FeatureSchemaError, load_feature_store, save_feature_store


def _store(*, feature_kind: str = "logits") -> dict:
    metadata = {
        "schema_version": "1.0",
        "feature_kind": feature_kind,
        "logit_dimension": 5 if "logits" in feature_kind else 0,
        "hidden_dimension": 3 if "hidden" in feature_kind else 0,
        "modality_map": {"T1": 0, "T2": 1},
    }
    patient = {
        "label": 0,
        "slice_ids": ["s1", "s2"],
        "modalities": [0, 1],
        "valid_mask": [True, True],
        "slice_metadata": [{"modality": "T1"}, {"modality": "T2"}],
    }
    if metadata["logit_dimension"]:
        patient["logits"] = torch.zeros(2, 5, dtype=torch.float32)
    if metadata["hidden_dimension"]:
        patient["hidden"] = torch.ones(2, 3, dtype=torch.float32)
    return {"metadata": metadata, "patients": {"patient-a": patient}}


def test_feature_store_round_trip_uses_safetensors_and_json(tmp_path: Path) -> None:
    path = tmp_path / "features.safetensors"

    save_feature_store(path, _store(feature_kind="hidden_and_logits"))
    loaded = load_feature_store(path)

    assert path.is_file()
    assert path.with_suffix(".json").is_file()
    assert set(load_file(path)) == {"patient.0.hidden", "patient.0.logits"}
    assert loaded["patients"]["patient-a"]["logits"].shape == (2, 5)
    assert loaded["patients"]["patient-a"]["hidden"].shape == (2, 3)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["format"] == "midline-agent-feature-store"
    assert sidecar["tensor_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with safe_open(path, framework="pt", device="cpu") as artifact:
        assert artifact.metadata()["artifact_id"] == sidecar["artifact_id"]
    assert "logits" not in sidecar["patients"]["patient-a"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda store: store["patients"]["patient-a"].update(logits=torch.zeros(2, 5, dtype=torch.float64)), "float32"),
        (lambda store: store["patients"]["patient-a"].update(logits=torch.zeros(2, 1, 5)), "rank-2"),
        (lambda store: store["patients"]["patient-a"].update(logits=torch.full((2, 5), float("nan"))), "finite"),
        (lambda store: store["patients"]["patient-a"].update(modalities=[0, 9]), "modality"),
        (lambda store: store["patients"]["patient-a"].update(valid_mask=[1, 1]), "boolean"),
        (lambda store: store["patients"]["patient-a"].update(valid_mask=[False, False]), "valid slice"),
        (
            lambda store: store["patients"]["patient-a"].update(
                slice_ids=[], modalities=[], logits=torch.empty(0, 5), valid_mask=[], slice_metadata=[]
            ),
            "non-empty",
        ),
    ],
)
def test_feature_store_rejects_invalid_bags(tmp_path: Path, mutation, message: str) -> None:
    store = _store()
    mutation(store)

    with pytest.raises(FeatureSchemaError, match=message):
        save_feature_store(tmp_path / "bad.safetensors", store)


def test_feature_store_rejects_inconsistent_slice_lengths(tmp_path: Path) -> None:
    store = _store()
    store["patients"]["patient-a"]["modalities"] = [0]

    with pytest.raises(FeatureSchemaError, match="aligned"):
        save_feature_store(tmp_path / "bad.safetensors", store)


@pytest.mark.parametrize("label", [-100, 5, True, 1.0, None])
def test_feature_store_rejects_invalid_optional_training_label(tmp_path: Path, label) -> None:
    store = _store()
    store["patients"]["patient-a"]["label"] = label

    with pytest.raises(FeatureSchemaError, match="label"):
        save_feature_store(tmp_path / "bad.safetensors", store)


def test_feature_store_requires_safetensors_extension(tmp_path: Path) -> None:
    with pytest.raises(FeatureSchemaError, match="safetensors"):
        save_feature_store(tmp_path / "features.pth", _store())


def test_feature_store_rejects_missing_tensor_on_load(tmp_path: Path) -> None:
    path = tmp_path / "features.safetensors"
    save_feature_store(path, _store())
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["patients"]["patient-a"]["tensor_keys"]["logits"] = "missing"
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(FeatureSchemaError, match="tensor"):
        load_feature_store(path)


def test_feature_store_rejects_tensor_not_declared_by_feature_kind(tmp_path: Path) -> None:
    store = _store(feature_kind="hidden")
    store["patients"]["patient-a"]["logits"] = torch.zeros(2, 5)

    with pytest.raises(FeatureSchemaError, match="feature_kind"):
        save_feature_store(tmp_path / "bad.safetensors", store)


def test_feature_store_wraps_invalid_safetensors_file(tmp_path: Path) -> None:
    path = tmp_path / "features.safetensors"
    save_feature_store(path, _store())
    path.write_bytes(b"not-a-safetensors-file")

    with pytest.raises(FeatureSchemaError, match="read"):
        load_feature_store(path)


def test_feature_store_rejects_same_shape_tensor_file_swap(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_feature_store(first, _store())
    other = _store()
    other["patients"]["patient-a"]["logits"].fill_(7.0)
    save_feature_store(second, other)
    first.write_bytes(second.read_bytes())

    with pytest.raises(FeatureSchemaError, match="binding"):
        load_feature_store(first)


def test_feature_store_rejects_sidecar_artifact_id_tampering(tmp_path: Path) -> None:
    path = tmp_path / "features.safetensors"
    save_feature_store(path, _store())
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["artifact_id"] = "0" * 32
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(FeatureSchemaError, match="binding"):
        load_feature_store(path)


def test_feature_store_save_replaces_final_symlinks_without_following_them(tmp_path: Path) -> None:
    tensor_target = tmp_path / "tensor-target"
    json_target = tmp_path / "json-target"
    tensor_target.write_bytes(b"tensor-sentinel")
    json_target.write_text("json-sentinel", encoding="utf-8")
    path = tmp_path / "features.safetensors"
    sidecar_path = path.with_suffix(".json")
    path.symlink_to(tensor_target)
    sidecar_path.symlink_to(json_target)

    save_feature_store(path, _store())

    assert tensor_target.read_bytes() == b"tensor-sentinel"
    assert json_target.read_text(encoding="utf-8") == "json-sentinel"
    assert not path.is_symlink()
    assert not sidecar_path.is_symlink()
    assert load_feature_store(path)["patients"]["patient-a"]["logits"].shape == (2, 5)
