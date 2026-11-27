import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from midline_agent.datasets.splits import SplitError, load_patient_splits
from midline_agent.diagnostic.attention_mil import AttentionMIL
from midline_agent.diagnostic.checkpoints import (
    AttentionMilPredictor,
    CheckpointError,
    load_attention_mil_checkpoint,
    save_attention_mil_checkpoint,
)
from midline_agent.diagnostic.mil_training import build_optimizer, train_attention_mil, training_step


def _write_split(path: Path, patient_ids: list[str]) -> None:
    path.write_text(json.dumps({"patient_ids": patient_ids}), encoding="utf-8")


def _training_store() -> dict:
    patients = {}
    for index, patient_id in enumerate(("train-a", "train-b", "validation-a")):
        patients[patient_id] = {
            "label": index % 2,
            "slice_ids": [f"{patient_id}:0", f"{patient_id}:1"],
            "modalities": [0, 1],
            "logits": torch.eye(5, dtype=torch.float32)[:2],
            "valid_mask": [True, True],
            "slice_metadata": [{}, {}],
        }
    return {
        "metadata": {
            "schema_version": "1.0",
            "feature_kind": "logits",
            "logit_dimension": 5,
            "hidden_dimension": 0,
            "modality_map": {"T1": 0, "T2": 1},
        },
        "patients": patients,
    }


def _checkpoint_metadata() -> dict:
    return {
        "feature_schema_version": "1.0",
        "feature_kind": "logits",
        "logit_dimension": 5,
        "hidden_dimension": 0,
        "modality_map": {"T1": 0, "T2": 1},
        "train_split_sha256": "a" * 64,
        "validation_split_sha256": "b" * 64,
        "best_epoch": 0,
        "validation_loss": 0.25,
    }


def _saved_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "attention_mil.safetensors"
    model = AttentionMIL(hidden_dimension=8, num_modalities=2)
    save_attention_mil_checkpoint(path, model, _checkpoint_metadata())
    return path


def _mutate_sidecar(path: Path, mutation) -> None:
    sidecar_path = path.with_suffix(".json")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutation(metadata)
    sidecar_path.write_text(json.dumps(metadata, allow_nan=True), encoding="utf-8")


def test_optimizer_contains_only_mil_parameters() -> None:
    model = AttentionMIL(hidden_dimension=8)
    optimizer = build_optimizer(model, learning_rate=1e-3)

    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}


def test_patient_splits_reject_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write_split(train, ["patient-a", "shared"])
    _write_split(validation, ["shared"])

    with pytest.raises(SplitError, match="overlap"):
        load_patient_splits(train, validation, {"patient-a", "shared"})


def test_patient_splits_record_exact_file_hashes(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write_split(train, ["train-a"])
    _write_split(validation, ["validation-a"])

    splits = load_patient_splits(train, validation, {"train-a", "validation-a"})

    assert splits.train_sha256 == hashlib.sha256(train.read_bytes()).hexdigest()
    assert splits.validation_sha256 == hashlib.sha256(validation.read_bytes()).hexdigest()


def test_patient_splits_reject_unknown_patient(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write_split(train, ["unknown"])
    _write_split(validation, ["validation-a"])

    with pytest.raises(SplitError, match="unknown"):
        load_patient_splits(train, validation, {"validation-a"})


def test_training_saves_best_validation_checkpoint_with_split_metadata(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    checkpoint = tmp_path / "attention_mil.safetensors"
    _write_split(train, ["train-a", "train-b"])
    _write_split(validation, ["validation-a"])

    result = train_attention_mil(
        store=_training_store(),
        train_split_path=train,
        validation_split_path=validation,
        checkpoint_path=checkpoint,
        epochs=2,
        learning_rate=1e-3,
        model_hidden_dimension=8,
        seed=7,
    )
    model, metadata = load_attention_mil_checkpoint(checkpoint)

    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".json").is_file()
    assert metadata["train_split_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert metadata["validation_split_sha256"] == hashlib.sha256(validation.read_bytes()).hexdigest()
    assert metadata["best_epoch"] == result.best_epoch
    assert metadata["validation_loss"] == pytest.approx(result.validation_loss)
    assert isinstance(model, AttentionMIL)


def test_checkpoint_predictor_rejects_incompatible_features(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    checkpoint = tmp_path / "attention_mil.safetensors"
    _write_split(train, ["train-a", "train-b"])
    _write_split(validation, ["validation-a"])
    train_attention_mil(
        store=_training_store(),
        train_split_path=train,
        validation_split_path=validation,
        checkpoint_path=checkpoint,
        epochs=1,
        model_hidden_dimension=8,
    )
    predictor = AttentionMilPredictor(checkpoint)
    store = _training_store()
    store["metadata"]["logit_dimension"] = 4

    with pytest.raises(ValueError, match="logit_dimension"):
        predictor.predict_patient(store["patients"]["validation-a"], store["metadata"])


def test_training_cli_requires_explicit_train_and_validation_splits() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_attention_mil", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--train-split" in result.stdout
    assert "--validation-split" in result.stdout
    source = (root / "scripts/train_attention_mil.py").read_text(encoding="utf-8")
    assert "torch.save" not in source
    assert "torch.load" not in source
    assert "sys.path" not in source


def test_checkpoint_loader_wraps_invalid_safetensors_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "attention_mil.safetensors"
    checkpoint.write_bytes(b"not-a-safetensors-file")
    checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointError, match="read"):
        load_attention_mil_checkpoint(checkpoint)


def test_checkpoint_rejects_same_shape_tensor_file_swap(tmp_path: Path) -> None:
    first = _saved_checkpoint(tmp_path / "first")
    torch.manual_seed(19)
    second_path = tmp_path / "second" / "attention_mil.safetensors"
    save_attention_mil_checkpoint(
        second_path,
        AttentionMIL(hidden_dimension=8, num_modalities=2),
        _checkpoint_metadata(),
    )
    first.write_bytes(second_path.read_bytes())

    with pytest.raises(CheckpointError, match="binding"):
        load_attention_mil_checkpoint(first)


def test_checkpoint_save_replaces_final_symlinks_without_following_them(tmp_path: Path) -> None:
    tensor_target = tmp_path / "tensor-target"
    json_target = tmp_path / "json-target"
    tensor_target.write_bytes(b"tensor-sentinel")
    json_target.write_text("json-sentinel", encoding="utf-8")
    checkpoint = tmp_path / "attention_mil.safetensors"
    sidecar = checkpoint.with_suffix(".json")
    checkpoint.symlink_to(tensor_target)
    sidecar.symlink_to(json_target)

    save_attention_mil_checkpoint(
        checkpoint,
        AttentionMIL(hidden_dimension=8, num_modalities=2),
        _checkpoint_metadata(),
    )

    assert tensor_target.read_bytes() == b"tensor-sentinel"
    assert json_target.read_text(encoding="utf-8") == "json-sentinel"
    assert not checkpoint.is_symlink()
    assert not sidecar.is_symlink()
    assert isinstance(load_attention_mil_checkpoint(checkpoint)[0], AttentionMIL)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(feature_schema_version="2.0"), "schema"),
        (lambda metadata: metadata.update(feature_kind="hidden"), "feature_kind"),
        (lambda metadata: metadata.update(modality_map={"T1": 0, "T2": 0}), "modality"),
        (lambda metadata: metadata.update(train_split_sha256="not-a-hash"), "hash"),
        (lambda metadata: metadata.update(best_epoch=-1), "best_epoch"),
        (lambda metadata: metadata.update(validation_loss=float("inf")), "validation_loss"),
        (lambda metadata: metadata.update(checkpoint_path="synthetic-path"), "unexpected"),
        (lambda metadata: metadata["model_config"].update(dropout=1.0), "dropout"),
        (lambda metadata: metadata["model_config"].pop("num_classes"), "model_config"),
        (lambda metadata: metadata["model_config"].update(extra_dimension=4), "model_config"),
    ],
)
def test_checkpoint_rejects_adversarial_metadata(tmp_path: Path, mutation, message: str) -> None:
    checkpoint = _saved_checkpoint(tmp_path)
    _mutate_sidecar(checkpoint, mutation)

    with pytest.raises(CheckpointError, match=message):
        load_attention_mil_checkpoint(checkpoint)


@pytest.mark.parametrize("state_mutation", ["integer", "nonfinite", "unexpected"])
def test_checkpoint_rejects_adversarial_state_tensors(tmp_path: Path, state_mutation: str) -> None:
    checkpoint = _saved_checkpoint(tmp_path)
    state = load_file(checkpoint)
    tensor_name = next(iter(state))
    if state_mutation == "integer":
        state[tensor_name] = state[tensor_name].to(torch.int64)
    elif state_mutation == "nonfinite":
        state[tensor_name] = state[tensor_name].clone()
        state[tensor_name].flatten()[0] = float("nan")
    else:
        state["unexpected.weight"] = torch.ones(1)
    sidecar_path = checkpoint.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    save_file(
        state,
        checkpoint,
        metadata={
            "artifact_id": sidecar["artifact_id"],
            "format": sidecar["format"],
            "format_version": sidecar["format_version"],
        },
    )
    sidecar["tensor_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(CheckpointError, match="tensor"):
        load_attention_mil_checkpoint(checkpoint)


def test_training_rejects_label_outside_configured_class_range(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write_split(train, ["train-a", "train-b"])
    _write_split(validation, ["validation-a"])

    with pytest.raises(ValueError, match="label"):
        train_attention_mil(
            store=_training_store(),
            train_split_path=train,
            validation_split_path=validation,
            checkpoint_path=tmp_path / "attention_mil.safetensors",
            epochs=1,
            num_classes=1,
        )


def test_training_step_rejects_nonfinite_loss(monkeypatch) -> None:
    model = AttentionMIL(hidden_dimension=8)
    optimizer = build_optimizer(model, 1e-3)
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True),
    )

    with pytest.raises(FloatingPointError, match="training loss"):
        training_step(
            model,
            torch.ones(1, 1, 5),
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            torch.tensor([0]),
            optimizer,
        )


def test_training_rejects_nonfinite_validation_loss(tmp_path: Path, monkeypatch) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    _write_split(train, ["train-a", "train-b"])
    _write_split(validation, ["validation-a"])
    call_count = 0

    def controlled_loss(logits, _label):
        nonlocal call_count
        call_count += 1
        value = float("nan") if call_count == 3 else 1.0
        return logits.sum() * 0 + value

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", controlled_loss)

    with pytest.raises(FloatingPointError, match="validation loss"):
        train_attention_mil(
            store=_training_store(),
            train_split_path=train,
            validation_split_path=validation,
            checkpoint_path=tmp_path / "attention_mil.safetensors",
            epochs=1,
            model_hidden_dimension=8,
        )
