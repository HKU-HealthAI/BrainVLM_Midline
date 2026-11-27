import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from midline_agent.datasets.annotations import AnnotationError
from midline_agent.datasets.brainvlm import PREPROCESSING_VERSION
from midline_agent.datasets.prompts import PROMPT_VERSION
from midline_agent.diagnostic.brainvlm_runtime import SliceFeatures
from midline_agent.diagnostic.feature_store import load_feature_store
from scripts import extract_features


EXPECTED_FLAGS = {
    "--annotations",
    "--image-root",
    "--llama-model",
    "--biomedclip-pretrained",
    "--checkpoint",
    "--device",
    "--output",
    "--labeled",
}


def _write_image(path: Path, value: int) -> None:
    Image.new("RGB", (3, 2), color=(value, value, value)).save(path)


def _write_annotations(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _arguments(tmp_path: Path, annotations: Path, *, labeled: bool = False) -> list[str]:
    arguments = [
        "extract_features.py",
        "--annotations",
        str(annotations),
        "--image-root",
        str(tmp_path / "images"),
        "--llama-model",
        str(tmp_path / "llama"),
        "--biomedclip-pretrained",
        str(tmp_path / "biomedclip.bin"),
        "--checkpoint",
        str(tmp_path / "brainvlm.pth"),
        "--device",
        "cuda:7",
        "--output",
        str(tmp_path / "features.safetensors"),
    ]
    if labeled:
        arguments.append("--labeled")
    return arguments


class _FakeRuntime:
    load_arguments: tuple[object, ...] | None = None
    calls: list[tuple[Path, object]] = []

    @classmethod
    def load(
        cls,
        llama_model: Path,
        checkpoint: Path,
        *,
        device: str,
        biomedclip_pretrained: str | None,
    ) -> "_FakeRuntime":
        cls.load_arguments = (llama_model, checkpoint, device, biomedclip_pretrained)
        cls.calls = []
        return cls()

    def predict_slice_features(self, image_path: Path, metadata: object) -> SliceFeatures:
        self.calls.append((image_path, metadata))
        value = float(getattr(metadata, "slice_index"))
        return SliceFeatures(
            logits=torch.full((5,), value, dtype=torch.float32),
            hidden=torch.full((4096,), value, dtype=torch.float32),
        )


def test_feature_extraction_cli_exposes_only_internal_runtime_arguments() -> None:
    parser = extract_features._parser()
    flags = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.extract_features", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert flags == EXPECTED_FLAGS
    assert "--source-root" not in completed.stdout
    assert "--adapter" not in completed.stdout
    source = (root / "scripts/extract_features.py").read_text(encoding="utf-8")
    assert "sys.path" not in source
    assert "load_brainvlm_adapter" not in source


def test_label_free_extraction_is_ordered_reproducible_and_omits_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "t2.png", 22)
    _write_image(images / "t1-late.png", 12)
    _write_image(images / "t1-early.png", 11)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {"patient_id": "case-a", "image_path": "t2.png", "modality": "T2", "slice_index": 2},
            {"patient_id": "case-a", "image_path": "t1-late.png", "modality": "T1", "slice_index": 3},
            {"patient_id": "case-a", "image_path": "t1-early.png", "modality": "T1", "slice_index": 1},
        ],
    )
    (tmp_path / "llama").mkdir()
    (tmp_path / "biomedclip.bin").write_bytes(b"vision")
    checkpoint = tmp_path / "brainvlm.pth"
    checkpoint.write_bytes(b"checkpoint bytes")
    monkeypatch.setattr(extract_features, "BrainVlmRuntime", _FakeRuntime)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, annotations))

    assert extract_features.main() == 0

    store = load_feature_store(tmp_path / "features.safetensors")
    metadata = store["metadata"]
    patient = store["patients"]["case-a"]
    assert metadata == {
        "schema_version": "1.0",
        "feature_kind": "hidden_and_logits",
        "logit_dimension": 5,
        "hidden_dimension": 4096,
        "modality_map": {"T1": 0, "T1CE": 1, "T2": 2, "FLAIR": 3},
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    assert "label" not in patient
    assert patient["slice_ids"] == ["case-a:T1:1", "case-a:T1:3", "case-a:T2:2"]
    assert patient["modalities"] == [0, 0, 2]
    assert patient["valid_mask"] == [True, True, True]
    assert patient["logits"].shape == (3, 5)
    assert patient["hidden"].shape == (3, 4096)
    assert [row["slice_index"] for row in patient["slice_metadata"]] == [1, 3, 2]
    assert [row["checksum"] for row in patient["slice_metadata"]] == [
        hashlib.sha256((images / name).read_bytes()).hexdigest()
        for name in ("t1-early.png", "t1-late.png", "t2.png")
    ]
    assert [getattr(call[1], "slice_index") for call in _FakeRuntime.calls] == [1, 3, 2]
    assert _FakeRuntime.load_arguments == (
        tmp_path / "llama",
        checkpoint,
        "cuda:7",
        str(tmp_path / "biomedclip.bin"),
    )
    sidecar_text = (tmp_path / "features.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in sidecar_text
    assert "generation" not in sidecar_text


def test_label_free_extraction_rejects_diagnostic_labels_before_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "slice.png", 20)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            }
        ],
    )
    monkeypatch.setattr(extract_features, "BrainVlmRuntime", _FakeRuntime)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, annotations))
    _FakeRuntime.load_arguments = None

    with pytest.raises(AnnotationError, match="unexpected"):
        extract_features.main()

    assert _FakeRuntime.load_arguments is None


def test_labeled_extraction_stores_one_consistent_zero_based_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "t2.png", 22)
    _write_image(images / "t1.png", 11)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "patient_id": "case-a",
                "image_path": "t2.png",
                "modality": "T2",
                "slice_index": 2,
                "tumor_type": "tumor_3",
            },
            {
                "patient_id": "case-a",
                "image_path": "t1.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_3",
            },
        ],
    )
    (tmp_path / "llama").mkdir()
    (tmp_path / "biomedclip.bin").write_bytes(b"vision")
    (tmp_path / "brainvlm.pth").write_bytes(b"checkpoint")
    monkeypatch.setattr(extract_features, "BrainVlmRuntime", _FakeRuntime)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, annotations, labeled=True))

    assert extract_features.main() == 0

    patient = load_feature_store(tmp_path / "features.safetensors")["patients"]["case-a"]
    assert patient["label"] == 2
    assert patient["slice_ids"] == ["case-a:T1:1", "case-a:T2:2"]


@pytest.mark.parametrize(
    "records",
    [
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_9",
            }
        ],
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": "not-an-index",
                "tumor_type": "tumor_1",
            }
        ],
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            },
            {
                "patient_id": "case-a",
                "image_path": "slice-2.png",
                "modality": "T2",
                "slice_index": 2,
                "tumor_type": "tumor_2",
            },
        ],
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            },
            {
                "patient_id": "case-b",
                "image_path": "slice-2.png",
                "modality": "T2",
                "slice_index": 2,
                "tumor_type": "tumor_1",
            },
        ],
        [
            {
                "patient_id": "case-a",
                "image_path": "slice.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            },
            {
                "patient_id": "case-a",
                "image_path": "slice-2.png",
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            },
        ],
    ],
)
def test_labeled_extraction_rejects_invalid_patient_label_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: list[dict[str, object]]
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "slice.png", 20)
    _write_image(images / "slice-2.png", 21)
    annotations = tmp_path / "annotations.json"
    _write_annotations(annotations, records)
    monkeypatch.setattr(extract_features, "BrainVlmRuntime", _FakeRuntime)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, annotations, labeled=True))
    _FakeRuntime.load_arguments = None

    with pytest.raises(AnnotationError):
        extract_features.main()

    assert _FakeRuntime.load_arguments is None


def test_labeled_extraction_rejects_absolute_image_paths_even_within_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "slice.png"
    _write_image(image, 20)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "patient_id": "case-a",
                "image_path": str(image),
                "modality": "T1",
                "slice_index": 1,
                "tumor_type": "tumor_1",
            }
        ],
    )
    monkeypatch.setattr(extract_features, "BrainVlmRuntime", _FakeRuntime)
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, annotations, labeled=True))

    with pytest.raises(AnnotationError, match="relative"):
        extract_features.main()
