from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from midline_agent.datasets.inference_annotations import InferenceSlice
from midline_agent.diagnostic import brainvlm_runtime
from midline_agent.diagnostic.brainvlm_runtime import (
    BrainVlmRuntime,
    BrainVlmRuntimeError,
    SliceFeatures,
    _validate_device,
)


class FakeInternalModel(nn.Module):
    def __init__(self, generation: object = "<tumor_4> <grade_3>") -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.generation = generation
        self.feature_calls: list[tuple[torch.Tensor, str, tuple[str, ...]]] = []
        self.generation_calls: list[dict[str, object]] = []

    def maybe_autocast(self):
        return nullcontext()

    def extract_slice_features(self, image, prompt, class_tokens):
        self.feature_calls.append((image.detach().cpu(), prompt, tuple(class_tokens)))
        return torch.arange(5.0).unsqueeze(0), torch.arange(4096.0).unsqueeze(0)

    def generate_step(self, image, prompt, high_resolution, **kwargs):
        self.generation_calls.append(
            {
                "image": image.detach().cpu(),
                "prompt": prompt,
                "high_resolution": high_resolution,
                **kwargs,
            }
        )
        return self.generation


def _write_rgb(path: Path) -> None:
    Image.new("RGB", (32, 16), color=(0, 64, 128)).save(path)


def _metadata(path: Path, age: float = 53.0) -> InferenceSlice:
    return InferenceSlice("BT0156", path, "T1", 4, age)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:-1", "cuda:abc", "cuda:2:1"])
def test_device_validation_requires_indexed_cuda(device: str) -> None:
    with pytest.raises(BrainVlmRuntimeError, match="cuda"):
        _validate_device(device)


def test_device_validation_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(BrainVlmRuntimeError, match="not available"):
        _validate_device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(BrainVlmRuntimeError, match="index"):
        _validate_device("cuda:1")


def test_runtime_load_constructs_only_the_internal_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama = tmp_path / "llama"
    llama.mkdir()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    biomedclip = tmp_path / "biomedclip.bin"
    biomedclip.write_bytes(b"weights")
    model = FakeInternalModel()
    observed: dict[str, object] = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: observed.update(device=device))
    monkeypatch.setattr(
        brainvlm_runtime,
        "_construct_internal_model",
        lambda assets: observed.update(assets=assets) or model,
    )
    monkeypatch.setattr(
        brainvlm_runtime,
        "load_brainvlm_checkpoint",
        lambda path, loaded_model: observed.update(checkpoint=path, model=loaded_model),
    )
    monkeypatch.setattr(model, "to", lambda device: observed.update(model_device=device) or model)

    runtime = BrainVlmRuntime.load(
        llama,
        checkpoint,
        device="cuda:0",
        biomedclip_pretrained=str(biomedclip),
    )

    assert runtime.model is model
    assert observed["model"] is model
    assert observed["checkpoint"] == checkpoint.resolve()
    assert observed["assets"].llama_model == llama.resolve()
    assert observed["assets"].biomedclip_pretrained == str(biomedclip.resolve())


def test_runtime_load_rejects_missing_foundation_assets(tmp_path: Path) -> None:
    with pytest.raises(BrainVlmRuntimeError, match="llama"):
        BrainVlmRuntime.load(tmp_path / "missing", tmp_path / "missing.pth")


def test_slice_features_are_from_one_forward_and_tumor_reuses_them(tmp_path: Path) -> None:
    image = tmp_path / "slice.png"
    _write_rgb(image)
    model = FakeInternalModel()
    runtime = BrainVlmRuntime.from_model(model)

    features = runtime.predict_slice_features(image, _metadata(image))
    logits = runtime.predict_tumor_logits(image, _metadata(image))

    assert isinstance(features, SliceFeatures)
    assert features.logits.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert features.hidden.shape == (4096,)
    assert logits.tolist() == features.logits.tolist()
    assert len(model.feature_calls) == 2
    assert model.feature_calls[0][2] == ("1", "2", "3", "4", "5")
    assert model.feature_calls[0][0].shape == (1, 3, 224, 224)


def test_slice_features_reject_incompatible_outputs(tmp_path: Path) -> None:
    image = tmp_path / "slice.png"
    _write_rgb(image)
    model = FakeInternalModel()
    model.extract_slice_features = lambda *args: (torch.ones(1, 4), torch.ones(1, 8))  # type: ignore[method-assign]

    with pytest.raises(BrainVlmRuntimeError, match="invalid"):
        BrainVlmRuntime.from_model(model).predict_slice_features(image, _metadata(image))


def test_who_generation_uses_historical_adult_prompt_and_greedy_arguments(tmp_path: Path) -> None:
    image = tmp_path / "slice.png"
    _write_rgb(image)
    model = FakeInternalModel("  $<tumor_4> <grade_3>$  ")
    runtime = BrainVlmRuntime.from_model(model)

    output = runtime.predict_who_generation(image, _metadata(image), "tumor_4", 53.0)

    assert output == "<tumor_4> <grade_3>"
    call = model.generation_calls[0]
    assert "This is a adult patient." in call["prompt"]
    assert call["max_new_tokens"] == 50
    assert call["temperature"] == 0.0
    assert call["do_sample"] is False
    assert call["high_resolution"] is None


def test_runtime_rejects_bad_image_and_bad_generation(tmp_path: Path) -> None:
    image = tmp_path / "slice.png"
    image.write_text("not an image", encoding="ascii")
    runtime = BrainVlmRuntime.from_model(FakeInternalModel())
    with pytest.raises(BrainVlmRuntimeError, match="decode"):
        runtime.predict_tumor_logits(image, _metadata(image))

    _write_rgb(image)
    runtime = BrainVlmRuntime.from_model(FakeInternalModel(""))
    with pytest.raises(BrainVlmRuntimeError, match="text"):
        runtime.predict_who_generation(image, _metadata(image), "tumor_4", 53.0)
