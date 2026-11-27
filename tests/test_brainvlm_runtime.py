from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from midline_agent.datasets.inference_annotations import InferenceSlice
from midline_agent.diagnostic.brainvlm_runtime import (
    BIOMEDCLIP_MEAN,
    BIOMEDCLIP_STD,
    BrainVlmRuntime,
    BrainVlmRuntimeError,
    _validate_device,
    _validate_external_source,
    _validate_module_origins,
    load_brainvlm_checkpoint,
)


class TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)


def _save_model_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"model": state, "epoch": 3, "config": {"private": "ignored"}}, path)


def test_external_source_validation_rejects_missing_or_escaped_package(tmp_path: Path) -> None:
    with pytest.raises(BrainVlmRuntimeError, match="source root"):
        _validate_external_source(tmp_path / "missing")

    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    (outside / "models").mkdir(parents=True)
    (outside / "__init__.py").write_text("", encoding="ascii")
    (outside / "models" / "minigpt4.py").write_text("", encoding="ascii")
    (source / "minigpt4").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BrainVlmRuntimeError, match="inside source root"):
        _validate_external_source(source)


def test_module_origin_validation_rejects_another_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "minigpt4"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("", encoding="ascii")
    outside_file = tmp_path / "elsewhere.py"
    outside_file.write_text("", encoding="ascii")

    with pytest.raises(BrainVlmRuntimeError, match="outside source root"):
        _validate_module_origins(source, {"minigpt4": SimpleNamespace(__file__=outside_file)})

    _validate_module_origins(source, {"minigpt4": SimpleNamespace(__file__=package_file)})


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


def test_checkpoint_loader_uses_weights_only_and_ignores_training_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyCheckpointModel()
    checkpoint = tmp_path / "model.pth"
    state = {name: torch.full_like(value, 2.0) for name, value in model.state_dict().items()}
    _save_model_checkpoint(checkpoint, state)
    original_load = torch.load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    loaded_names = load_brainvlm_checkpoint(model, checkpoint)

    assert loaded_names == tuple(sorted(state))
    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"
    assert all(torch.equal(value, torch.full_like(value, 2.0)) for value in model.state_dict().values())


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"unknown": torch.ones(1)}, "unrecognized"),
        ({"projection.weight": torch.ones(3, 3)}, "shape"),
        ({"projection.weight": torch.full((2, 2), float("nan"))}, "finite"),
        ({}, "empty"),
    ],
)
def test_checkpoint_loader_rejects_incompatible_state(
    tmp_path: Path,
    state: dict[str, torch.Tensor],
    message: str,
) -> None:
    checkpoint = tmp_path / "model.pth"
    _save_model_checkpoint(checkpoint, state)

    with pytest.raises(BrainVlmRuntimeError, match=message):
        load_brainvlm_checkpoint(TinyCheckpointModel(), checkpoint)


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in {"1", "2", "3", "4", "5"}:
            return [10 + int(text)]
        return [2]


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.lm_head = nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            for position, token_id in enumerate(range(11, 16)):
                self.lm_head.weight[token_id, position % 4] = float(position + 1)

    def forward(self, **kwargs):
        batch, sequence, _ = kwargs["inputs_embeds"].shape
        hidden = torch.ones(batch, sequence, 4, device=kwargs["inputs_embeds"].device)
        return SimpleNamespace(hidden_states=(hidden,))


class FakeBrainVlmModel(nn.Module):
    def __init__(self, generation: object = "<tumor_4> <grade_3>") -> None:
        super().__init__()
        self.llama_tokenizer = FakeTokenizer()
        self.llama_model = FakeLanguageModel()
        self.generation = generation
        self.seen_images: list[torch.Tensor] = []
        self.seen_prompts: list[str] = []
        self.generation_calls: list[dict[str, object]] = []

    def encode_img(self, image: torch.Tensor, resolution_type: str):
        assert resolution_type == "LR_Encoding"
        self.seen_images.append(image.detach().cpu())
        return torch.ones(1, 1, 4, device=image.device), torch.ones(1, 1, dtype=torch.long, device=image.device), None

    def llama_proj_mlps(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def prompt_wrap(self, image: torch.Tensor, attention: torch.Tensor, prompts: list[str]):
        del image
        self.seen_prompts.extend(prompts)
        return (
            torch.ones(1, 2, 4, device=attention.device),
            torch.ones(1, 2, dtype=torch.long, device=attention.device),
            torch.ones(1, 2, dtype=torch.long, device=attention.device),
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(*token_ids.shape, 4, device=token_ids.device)

    def maybe_autocast(self):
        return torch.autocast(device_type="cpu", enabled=False)

    def generate_step(self, images: torch.Tensor, prompts: str, hr_images, **kwargs):
        self.seen_images.append(images.detach().cpu())
        self.seen_prompts.append(prompts)
        self.generation_calls.append({"hr_images": hr_images, **kwargs})
        return self.generation


def _write_rgb_image(path: Path) -> None:
    Image.new("RGB", (224, 224), color=(0, 64, 128)).save(path)


def _metadata(path: Path) -> InferenceSlice:
    return InferenceSlice("patient-a", path, "T1", 2, 40.0)


def test_tumor_inference_normalizes_image_and_returns_five_finite_logits(tmp_path: Path) -> None:
    image_path = tmp_path / "slice.png"
    _write_rgb_image(image_path)
    model = FakeBrainVlmModel()
    runtime = BrainVlmRuntime.from_model(model, device="cpu")

    logits = runtime.predict_tumor_logits(image_path, _metadata(image_path))

    assert logits.shape == (5,)
    assert bool(torch.isfinite(logits).all())
    assert logits.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    observed = model.seen_images[0][0]
    assert observed.shape == (3, 224, 224)
    expected_corner = (torch.tensor([0.0, 64 / 255, 128 / 255]) - torch.tensor(BIOMEDCLIP_MEAN)) / torch.tensor(BIOMEDCLIP_STD)
    torch.testing.assert_close(observed[:, 0, 0], expected_corner)
    assert model.seen_prompts[0].endswith("ASSISTANT:<tumor_")


def test_who_generation_is_greedy_and_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "slice.png"
    _write_rgb_image(image_path)
    model = FakeBrainVlmModel()
    runtime = BrainVlmRuntime.from_model(model, device="cpu")

    output = runtime.predict_who_generation(image_path, _metadata(image_path), "tumor_4", 40.0)

    assert output == "<tumor_4> <grade_3>"
    assert "adult patient" in model.seen_prompts[0]
    assert model.generation_calls == [
        {"hr_images": None, "max_new_tokens": 50, "temperature": 0.0, "do_sample": False}
    ]


@pytest.mark.parametrize("generation", [None, "", [], ["<tumor_4> <grade_3>"]])
def test_who_generation_rejects_empty_or_malformed_output(tmp_path: Path, generation: object) -> None:
    image_path = tmp_path / "slice.png"
    _write_rgb_image(image_path)
    runtime = BrainVlmRuntime.from_model(FakeBrainVlmModel(generation), device="cpu")

    with pytest.raises(BrainVlmRuntimeError, match="non-empty string"):
        runtime.predict_who_generation(image_path, _metadata(image_path), "tumor_4", None)
