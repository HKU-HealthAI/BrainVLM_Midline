from types import SimpleNamespace

import pytest
import torch
from torch import nn

from midline_agent.diagnostic.brainvlm_midline.config import ModelSettings
from midline_agent.diagnostic.brainvlm_midline.llama import (
    build_supervised_batch,
    causal_lm_loss,
    create_llama,
)
from midline_agent.diagnostic.brainvlm_midline.model import (
    BrainVlmMidline,
    ResolutionAttention,
)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    bos_token = "<bos>"

    def __init__(self) -> None:
        self.pad_token = None
        self.padding_side = "left"
        self.added: list[dict[str, list[str]]] = []

    def __len__(self) -> int:
        return 17

    def add_special_tokens(self, value: dict[str, list[str]]) -> None:
        self.added.append(value)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "###":
            return [9]
        if text in {"1", "2", "3", "4", "5"}:
            return [10 + int(text)]
        return [3 + (ord(char) % 10) for char in text]

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return " ".join(str(item) for item in token_ids)

    def __call__(
        self,
        text,
        *,
        return_tensors="pt",
        padding=None,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
    ):
        del return_tensors, padding
        values = [text] if isinstance(text, str) else list(text)
        encoded = []
        for item in values:
            tokens = [3 + (ord(char) % 10) for char in item]
            if add_special_tokens:
                tokens.insert(0, self.bos_token_id)
            if truncation and max_length is not None:
                tokens = tokens[:max_length]
            encoded.append(tokens)
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class TinyLanguageModel(nn.Module):
    def __init__(self, width: int = 8, vocab: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.resized_to = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.resized_to = new_num_tokens


class RecordingLanguageModel(TinyLanguageModel):
    def __init__(self) -> None:
        super().__init__(width=8, vocab=32)
        self.calls: list[dict[str, object]] = []
        self.generated = [7, 8, 9]

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["inputs_embeds"]
        batch, length, _ = inputs.shape
        hidden = torch.arange(
            batch * length * 8,
            dtype=inputs.dtype,
            device=inputs.device,
        ).reshape(batch, length, 8)
        logits = self.lm_head(hidden)
        if kwargs.get("use_cache"):
            logits = torch.full((batch, length, 32), -100.0, device=inputs.device)
            logits[:, -1, self.generated.pop(0)] = 100.0
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=(len(self.calls),),
        )


def test_create_llama_preserves_reference_configuration_and_token_order() -> None:
    tokenizer = TinyTokenizer()
    language_model = TinyLanguageModel()
    calls: dict[str, object] = {}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"] = (path, kwargs)
            return tokenizer

    class ModelLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"] = (path, kwargs)
            return language_model

    def prepare(model):
        calls["prepared"] = model
        return model

    def config_factory(**kwargs):
        calls["lora_config"] = kwargs
        return kwargs

    def wrap(model, config):
        calls["wrapped"] = (model, config)
        return model

    result_model, result_tokenizer = create_llama(
        "/models/llama",
        tokenizer_loader=TokenizerLoader,
        model_loader=ModelLoader,
        prepare_model=prepare,
        lora_config_factory=config_factory,
        peft_wrapper=wrap,
    )

    assert result_model is language_model
    assert result_tokenizer is tokenizer
    assert tokenizer.pad_token == "$$"
    assert tokenizer.added == [
        {"additional_special_tokens": ["<box>", "<Img>", "</Img>", "<t>"]}
    ]
    assert language_model.resized_to == len(tokenizer)
    assert calls["lora_config"] == {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],
    }
    assert calls["model"][1]["torch_dtype"] == torch.float16


def test_supervised_batch_masks_bos_prompt_image_and_padding() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLanguageModel()
    image_embeddings = torch.ones(2, 1, 8)

    batch = build_supervised_batch(
        tokenizer,
        model.get_input_embeddings(),
        image_embeddings,
        ["A<ImageHere>B", "Long<ImageHere>C"],
        ["x", "yz"],
        end_symbol="###",
        max_text_length=3,
        max_context_length=12,
    )

    assert batch.inputs_embeds.shape[:2] == batch.attention_mask.shape
    assert batch.labels.shape == batch.attention_mask.shape
    assert torch.all(batch.labels[:, 0] == -100)
    for row, answer_start in enumerate(batch.answer_starts):
        assert torch.all(batch.labels[row, :answer_start] == -100)
        answer_labels = batch.labels[row, answer_start:]
        assert bool((answer_labels != -100).any())
    assert batch.labels.shape[1] <= 1 + 12 + 3


def test_causal_lm_loss_supports_mean_and_historical_per_sample_reduction() -> None:
    logits = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]], [[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]]
    )
    labels = torch.tensor([[-100, 1, 0], [-100, 0, -100]])

    mean = causal_lm_loss(logits, labels, reduction="mean")
    per_sample = causal_lm_loss(logits, labels, reduction="none")

    token_losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 2),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(2, -1)
    torch.testing.assert_close(per_sample, token_losses.mean(dim=1))
    assert mean.ndim == 0
    with pytest.raises(ValueError, match="reduction"):
        causal_lm_loss(logits, labels, reduction="sum")


def test_resolution_attention_preserves_checkpoint_names_and_shape() -> None:
    module = ResolutionAttention(width=4, dropout=0.0)
    output = module(torch.ones(1, 8, 4), torch.ones(1, 8, 4))

    assert output.shape == (1, 8, 4)
    assert set(module.state_dict()) == {
        "Wq.weight",
        "Wq.bias",
        "Wk.weight",
        "Wk.bias",
        "Wv.weight",
        "Wv.bias",
        "output_linear.weight",
        "output_linear.bias",
        "norm.weight",
        "norm.bias",
    }


def test_model_assembly_preserves_project_specific_state_names() -> None:
    settings = ModelSettings(
        vision_width=4,
        llama_width=8,
        compatibility_width=6,
        llama_layers=1,
        key_value_width=2,
    )
    visual = nn.Sequential()
    visual.norm = nn.LayerNorm(4)  # type: ignore[attr-defined]
    model = BrainVlmMidline(
        TinyLanguageModel(width=8),
        TinyTokenizer(),
        visual,
        settings=settings,
    )

    state = model.state_dict()
    assert state["visual_encoder.norm.weight"].shape == (4,)
    assert state["llama_proj_mlps.0.weight"].shape == (8, 4)
    assert state["llama_proj_mlps.2.weight"].shape == (8, 8)
    assert state["v_q_project.weight"].shape == (6, 4)
    assert state["cross_attention.Wq.weight"].shape == (4, 4)


def _tiny_assembled_model(language_model: nn.Module | None = None) -> BrainVlmMidline:
    settings = ModelSettings(
        vision_width=4,
        llama_width=8,
        compatibility_width=6,
        llama_layers=1,
        key_value_width=2,
    )
    visual = nn.Sequential()
    visual.norm = nn.LayerNorm(4)  # type: ignore[attr-defined]
    return BrainVlmMidline(
        language_model or RecordingLanguageModel(),
        TinyTokenizer(),
        visual,
        settings=settings,
    )


def test_slice_feature_forward_returns_class_logits_and_same_pass_hidden() -> None:
    model = _tiny_assembled_model()
    projected = torch.ones(1, 1, 8)

    logits, hidden = model.extract_projected_features(
        projected,
        "A<ImageHere>B",
        ("1", "2", "3", "4", "5"),
    )

    assert logits.shape == (1, 5)
    assert hidden.shape == (1, 8)
    assert len(model.llama_model.calls) == 1
    assert model.llama_model.calls[0]["output_hidden_states"] is True
    assert model.llama_model.calls[0]["use_cache"] is False


def test_generation_is_greedy_and_stops_before_end_symbol() -> None:
    language_model = RecordingLanguageModel()
    model = _tiny_assembled_model(language_model)
    model.encode_img = lambda image, resolution_type: (  # type: ignore[method-assign]
        torch.ones(image.shape[0], 1, 4),
        torch.ones(image.shape[0], 1, dtype=torch.long),
        None,
    )

    text = model.generate_step(
        torch.zeros(1, 3, 224, 224),
        "A<ImageHere>B",
        None,
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
    )

    assert text == "7 8"
    assert len(language_model.calls) == 3
    assert language_model.calls[0]["past_key_values"] is None
    assert language_model.calls[1]["past_key_values"] == (1,)
