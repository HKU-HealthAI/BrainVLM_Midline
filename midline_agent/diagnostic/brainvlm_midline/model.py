"""Checkpoint-compatible model assembly for BrainVLM-Midline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelSettings
from .llama import build_supervised_batch, causal_lm_loss
from .vision import encode_low_resolution


class ResolutionAttention(nn.Module):
    """Projected cross-attention retained for checkpoint compatibility."""

    def __init__(self, width: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.Wq = nn.Linear(width, width)
        self.Wk = nn.Linear(width, width)
        self.Wv = nn.Linear(width, width)
        self.output_linear = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, low_resolution: torch.Tensor, high_resolution: torch.Tensor) -> torch.Tensor:
        query = self.Wq(low_resolution)
        key = self.Wk(high_resolution)
        value = self.Wv(high_resolution)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = self.output_linear(torch.matmul(weights, value))
        return self.norm(query + self.dropout_2(attended))


class BrainVlmMidline(nn.Module):
    """Focused vision-language model used by the Diagnostic Sub-Agent."""

    def __init__(
        self,
        llama_model: nn.Module,
        llama_tokenizer: Any,
        visual_encoder: nn.Module,
        *,
        settings: ModelSettings | None = None,
        max_text_length: int = 256,
        max_context_length: int = 3800,
        end_symbol: str = "###",
    ) -> None:
        super().__init__()
        self.settings = settings or ModelSettings()
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer
        self.visual_encoder = visual_encoder
        self.llama_proj_mlps = nn.Sequential(
            nn.Linear(self.settings.vision_width, self.settings.llama_width),
            nn.GELU(),
            nn.Linear(self.settings.llama_width, self.settings.llama_width),
        )
        self.v_q_project = nn.Linear(
            self.settings.vision_width,
            self.settings.compatibility_width,
        )
        self.cross_attention = ResolutionAttention(self.settings.vision_width)
        self.max_text_length = max_text_length
        self.max_context_length = max_context_length
        self.end_symbol = end_symbol

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def maybe_autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.llama_model.get_input_embeddings()(token_ids)

    def encode_img(
        self,
        image: torch.Tensor,
        resolution_type: str = "LR_Encoding",
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if resolution_type != "LR_Encoding":
            raise ValueError("only LR_Encoding is supported by the focused public model")
        features = encode_low_resolution(self.visual_encoder, image)
        features = features.unsqueeze(1)
        attention = torch.ones(features.shape[:2], dtype=torch.long, device=features.device)
        return features, attention, None

    def prompt_wrap(
        self,
        image_embeddings: torch.Tensor,
        image_attention: torch.Tensor,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del image_attention
        if len(prompts) != image_embeddings.shape[0]:
            raise ValueError("prompt batch size must match image embeddings")
        rows: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for index, prompt in enumerate(prompts):
            if prompt.count("<ImageHere>") != 1:
                raise ValueError("each prompt must contain exactly one <ImageHere> placeholder")
            left, right = prompt.split("<ImageHere>")
            left_ids = self.llama_tokenizer(
                left,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(image_embeddings.device)
            right_ids = self.llama_tokenizer(
                right,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(image_embeddings.device)
            row = torch.cat(
                [
                    self.embed_tokens(left_ids),
                    image_embeddings[index : index + 1],
                    self.embed_tokens(right_ids),
                ],
                dim=1,
            )[:, : self.max_context_length]
            rows.append(row[0])
            ids.append(torch.cat([left_ids, right_ids], dim=1)[0])
        width = max(row.shape[0] for row in rows)
        id_width = max(item.shape[0] for item in ids)
        pad_id = int(self.llama_tokenizer.pad_token_id)
        pad = self.embed_tokens(torch.tensor(pad_id, device=image_embeddings.device))
        embeddings = pad.expand(len(rows), width, -1).clone()
        attention = torch.zeros((len(rows), width), dtype=torch.long, device=image_embeddings.device)
        token_ids = torch.full(
            (len(rows), id_width),
            pad_id,
            dtype=torch.long,
            device=image_embeddings.device,
        )
        for index, (row, item_ids) in enumerate(zip(rows, ids)):
            embeddings[index, : row.shape[0]] = row
            attention[index, : row.shape[0]] = 1
            token_ids[index, : item_ids.shape[0]] = item_ids
        return embeddings, attention, token_ids

    def forward(self, samples: dict[str, Any], reduction: str = "mean") -> dict[str, torch.Tensor]:
        images = samples.get("image")
        prompts = samples.get("instruction_input")
        answers = samples.get("answer")
        if not isinstance(images, torch.Tensor) or not isinstance(prompts, Sequence) or not isinstance(answers, Sequence):
            raise ValueError("samples require image, instruction_input, and answer batches")
        visual, _, _ = self.encode_img(images, "LR_Encoding")
        projected = self.llama_proj_mlps(visual)
        batch = build_supervised_batch(
            self.llama_tokenizer,
            self.embed_tokens,
            projected,
            prompts,
            answers,
            end_symbol=self.end_symbol,
            max_text_length=self.max_text_length,
            max_context_length=self.max_context_length,
        )
        outputs = self.llama_model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return {"loss": causal_lm_loss(outputs.logits.float(), batch.labels, reduction=reduction)}

    def _with_bos(
        self,
        embeddings: torch.Tensor,
        attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bos = torch.full(
            (embeddings.shape[0], 1),
            int(self.llama_tokenizer.bos_token_id),
            dtype=torch.long,
            device=embeddings.device,
        )
        return (
            torch.cat([self.embed_tokens(bos), embeddings], dim=1),
            torch.cat([torch.ones_like(attention[:, :1]), attention], dim=1),
        )

    def extract_projected_features(
        self,
        projected_image: torch.Tensor,
        prompt: str,
        class_tokens: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and the final prompt hidden state from one pass."""
        attention = torch.ones(
            projected_image.shape[:2],
            dtype=torch.long,
            device=projected_image.device,
        )
        inputs, input_attention, _ = self.prompt_wrap(
            projected_image,
            attention,
            [prompt] * projected_image.shape[0],
        )
        inputs, input_attention = self._with_bos(inputs, input_attention)
        outputs = self.llama_model(
            inputs_embeds=inputs,
            attention_mask=input_attention,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]
        hidden = final_hidden[:, -1, :]
        vocabulary_logits = self.llama_model.lm_head(hidden)
        token_ids: list[int] = []
        for token in class_tokens:
            encoded = self.llama_tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"class token {token!r} must encode to one token")
            token_ids.append(int(encoded[0]))
        logits = vocabulary_logits[:, token_ids]
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("language model returned non-finite features")
        return logits.float(), hidden.float()

    def extract_slice_features(
        self,
        image: torch.Tensor,
        prompt: str,
        class_tokens: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual, _, _ = self.encode_img(image, "LR_Encoding")
        return self.extract_projected_features(
            self.llama_proj_mlps(visual),
            prompt,
            class_tokens,
        )

    @torch.no_grad()
    def generate_step(
        self,
        images: torch.Tensor,
        texts: str,
        high_resolution_images: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        do_sample: bool = False,
        num_beams: int = 1,
        **unused: Any,
    ) -> str:
        """Generate one deterministic report with the historical stop symbol."""
        del unused
        if high_resolution_images is not None:
            raise ValueError("high-resolution generation is not supported")
        if temperature != 0.0 or do_sample or num_beams != 1:
            raise ValueError("the public inference path supports greedy generation only")
        visual, attention, _ = self.encode_img(images, "LR_Encoding")
        projected = self.llama_proj_mlps(visual)
        inputs, input_attention, _ = self.prompt_wrap(projected, attention, [texts])
        inputs, input_attention = self._with_bos(inputs, input_attention)
        stop_id = int(self.llama_tokenizer.encode(self.end_symbol)[-1])
        past = None
        generated: list[int] = []
        current = inputs
        for step in range(max_new_tokens):
            outputs = self.llama_model(
                past_key_values=past,
                inputs_embeds=current,
                attention_mask=input_attention if step == 0 else None,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            next_token = int(torch.argmax(outputs.logits[:, -1], dim=-1).item())
            if next_token == stop_id:
                break
            generated.append(next_token)
            token = torch.tensor([next_token], dtype=torch.long, device=inputs.device)
            current = self.embed_tokens(token).reshape(1, 1, -1)
            past = outputs.past_key_values
        return self.llama_tokenizer.decode(generated).replace("$", "").strip()
