"""Internal BrainVLM-Midline runtime for slice inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn

from midline_agent.datasets.brainvlm import PREPROCESSING_VERSION
from midline_agent.datasets.inference_annotations import InferenceSlice

from .brainvlm_midline.config import FoundationAssets
from .brainvlm_midline.llama import create_llama
from .brainvlm_midline.model import BrainVlmMidline
from .brainvlm_midline.vision import create_biomedclip_vision, preprocess_image
from .checkpoints import BrainVlmCheckpointError, load_brainvlm_checkpoint
from .logits_mil import TUMOR_TYPES


TUMOR_PROMPT = (
    "You are a language vision assistant. Based on the MRI image<ImageHere>, question, and context, "
    "please give an appropriate tumor choice among WHO tumor classification labels. USER:Please predict "
    "the WHO classification for this patient based on the MRI image. Context:This is WHO tumor "
    "classification label: "
    "'H3K27M-mutated midline glioma': '<tumor_1>', 'Pediatric-type diffuse glioma': '<tumor_2>', "
    "'IDH-wildtype glioblastoma': '<tumor_3>', 'IDH-mutated astrocytoma': '<tumor_4>', "
    "'IDH-mutated oligodendroglioma': '<tumor_5>' ASSISTANT:<tumor_"
)
WHO_INSTRUCTION = (
    "Please classify both the WHO tumor type and WHO grade for this midline glioma patient based on "
    "the MRI image."
)
WHO_PROMPT_TEMPLATE = (
    "You are a language vision assistant. Based on the MRI image<ImageHere>, please classify both the "
    "tumor type and WHO grade. USER:{age}{instruction} Context:WHO tumor classification labels: "
    "'<tumor_1>' for H3K27M-mutated midline glioma (always <grade_4>), '<tumor_2>' for Pediatric-type "
    "diffuse glioma (can be <grade_2> or <grade_3> or <grade_4>), '<tumor_3>' for IDH-wildtype "
    "glioblastoma (always <grade_4>), '<tumor_4>' for IDH-mutated astrocytoma (can be <grade_2> or "
    "<grade_3> or <grade_4>), '<tumor_5>' for IDH-mutated oligodendroglioma (can be <grade_2> or "
    "<grade_3>). Your output format: '<tumor_X> <grade_Y>' ASSISTANT:"
)
_CUDA_DEVICE_PATTERN = re.compile(r"cuda:(\d+)")
_CLASS_TOKENS = ("1", "2", "3", "4", "5")


class BrainVlmRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SliceFeatures:
    logits: torch.Tensor
    hidden: torch.Tensor

    def __post_init__(self) -> None:
        logits = self.logits.detach().float().cpu().contiguous()
        hidden = self.hidden.detach().float().cpu().contiguous()
        if logits.shape != (5,) or hidden.shape != (4096,):
            raise ValueError("slice features require five logits and 4096 hidden values")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("slice features must contain only finite values")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "hidden", hidden)


def _validate_device(device: str) -> torch.device:
    if not isinstance(device, str):
        raise BrainVlmRuntimeError("device must be an indexed cuda device")
    match = _CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise BrainVlmRuntimeError("device must use the cuda:N format")
    if not torch.cuda.is_available():
        raise BrainVlmRuntimeError("CUDA is not available")
    index = int(match.group(1))
    if index >= torch.cuda.device_count():
        raise BrainVlmRuntimeError("CUDA device index is unavailable")
    return torch.device(device)


def _construct_internal_model(assets: FoundationAssets) -> BrainVlmMidline:
    language_model, tokenizer = create_llama(assets.llama_model)
    vision = create_biomedclip_vision(assets.biomedclip_pretrained)
    return BrainVlmMidline(language_model, tokenizer, vision)


def _age_description(age: float | None) -> str:
    if age is None:
        return ""
    category = "child" if age <= 18 else "adult"
    return f"This is a {category} patient. "


class BrainVlmRuntime:
    preprocessing_metadata = MappingProxyType(
        {
            "preprocessing_version": PREPROCESSING_VERSION,
            "image_size": 224,
            "normalization": "BiomedCLIP mean and standard deviation",
            "resize": "bicubic",
        }
    )

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls,
        llama_model: Path,
        checkpoint: Path,
        *,
        device: str = "cuda:0",
        biomedclip_pretrained: str | None = None,
    ) -> "BrainVlmRuntime":
        try:
            assets = FoundationAssets.from_paths(
                llama_model,
                checkpoint,
                biomedclip_pretrained,
            )
        except ValueError as exc:
            raise BrainVlmRuntimeError(str(exc)) from exc
        torch_device = _validate_device(device)
        torch.cuda.set_device(torch_device)
        try:
            model = _construct_internal_model(assets)
            load_brainvlm_checkpoint(assets.checkpoint, model)
        except (BrainVlmCheckpointError, RuntimeError, ValueError) as exc:
            raise BrainVlmRuntimeError("unable to construct internal BrainVLM-Midline") from exc
        model.to(torch_device)
        model.eval()
        return cls(model, torch_device)

    @classmethod
    def from_model(cls, model: nn.Module, *, device: str = "cpu") -> "BrainVlmRuntime":
        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()
        return cls(model, torch_device)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        try:
            image = preprocess_image(image_path)
        except RuntimeError as exc:
            raise BrainVlmRuntimeError("unable to decode MRI slice") from exc
        return image.unsqueeze(0).to(self.device)

    def predict_slice_features(
        self,
        image_path: Path,
        metadata: InferenceSlice,
    ) -> SliceFeatures:
        del metadata
        image = self._load_image(image_path)
        try:
            with torch.inference_mode():
                with self.model.maybe_autocast():
                    logits, hidden = self.model.extract_slice_features(
                        image,
                        TUMOR_PROMPT,
                        _CLASS_TOKENS,
                    )
        except Exception as exc:
            raise BrainVlmRuntimeError("BrainVLM-Midline feature inference failed") from exc
        try:
            return SliceFeatures(logits[0], hidden[0])
        except (IndexError, ValueError) as exc:
            raise BrainVlmRuntimeError("BrainVLM-Midline returned invalid slice features") from exc

    def predict_tumor_logits(
        self,
        image_path: Path,
        metadata: InferenceSlice,
    ) -> torch.Tensor:
        return self.predict_slice_features(image_path, metadata).logits

    def predict_who_generation(
        self,
        image_path: Path,
        metadata: InferenceSlice,
        tumor_type: str,
        age: float | None,
    ) -> str:
        del metadata
        if tumor_type not in TUMOR_TYPES:
            raise BrainVlmRuntimeError("unsupported tumor type for WHO generation")
        image = self._load_image(image_path)
        prompt = WHO_PROMPT_TEMPLATE.format(
            age=_age_description(age),
            instruction=WHO_INSTRUCTION,
        )
        try:
            with torch.inference_mode():
                with self.model.maybe_autocast():
                    output = self.model.generate_step(
                        image,
                        prompt,
                        None,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                    )
        except Exception as exc:
            raise BrainVlmRuntimeError("BrainVLM-Midline WHO generation failed") from exc
        if not isinstance(output, str) or not output.strip():
            raise BrainVlmRuntimeError("BrainVLM-Midline WHO generation must return text")
        return output.replace("$", "").strip()
