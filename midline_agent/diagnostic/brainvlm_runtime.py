"""Concrete inference bridge for an external BrainVLM source checkout."""

from __future__ import annotations

import importlib
import pickle
import re
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from midline_agent.datasets.brainvlm import PREPROCESSING_VERSION, preprocess_slice
from midline_agent.datasets.inference_annotations import InferenceSlice

from .logits_mil import TUMOR_TYPES


BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)
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
_IMPORT_LOCK = threading.RLock()


class BrainVlmRuntimeError(RuntimeError):
    pass


def _validate_external_source(source_root: Path) -> Path:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise BrainVlmRuntimeError("BrainVLM source root must be an existing directory") from exc
    if not root.is_dir():
        raise BrainVlmRuntimeError("BrainVLM source root must be an existing directory")
    required = (root / "minigpt4" / "__init__.py", root / "minigpt4" / "models" / "minigpt4.py")
    for candidate in required:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise BrainVlmRuntimeError("external minigpt4 package must remain inside source root") from exc
        if not resolved.is_file():
            raise BrainVlmRuntimeError("external minigpt4 package is incomplete")
    return root


def _validate_module_origins(source_root: Path, modules: Mapping[str, object]) -> None:
    root = Path(source_root).resolve()
    for name, module in modules.items():
        if name != "minigpt4" and not name.startswith("minigpt4."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(root)
        except ValueError as exc:
            raise BrainVlmRuntimeError("external minigpt4 module resolved outside source root") from exc


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


def load_brainvlm_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> tuple[str, ...]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise BrainVlmRuntimeError("BrainVLM checkpoint path must be a file")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise BrainVlmRuntimeError("unable to load BrainVLM checkpoint safely") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise BrainVlmRuntimeError("BrainVLM checkpoint is missing the model state mapping")
    state = checkpoint["model"]
    if not state:
        raise BrainVlmRuntimeError("BrainVLM checkpoint model state is empty")
    expected = model.state_dict()
    unknown = sorted(set(state) - set(expected))
    if unknown:
        raise BrainVlmRuntimeError(f"BrainVLM checkpoint contains unrecognized tensor: {unknown[0]}")
    checked_state: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise BrainVlmRuntimeError(f"BrainVLM checkpoint tensor {name} is not a tensor")
        if tensor.shape != expected[name].shape:
            raise BrainVlmRuntimeError(f"BrainVLM checkpoint tensor {name} has an incompatible shape")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise BrainVlmRuntimeError(f"BrainVLM checkpoint tensor {name} must be finite")
        checked_state[name] = tensor
    try:
        incompatible = model.load_state_dict(checked_state, strict=False)
    except RuntimeError as exc:
        raise BrainVlmRuntimeError("BrainVLM checkpoint tensors could not be loaded") from exc
    if incompatible.unexpected_keys:
        raise BrainVlmRuntimeError("BrainVLM checkpoint contains unexpected model tensors")
    return tuple(sorted(checked_state))


@contextmanager
def _external_model_class(source_root: Path) -> Iterator[type]:
    with _IMPORT_LOCK:
        previous_path = list(sys.path)
        previous_modules = {name: module for name, module in sys.modules.items() if name == "minigpt4" or name.startswith("minigpt4.")}
        for name in previous_modules:
            sys.modules.pop(name, None)
        try:
            sys.path.insert(0, str(source_root))
            importlib.invalidate_caches()
            try:
                module = importlib.import_module("minigpt4.models.minigpt4")
                loaded = {
                    name: value
                    for name, value in sys.modules.items()
                    if name == "minigpt4" or name.startswith("minigpt4.")
                }
                _validate_module_origins(source_root, loaded)
                model_class = getattr(module, "MiniGPT4", None)
                if model_class is None or not callable(getattr(model_class, "from_config", None)):
                    raise BrainVlmRuntimeError("external source does not expose MiniGPT4.from_config")
            except BrainVlmRuntimeError:
                raise
            except Exception as exc:
                raise BrainVlmRuntimeError("unable to import the external BrainVLM source") from exc
            yield model_class
        finally:
            for name in list(sys.modules):
                if name == "minigpt4" or name.startswith("minigpt4."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous_modules)
            sys.path[:] = previous_path


@contextmanager
def _biomedclip_override(pretrained: str | None) -> Iterator[None]:
    if pretrained is None:
        yield
        return
    if not isinstance(pretrained, str) or not pretrained.strip():
        raise BrainVlmRuntimeError("BiomedCLIP pretrained location must be a non-empty string")
    try:
        open_clip = importlib.import_module("open_clip")
        original = open_clip.create_model_and_transforms
    except Exception as exc:
        raise BrainVlmRuntimeError("open_clip is required to construct BrainVLM") from exc

    def create_model_and_transforms(model_name, *args, **kwargs):
        kwargs["pretrained"] = pretrained
        return original(model_name, *args, **kwargs)

    open_clip.create_model_and_transforms = create_model_and_transforms
    try:
        yield
    finally:
        open_clip.create_model_and_transforms = original


def _construct_external_model(
    source_root: Path,
    llama_model: Path,
    biomedclip_pretrained: str | None,
) -> torch.nn.Module:
    config: dict[str, Any] = {
        "vit_model": "eva_clip_g",
        "q_former_model": "",
        "image_size": 224,
        "num_query_token": 32,
        "llama_model": str(llama_model),
        "drop_path_rate": 0,
        "use_grad_checkpoint": False,
        "vit_precision": "fp16",
        "freeze_vit": True,
        "has_qformer": False,
        "freeze_qformer": True,
        "low_resource": False,
        "device_8bit": 0,
        "prompt_path": "",
        "prompt_template": "###Human: {} ###Assistant: ",
        "max_txt_len": 256,
        "end_sym": "###",
        "model_2d_or_3d": "2d",
        "ckpt": "",
    }
    try:
        with _external_model_class(source_root) as model_class:
            with _biomedclip_override(biomedclip_pretrained):
                model = model_class.from_config(config)
    except BrainVlmRuntimeError:
        raise
    except Exception as exc:
        raise BrainVlmRuntimeError("external BrainVLM model construction failed") from exc
    if not isinstance(model, torch.nn.Module):
        raise BrainVlmRuntimeError("external MiniGPT4 factory did not return a torch module")
    return model


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

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(
        cls,
        source_root: Path,
        llama_model: Path,
        checkpoint: Path,
        *,
        device: str = "cuda:0",
        biomedclip_pretrained: str | None = None,
    ) -> "BrainVlmRuntime":
        source = _validate_external_source(source_root)
        llama_path = Path(llama_model).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not llama_path.is_dir():
            raise BrainVlmRuntimeError("Llama model path must be an existing directory")
        if not checkpoint_path.is_file():
            raise BrainVlmRuntimeError("BrainVLM checkpoint path must be an existing file")
        torch_device = _validate_device(device)
        torch.cuda.set_device(torch_device)
        model = _construct_external_model(source, llama_path, biomedclip_pretrained)
        load_brainvlm_checkpoint(model, checkpoint_path)
        model.to(torch_device)
        model.eval()
        return cls(model, torch_device)

    @classmethod
    def from_model(cls, model: torch.nn.Module, *, device: str = "cpu") -> "BrainVlmRuntime":
        """Construct a runtime around an in-memory model for contract testing."""

        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()
        return cls(model, torch_device)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        try:
            with Image.open(image_path) as image:
                source = np.asarray(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise BrainVlmRuntimeError("unable to decode MRI slice") from exc
        processed = preprocess_slice(source, size=224)
        image = torch.from_numpy(np.array(processed.image, copy=True))
        mean = torch.tensor(BIOMEDCLIP_MEAN, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(BIOMEDCLIP_STD, dtype=image.dtype).view(3, 1, 1)
        normalized = (image - mean) / std
        if not bool(torch.isfinite(normalized).all()):
            raise BrainVlmRuntimeError("preprocessed MRI slice contains non-finite values")
        return normalized.unsqueeze(0).to(self.device)

    def predict_tumor_logits(self, image_path: Path, metadata: InferenceSlice) -> torch.Tensor:
        del metadata
        image = self._load_image(image_path)
        tokenizer = self.model.llama_tokenizer
        token_ids: list[int] = []
        for label in ("1", "2", "3", "4", "5"):
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if not isinstance(encoded, list) or len(encoded) != 1 or not isinstance(encoded[0], int):
                raise BrainVlmRuntimeError(f"tumor class token {label} must encode to exactly one token")
            token_ids.append(encoded[0])
        try:
            with torch.inference_mode():
                image_embeddings, image_attention, _ = self.model.encode_img(image, "LR_Encoding")
                image_embeddings = self.model.llama_proj_mlps(image_embeddings)
                image_embeddings = image_embeddings.reshape(1, -1, image_embeddings.shape[-1])
                inputs, attention, _ = self.model.prompt_wrap(image_embeddings, image_attention, [TUMOR_PROMPT])
                bos = torch.ones_like(attention[:, :1]) * tokenizer.bos_token_id
                inputs = torch.cat([self.model.embed_tokens(bos), inputs], dim=1)
                attention = torch.cat([attention[:, :1], attention], dim=1)
                with self.model.maybe_autocast():
                    outputs = self.model.llama_model(
                        inputs_embeds=inputs,
                        attention_mask=attention,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                final_hidden = outputs.hidden_states[-1]
                if not isinstance(final_hidden, torch.Tensor):
                    raise BrainVlmRuntimeError("language model returned non-tensor hidden states")
                if final_hidden.ndim == 3:
                    hidden = final_hidden[:, -1, :]
                elif final_hidden.ndim == 2:
                    hidden = final_hidden[-1, :].unsqueeze(0)
                else:
                    raise BrainVlmRuntimeError("language model returned incompatible hidden states")
                vocabulary_logits = self.model.llama_model.lm_head(hidden)[0]
                if any(token_id < 0 or token_id >= vocabulary_logits.shape[0] for token_id in token_ids):
                    raise BrainVlmRuntimeError("tumor class token is outside the language-model vocabulary")
                logits = torch.stack([vocabulary_logits[token_id] for token_id in token_ids]).float().cpu()
        except BrainVlmRuntimeError:
            raise
        except Exception as exc:
            raise BrainVlmRuntimeError("BrainVLM tumor inference failed") from exc
        if logits.shape != (5,) or not bool(torch.isfinite(logits).all()):
            raise BrainVlmRuntimeError("BrainVLM tumor inference returned invalid logits")
        return logits

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
        prompt = WHO_PROMPT_TEMPLATE.format(age=_age_description(age), instruction=WHO_INSTRUCTION)
        try:
            with torch.inference_mode():
                output = self.model.generate_step(
                    image,
                    prompt,
                    None,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                )
        except Exception as exc:
            raise BrainVlmRuntimeError("BrainVLM WHO generation failed") from exc
        if not isinstance(output, str) or not output.strip():
            raise BrainVlmRuntimeError("BrainVLM WHO generation must return a non-empty string")
        return output.replace("$", "").strip()
