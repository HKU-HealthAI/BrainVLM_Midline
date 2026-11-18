"""Official public bridge for a user-owned external BrainVLM checkout."""

from __future__ import annotations

import importlib.util
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..datasets.brainvlm import BrainVlmRequest, build_inference_request
from ..datasets.prompts import (
    COMBINED_TASK,
    PROMPT_VERSION,
    SUPPORTED_TASKS,
    TUMOR_TASK,
    LabelError,
    validate_generated_label,
)
from .adapters import WhoGradeRequest, WhoGradeSliceResult


class ExternalSourceError(RuntimeError):
    """Raised when the external source or checkpoint violates the adapter contract."""


@dataclass(frozen=True)
class BrainVlmPrediction:
    """Validated prediction returned by the external runner."""

    task: str
    label: str
    raw_text: str | None
    logits: np.ndarray | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.task not in SUPPORTED_TASKS:
            raise ValueError("prediction task is not supported")
        try:
            normalized_label = validate_generated_label(self.task, self.label)
        except LabelError as exc:
            raise ValueError("prediction label is invalid for the task") from exc
        if normalized_label != self.label:
            raise ValueError("prediction label must use the canonical task vocabulary")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise ValueError("prediction raw_text must be a string or None")
        if self.logits is not None:
            logits = np.asarray(self.logits, dtype=np.float32).copy()
            if logits.ndim != 1 or not bool(np.isfinite(logits).all()):
                raise ValueError("prediction logits must be a finite one-dimensional array")
            if self.task in {TUMOR_TASK, COMBINED_TASK} and logits.shape != (5,):
                raise ValueError("tumor prediction logits must have shape (5,)")
            logits.setflags(write=False)
            object.__setattr__(self, "logits", logits)
        if self.task in {TUMOR_TASK, COMBINED_TASK} and self.logits is None:
            raise ValueError("tumor prediction logits are required")
        metadata = dict(self.metadata)
        if metadata.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("prediction metadata must contain the public prompt version")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


_FACTORY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_PART_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOURCE_IMPORT_LOCK = threading.RLock()


def _source_module_names(source_root: Path) -> set[str]:
    names: set[str] = set()
    for path in source_root.rglob("*.py"):
        relative = path.relative_to(source_root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts and all(_MODULE_PART_PATTERN.fullmatch(part) for part in parts):
            names.add(".".join(parts))
    return names


def _resolve_factory_module(source_root: Path, module_name: str) -> tuple[Path, bool]:
    parts = module_name.split(".")
    if not all(_MODULE_PART_PATTERN.fullmatch(part) for part in parts):
        raise ExternalSourceError("factory module name contains an invalid component")
    root = source_root.resolve()
    candidates = [(root.joinpath(*parts).with_suffix(".py"), False), (root.joinpath(*parts, "__init__.py"), True)]
    for candidate, is_package in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ExternalSourceError("factory module must remain inside source root") from exc
        if resolved.is_file():
            return resolved, is_package
    raise ExternalSourceError("external BrainVLM factory module was not found under source root")


def _import_factory(source_root: Path, factory_path: str):
    if not _FACTORY_PATTERN.fullmatch(factory_path):
        raise ExternalSourceError("factory path must be formatted module:callable")
    module_name, callable_name = factory_path.split(":", maxsplit=1)
    module_path, is_package = _resolve_factory_module(source_root, module_name)
    isolated_name = module_name
    search_locations = [str(module_path.parent)] if is_package else None
    spec = importlib.util.spec_from_file_location(
        isolated_name,
        module_path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ExternalSourceError("unable to create an isolated external BrainVLM module")
    module = importlib.util.module_from_spec(spec)
    package_registration = True
    source_names = _source_module_names(source_root)
    with _SOURCE_IMPORT_LOCK:
        original_path = list(sys.path)
        original_modules = dict(sys.modules)
        try:
            for name in source_names:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(source_root))
            if package_registration:
                sys.modules[isolated_name] = module
            spec.loader.exec_module(module)
            root = source_root.resolve()
            for name in source_names:
                loaded = sys.modules.get(name)
                origin = getattr(loaded, "__file__", None)
                if origin is None:
                    continue
                try:
                    Path(origin).resolve().relative_to(root)
                except ValueError as exc:
                    raise ExternalSourceError("external dependency resolved outside source root") from exc
            factory = getattr(module, callable_name, None)
            if not callable(factory):
                raise ExternalSourceError("external BrainVLM factory is not callable")
            return factory
        except ExternalSourceError:
            raise
        except Exception as exc:
            raise ExternalSourceError("unable to import the external BrainVLM factory") from exc
        finally:
            sys.path[:] = original_path
            for name in list(sys.modules):
                if name not in original_modules:
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)


def _load_runner(source_root: Path, checkpoint: Path, factory_path: str):
    factory = _import_factory(source_root, factory_path)
    try:
        runner = factory(checkpoint=str(checkpoint))
    except Exception as exc:
        raise ExternalSourceError("external BrainVLM factory failed to load checkpoint") from exc
    if not callable(getattr(runner, "predict", None)):
        raise ExternalSourceError("external BrainVLM runner must expose predict")
    return runner


def _output_value(output: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in output:
            return output[name]
    return None


class BrainVlmAdapter:
    """Load and call a user-owned BrainVLM model through a narrow interface.

    The external factory must be importable as ``module:callable`` from
    ``source_root`` and accept ``checkpoint=...``. It must return an object
    exposing ``predict(*, task, prompt, image, metadata)``. No external source
    files or checkpoint bytes are copied into this package.
    """

    def __init__(self, source_root: Path | str, checkpoint: Path | str, factory_path: str) -> None:
        source = Path(source_root).expanduser().resolve()
        model_checkpoint = Path(checkpoint).expanduser().resolve()
        if not source.is_dir():
            raise ExternalSourceError("external BrainVLM source root must be an existing directory")
        if not model_checkpoint.is_file():
            raise ExternalSourceError("external BrainVLM checkpoint must be an existing file")
        self.source_root = source
        self.checkpoint = model_checkpoint
        self.factory_path = factory_path
        self._runner = _load_runner(source, model_checkpoint, factory_path)

    def predict(self, request: BrainVlmRequest) -> BrainVlmPrediction:
        """Run one constrained task and validate the external response."""
        if not isinstance(request, BrainVlmRequest):
            raise TypeError("request must be a BrainVlmRequest")
        if request.prompt_version != PROMPT_VERSION:
            raise ExternalSourceError("request prompt version does not match BrainVLM adapter")
        try:
            output = self._runner.predict(
                task=request.task,
                prompt=request.prompt,
                image=request.image,
                metadata=dict(request.metadata),
            )
        except Exception as exc:
            raise ExternalSourceError("external BrainVLM prediction failed") from exc
        if not isinstance(output, Mapping):
            raise ExternalSourceError("external BrainVLM prediction must be a mapping")
        raw_label = _output_value(output, "label", "prediction")
        if raw_label is None:
            raise ExternalSourceError("external BrainVLM prediction is missing label")
        try:
            label = validate_generated_label(request.task, raw_label)
        except LabelError as exc:
            raise ExternalSourceError(str(exc)) from exc
        raw_text = _output_value(output, "raw_text", "text")
        if raw_text is not None and not isinstance(raw_text, str):
            raise ExternalSourceError("external BrainVLM raw text must be a string")
        raw_logits = _output_value(output, "logits", "tumor_logits")
        logits = None
        if raw_logits is not None:
            logits = np.asarray(raw_logits, dtype=np.float32)
            if logits.shape != (5,):
                raise ExternalSourceError("external BrainVLM tumor logits must have shape (5,)")
            if not bool(np.isfinite(logits).all()):
                raise ExternalSourceError("external BrainVLM tumor logits must be finite")
        if request.task in {TUMOR_TASK, COMBINED_TASK} and logits is None:
            raise ExternalSourceError("tumor prediction is missing five-class logits")
        if request.task == COMBINED_TASK:
            if not raw_text:
                raise ExternalSourceError("combined tumor/WHO prediction is missing constrained text")
            try:
                text_label = validate_generated_label(request.task, raw_text)
            except LabelError as exc:
                raise ExternalSourceError("combined tumor/WHO text has invalid constrained labels") from exc
            if text_label != label:
                raise ExternalSourceError("combined tumor/WHO label does not match constrained text")
        return BrainVlmPrediction(request.task, label, raw_text, logits, request.metadata)

    def predict_who_grade(self, request: WhoGradeRequest) -> WhoGradeSliceResult:
        """Adapt the combined task to the legacy diagnostic protocol."""
        if request.prompt_version != PROMPT_VERSION:
            raise ExternalSourceError("caller prompt version does not match BrainVLM adapter")
        image = np.asarray(request.image)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        inference = build_inference_request(COMBINED_TASK, image)
        prediction = self.predict(inference)
        if not prediction.raw_text or prediction.logits is None:
            raise ExternalSourceError("combined tumor/WHO prediction is incomplete")
        return WhoGradeSliceResult(
            raw_text=prediction.raw_text,
            logits=prediction.logits,
            prompt_version=request.prompt_version,
        )


def load_brainvlm_adapter(
    source_root: Path | str,
    checkpoint: Path | str,
    factory_path: str,
) -> BrainVlmAdapter:
    """Construct the external adapter with explicit source and checkpoint paths."""
    return BrainVlmAdapter(source_root, checkpoint, factory_path)
