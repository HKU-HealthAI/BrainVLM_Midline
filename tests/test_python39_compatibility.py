from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest
import torch

from midline_agent.datasets.inference_annotations import InferenceCase, InferenceSlice
from midline_agent.diagnostic.brainvlm_runtime import SliceFeatures
from midline_agent.diagnostic.feature_store import validate_feature_store
from scripts.extract_features import _extract_store


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCE_ROOTS = (REPOSITORY_ROOT / "midline_agent", REPOSITORY_ROOT / "scripts")

PYTHON_310_PLUS_MODULES = frozenset({"tomllib"})
PYTHON_310_PLUS_SYMBOLS = {
    "asyncio": frozenset({"TaskGroup", "timeout", "timeout_at"}),
    "contextlib": frozenset({"aclosing", "chdir"}),
    "dataclasses": frozenset({"KW_ONLY"}),
    "enum": frozenset({"EnumCheck", "FlagBoundary", "ReprEnum", "StrEnum", "verify"}),
    "hashlib": frozenset({"file_digest"}),
    "inspect": frozenset({"get_annotations"}),
    "itertools": frozenset({"batched", "pairwise"}),
    "operator": frozenset({"call"}),
    "statistics": frozenset({"correlation", "covariance", "linear_regression"}),
    "types": frozenset({"EllipsisType", "NoneType", "NotImplementedType"}),
    "typing": frozenset(
        {
            "Concatenate",
            "LiteralString",
            "NotRequired",
            "ParamSpec",
            "Required",
            "Self",
            "TypeAlias",
            "TypeGuard",
            "TypeVarTuple",
            "Unpack",
        }
    ),
}
PYTHON_310_PLUS_BUILTINS = frozenset({"aiter", "anext", "BaseExceptionGroup", "ExceptionGroup"})
PYTHON_310_PLUS_METHODS = frozenset({"hardlink_to", "is_junction", "walk"})
PYTHON_310_PLUS_DATACLASS_OPTIONS = frozenset({"kw_only", "match_args", "slots", "weakref_slot"})


def _release_sources() -> list[Path]:
    return sorted(path for root in RELEASE_SOURCE_ROOTS for path in root.rglob("*.py"))


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _runtime_violations(path: Path, tree: ast.Module) -> list[str]:
    violations: list[str] = []
    module_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name] = alias.name
                if alias.name in PYTHON_310_PLUS_MODULES:
                    violations.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in PYTHON_310_PLUS_MODULES:
                violations.append(f"{path}:{node.lineno}: import from {module}")
            blocked = PYTHON_310_PLUS_SYMBOLS.get(module, frozenset())
            for alias in node.names:
                if alias.name in blocked:
                    violations.append(f"{path}:{node.lineno}: {module}.{alias.name}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name == "zip" and any(keyword.arg == "strict" for keyword in node.keywords):
                violations.append(f"{path}:{node.lineno}: zip(strict=...)")
            if name is not None and name.rsplit(".", 1)[-1] == "dataclass":
                for keyword in node.keywords:
                    if keyword.arg in PYTHON_310_PLUS_DATACLASS_OPTIONS:
                        violations.append(f"{path}:{node.lineno}: dataclass({keyword.arg}=...)")
            if name in PYTHON_310_PLUS_BUILTINS:
                violations.append(f"{path}:{node.lineno}: {name}()")
        elif isinstance(node, ast.Attribute):
            root = _qualified_name(node.value)
            module = module_aliases.get(root or "", root or "")
            if node.attr in PYTHON_310_PLUS_SYMBOLS.get(module, frozenset()):
                violations.append(f"{path}:{node.lineno}: {module}.{node.attr}")
            receiver = _qualified_name(node.value.func) if isinstance(node.value, ast.Call) else None
            if node.attr in PYTHON_310_PLUS_METHODS and receiver in {"Path", "pathlib.Path"}:
                violations.append(f"{path}:{node.lineno}: .{node.attr}()")

    return violations


def test_release_sources_parse_with_python_39_grammar() -> None:
    failures: list[str] = []
    for path in _release_sources():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
        except SyntaxError as exc:
            failures.append(f"{path}:{exc.lineno}: {exc.msg}")

    assert not failures, "\n".join(failures)


def test_release_sources_avoid_python_310_plus_runtime_apis() -> None:
    violations: list[str] = []
    for path in _release_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_runtime_violations(path, tree))

    assert not violations, "\n".join(violations)


def test_feature_extraction_runs_with_python_39_zip_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "slice.png"
    image.write_bytes(b"public-test-image")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"public-test-checkpoint")
    case = InferenceCase(
        patient_id="case-a",
        slices=(
            InferenceSlice(
                patient_id="case-a",
                image_path=image,
                modality="T1CE",
                slice_index=0,
            ),
        ),
    )

    class Runtime:
        def predict_slice_features(self, image_path: Path, metadata: InferenceSlice) -> SliceFeatures:
            return SliceFeatures(logits=torch.zeros(5), hidden=torch.zeros(4096))

    python_zip = builtins.zip

    def python_39_zip(*iterables, **kwargs):
        if kwargs:
            raise TypeError("zip() takes no keyword arguments")
        return python_zip(*iterables)

    monkeypatch.setattr(builtins, "zip", python_39_zip)

    store = _extract_store(case, Runtime(), checkpoint, None, (1,))
    validate_feature_store(store)
    patient = store["patients"]["case-a"]

    assert patient["slice_ids"] == ["case-a:T1CE:0"]
