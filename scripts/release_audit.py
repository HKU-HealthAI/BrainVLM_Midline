#!/usr/bin/env python3
"""Reject files that must not be included in a public source release."""

from __future__ import annotations

import argparse
import ast
import json
import re
import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519"}
ALLOWED_ENV_TEMPLATE_NAMES = {".env.example"}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bin",
    ".ckpt",
    ".dcm",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".onnx",
    ".pdf",
    ".pem",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".tar",
    ".tif",
    ".tiff",
    ".xls",
    ".xlsx",
    ".zip",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    "baseline",
    "baselines",
    "cache",
    "checkpoint",
    "checkpoints",
    "demo",
    "demos",
    "evaluation",
    "evaluations",
    "examples",
    "feature",
    "features",
    "logs",
    "output",
    "outputs",
    "prediction",
    "predictions",
    "report",
    "reports",
    "result",
    "results",
    "weight",
    "weights",
    "paper",
    ".venv",
    "venv",
}
FORBIDDEN_PROGRAM_NAME = re.compile(
    r"(?:^|[_-])(?:demo|eval|evaluate|evaluation|baseline)(?:[_-]|$)",
    re.IGNORECASE,
)
FORBIDDEN_ARTIFACT_NAME = re.compile(
    r"(?:^|[_-])(?:prediction|result)(?:s)?(?:[_-]|$)",
    re.IGNORECASE,
)
CHECKPOINT_ARTIFACT_NAME = re.compile(
    r"(?:^|[_-])checkpoints?(?:[_-]|$)",
    re.IGNORECASE,
)
SOURCE_IDENTIFIER_SUFFIXES = {".bash", ".md", ".py", ".pyi", ".rst", ".sbatch", ".sh"}
GENERATED_FEATURE_ARTIFACT_NAME = re.compile(
    r"(?:^|[_-])features?(?:[_-]|$)",
    re.IGNORECASE,
)
RESULT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".toml",
    ".tsv",
    ".txt",
    ".xls",
    ".xlsx",
    ".yaml",
    ".yml",
}
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
PROVIDER_SECRET_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)
PRIVATE_KEY_PATTERN = re.compile(
    r"-{5}BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-{5}",
    re.IGNORECASE,
)
ASSIGNMENT_PATTERN = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?P<name>[A-Za-z_][A-Za-z0-9_-]*)"
    r"[ \t]*[:=][ \t]*(?P<value>[^\r\n#]*)",
    re.IGNORECASE | re.MULTILINE,
)
ENV_REFERENCE_PATTERN = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
WINDOWS_PRIVATE_PATH_PATTERN = re.compile(
    rf"[A-Za-z]:(?:\\|/){'Us' + 'ers'}(?:\\|/)[^\s'\"]+",
    re.IGNORECASE,
)
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:._-])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+"
)
URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"`]+", re.IGNORECASE)
SAFE_POSIX_PATH_PREFIXES = (
    "/bin/",
    "/dev/",
    "/etc/",
    "/path/to/",
    "/proc/",
    "/sbin/",
    "/sys/",
    "/tmp/",
    "/usr/bin/",
    "/usr/local/bin/",
    "/var/tmp/",
)
TEST_FIXTURE_POSIX_PREFIXES = (
    "/absolute/",
    "/inputs/",
    "/models/",
    "/private/",
    "/run/",
    "/secure/",
    "/synthetic-storage/",
)
HAN_PATTERN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
CUSTOM_ENDPOINT_TERMS = (
    "OPENAI_BASE" + "_URL",
    "api" + "_base",
    "base" + "_url",
    "openai" + " compatible",
    "openai" + "-compatible",
)
RELAY_SERVICE_NAMES = (
    "lite" + "llm",
    "local" + "ai",
    "new" + "api",
    "oll" + "ama",
    "one" + "api",
    "open" + "router",
    "v" + "llm",
)
POLICY_CONTENT_PATHS = (
    "scripts/release_audit.py",
    "tests/test_release_audit.py",
)
APPROVED_PROHIBITION_PATHS = (
    "docs/superpowers/plans/2026-11-27-self-contained-brainvlm-midline.md",
    "docs/superpowers/specs/2026-11-27-self-contained-brainvlm-midline-design.md",
)
APPROVED_DESIGN_PATH = APPROVED_PROHIBITION_PATHS[1]
APPROVED_PROVENANCE_PATH = "docs/provenance/brainvlm-midline-sources.md"
REMOVED_ENVIRONMENT_NAMES = (
    "PYTHON" + "PATH",
    "BRAINVLM_" + "SOURCE",
    "BRAINVLM_SOURCE_" + "ROOT",
)
REMOVED_CLI_OPTIONS = (
    "--" + "adapter",
    "--" + "brainvlm-source",
    "--" + "model-factory",
    "--" + "source-root",
)
TEXT_INTERFACE_SUFFIXES = {
    ".bash",
    ".env",
    ".json",
    ".md",
    ".rst",
    ".sbatch",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SHELL_SUFFIXES = {".bash", ".sbatch", ".sh"}
LEGACY_RUNTIME_ROOT = "mini" + "gpt4"
PROHIBITION_MARKERS = (
    "do not",
    "does not",
    "forbid",
    "must not",
    "neither",
    "no executable",
    "not accept",
    "not committed",
    "reject",
    "unset",
    "without",
)
PUBLIC_TEST_IMAGE_DIRECTORY = Path("test_data/utsw_glioma_bt0156/images")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NON_LITERAL = object()


@dataclass(frozen=True)
class Violation:
    path: Path
    code: str
    detail: str


@dataclass(frozen=True)
class CandidateSource:
    path: Path
    content: bytes
    is_symlink: bool = False


def _is_policy_content(path: Path, root: Path | None) -> bool:
    return any(
        _matches_release_path(path, release_path, root)
        for release_path in POLICY_CONTENT_PATHS
    )


def _has_forbidden_path(path: Path, root: Path | None = None) -> bool:
    policy_path = path
    if root is not None:
        try:
            policy_path = path.absolute().relative_to(root)
        except ValueError:
            policy_path = Path(path.name)
    normalized_parts = {part.lower() for part in policy_path.parts}
    lowered_name = path.name.lower()
    if lowered_name in FORBIDDEN_NAMES or (
        lowered_name.startswith(".env.") and lowered_name not in ALLOWED_ENV_TEMPLATE_NAMES
    ):
        return True
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    if normalized_parts & FORBIDDEN_PARTS:
        return True
    if FORBIDDEN_PROGRAM_NAME.search(path.stem):
        return True
    if (
        CHECKPOINT_ARTIFACT_NAME.search(path.stem)
        and path.suffix.lower() not in SOURCE_IDENTIFIER_SUFFIXES
    ):
        return True
    if path.suffix.lower() not in RESULT_ARTIFACT_SUFFIXES:
        return False
    return bool(
        FORBIDDEN_ARTIFACT_NAME.search(path.stem)
        or GENERATED_FEATURE_ARTIFACT_NAME.search(path.stem)
    )


def _is_public_test_png_path(path: Path, root: Path | None) -> bool:
    if root is None or path.suffix.lower() != ".png":
        return False
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        return False
    return relative.parent == PUBLIC_TEST_IMAGE_DIRECTORY


def _png_scanline_sizes(
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    interlace: int,
) -> list[int] | None:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if color_type not in channels or bit_depth not in valid_depths[color_type]:
        return None
    if interlace == 0:
        passes = ((0, 0, 1, 1),)
    elif interlace == 1:
        passes = (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
    else:
        return None

    sizes: list[int] = []
    bits_per_pixel = channels[color_type] * bit_depth
    for x_start, y_start, x_step, y_step in passes:
        pass_width = max(0, (width - x_start + x_step - 1) // x_step)
        pass_height = max(0, (height - y_start + y_step - 1) // y_step)
        if not pass_width or not pass_height:
            continue
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        sizes.extend([row_bytes] * pass_height)
    return sizes


def _is_valid_png(raw_content: bytes) -> bool:
    if not raw_content.startswith(PNG_SIGNATURE):
        return False

    offset = len(PNG_SIGNATURE)
    header: tuple[int, int, int, int, int, int, int] | None = None
    compressed = bytearray()
    saw_image_data = False
    image_data_closed = False
    saw_palette = False
    saw_end = False
    chunk_index = 0
    while offset < len(raw_content):
        if len(raw_content) - offset < 12:
            return False
        length = struct.unpack(">I", raw_content[offset : offset + 4])[0]
        chunk_type = raw_content[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(raw_content) or not re.fullmatch(rb"[A-Za-z]{4}", chunk_type):
            return False
        data = raw_content[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", raw_content[offset + 8 + length : chunk_end])[0]
        if zlib.crc32(chunk_type + data) & 0xFFFFFFFF != stored_crc:
            return False
        if chunk_index == 0 and chunk_type != b"IHDR":
            return False
        if chunk_type == b"IHDR":
            if header is not None or length != 13:
                return False
            header = struct.unpack(">IIBBBBB", data)
            width, height, _, _, compression, filter_method, _ = header
            if not width or not height or compression != 0 or filter_method != 0:
                return False
        elif chunk_type == b"PLTE":
            if saw_image_data or saw_palette or not length or length % 3 or length > 768:
                return False
            saw_palette = True
        elif chunk_type == b"IDAT":
            if header is None or image_data_closed:
                return False
            saw_image_data = True
            compressed.extend(data)
        elif chunk_type == b"IEND":
            if length or not saw_image_data or chunk_end != len(raw_content):
                return False
            saw_end = True
            offset = chunk_end
            break
        else:
            if saw_image_data:
                image_data_closed = True
            if chunk_type[0] & 0x20 == 0:
                return False
        offset = chunk_end
        chunk_index += 1

    if header is None or not saw_end:
        return False
    width, height, bit_depth, color_type, _, _, interlace = header
    if color_type == 3 and not saw_palette:
        return False
    scanline_sizes = _png_scanline_sizes(width, height, bit_depth, color_type, interlace)
    if scanline_sizes is None:
        return False
    expected_size = sum(size + 1 for size in scanline_sizes)
    if expected_size > 100_000_000:
        return False
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(bytes(compressed), expected_size + 1)
    except zlib.error:
        return False
    if (
        len(decoded) != expected_size
        or not decompressor.eof
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        return False
    decoded_offset = 0
    for row_bytes in scanline_sizes:
        if decoded[decoded_offset] > 4:
            return False
        decoded_offset += row_bytes + 1
    return True


def _is_credential_placeholder(raw_value: str) -> bool:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    lowered = value.lower()
    return (
        not value
        or lowered in {"none", "null", "placeholder", "replace_me", "change_me", "changeme"}
        or lowered.endswith("_sentinel")
        or (value.startswith("<") and value.endswith(">"))
        or ENV_REFERENCE_PATTERN.fullmatch(value) is not None
    )


def _credential_name_tokens(name: str) -> set[str]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return {token for token in re.split(r"[^A-Za-z0-9]+", separated.lower()) if token}


def _is_credential_name(name: str) -> bool:
    tokens = _credential_name_tokens(name)
    ordered_tokens = [
        token
        for token in re.split(
            r"[^A-Za-z0-9]+",
            re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower(),
        )
        if token
    ]
    vocabulary_roles = {"bos", "cls", "eos", "mask", "pad", "sep", "unk"}
    if ordered_tokens and ordered_tokens[-1] in {"field", "name"}:
        return False
    if len(ordered_tokens) == 2 and ordered_tokens[1] == "token":
        if ordered_tokens[0] in vocabulary_roles:
            return False
    sensitive_endings = {"token", "secret", "password", "credential", "credentials"}
    if len(ordered_tokens) > 1 and ordered_tokens[0] in {"num", "number", "count"}:
        return False
    if len(ordered_tokens) == 1 and ordered_tokens[0] in sensitive_endings:
        return True
    if len(ordered_tokens) > 1 and ordered_tokens[-1] in sensitive_endings:
        return True
    return "key" in tokens and bool(tokens & {"api", "access", "private", "signing"})


def _json_has_credential_literal(path: Path, content: str) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False

    def visit(value: object) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and _is_credential_name(key):
                    if item is not None and (
                        not isinstance(item, str) or not _is_credential_placeholder(item)
                    ):
                        return True
                if visit(item):
                    return True
        elif isinstance(value, list):
            return any(visit(item) for item in value)
        return False

    return visit(parsed)


@dataclass
class _PythonFlowState:
    aliases: dict[str, str] = field(default_factory=dict)
    literals: dict[str, object] = field(default_factory=dict)
    strings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mappings: dict[str, dict[object, object]] = field(default_factory=dict)

    def clone(self) -> _PythonFlowState:
        return _PythonFlowState(
            aliases=dict(self.aliases),
            literals=dict(self.literals),
            strings=dict(self.strings),
            mappings={name: dict(value) for name, value in self.mappings.items()},
        )

    def forget(self, name: str) -> None:
        self.aliases.pop(name, None)
        self.literals.pop(name, None)
        self.strings.pop(name, None)
        self.mappings.pop(name, None)

    def replace(self, other: _PythonFlowState) -> None:
        self.aliases = other.aliases
        self.literals = other.literals
        self.strings = other.strings
        self.mappings = other.mappings


def _python_flow_literal(
    node: ast.AST | None,
    state: _PythonFlowState,
    seen: frozenset[str] = frozenset(),
) -> object:
    if isinstance(node, ast.Name):
        if node.id in seen:
            return NON_LITERAL
        return state.literals.get(node.id, NON_LITERAL)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _python_flow_literal(node.left, state, seen)
        right = _python_flow_literal(node.right, state, seen)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value_node in node.values:
            if isinstance(value_node, ast.FormattedValue):
                value = _python_flow_literal(value_node.value, state, seen)
            else:
                value = _python_flow_literal(value_node, state, seen)
            if value is NON_LITERAL:
                return NON_LITERAL
            parts.append(str(value))
        return "".join(parts)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id == "str":
            value = _python_flow_literal(node.args[0], state, seen)
            if value is not NON_LITERAL:
                return str(value)
        if isinstance(node.func, ast.Attribute):
            owner = _python_flow_literal(node.func.value, state, seen)
            if (
                isinstance(owner, str)
                and node.func.attr in {"strip", "lstrip", "rstrip"}
            ):
                characters = _python_flow_literal(node.args[0], state, seen)
                if isinstance(characters, str):
                    return getattr(owner, node.func.attr)(characters)
            if (
                isinstance(owner, str)
                and node.func.attr == "join"
                and isinstance(node.args[0], (ast.List, ast.Tuple))
            ):
                items = [_python_flow_literal(item, state, seen) for item in node.args[0].elts]
                if all(isinstance(item, str) for item in items):
                    return owner.join(items)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and not node.args
        and not node.keywords
        and node.func.attr in {"strip", "lstrip", "rstrip"}
    ):
        owner = _python_flow_literal(node.func.value, state, seen)
        if isinstance(owner, str):
            return getattr(owner, node.func.attr)()
    return NON_LITERAL


def _python_flow_qualified_name(
    node: ast.AST | None,
    state: _PythonFlowState,
) -> str | None:
    if isinstance(node, ast.Name):
        return state.aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _python_flow_qualified_name(node.value, state)
        return f"{owner}.{node.attr}" if owner is not None else None
    if isinstance(node, ast.Subscript):
        return _python_flow_qualified_name(node.value, state)
    if (
        isinstance(node, ast.Call)
        and _python_flow_qualified_name(node.func, state) == "getattr"
        and len(node.args) >= 2
    ):
        owner = _python_flow_qualified_name(node.args[0], state)
        attribute = _python_flow_literal(node.args[1], state)
        if owner is not None and isinstance(attribute, str):
            return f"{owner}.{attribute}"
    return None


def _python_flow_string_values(
    node: ast.AST,
    state: _PythonFlowState,
) -> list[str]:
    values: list[str] = []

    def add(value: str) -> None:
        if value not in values:
            values.append(value)

    def visit(child: ast.AST) -> None:
        literal = _python_flow_literal(child, state)
        if isinstance(literal, str):
            add(literal)
        if isinstance(child, ast.Name):
            for value in state.strings.get(child.id, ()):
                add(value)
            return
        for nested in ast.iter_child_nodes(child):
            visit(nested)

    visit(node)
    return values


def _python_flow_mapping(
    node: ast.AST,
    state: _PythonFlowState,
) -> dict[object, object] | None:
    if isinstance(node, ast.Name):
        mapping = state.mappings.get(node.id)
        return dict(mapping) if mapping is not None else None
    if isinstance(node, ast.Dict):
        mapping: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                nested = _python_flow_mapping(value_node, state)
                if nested is not None:
                    mapping.update(nested)
                continue
            key = _python_flow_literal(key_node, state)
            if key is NON_LITERAL:
                continue
            mapping[key] = _python_flow_literal(value_node, state)
        return mapping
    if isinstance(node, ast.Call) and _python_flow_qualified_name(node.func, state) == "dict":
        mapping: dict[object, object] = {}
        if node.args:
            nested = _python_flow_mapping(node.args[0], state)
            if nested is not None:
                mapping.update(nested)
        for keyword in node.keywords:
            if keyword.arg is None:
                nested = _python_flow_mapping(keyword.value, state)
                if nested is not None:
                    mapping.update(nested)
            else:
                mapping[keyword.arg] = _python_flow_literal(keyword.value, state)
        return mapping
    return None


def _python_merge_flow_states(states: list[_PythonFlowState]) -> _PythonFlowState:
    if not states:
        return _PythonFlowState()

    def common(mapping_name: str) -> dict:
        mappings = [getattr(state, mapping_name) for state in states]
        return {
            key: value
            for key, value in mappings[0].items()
            if all(key in mapping and mapping[key] == value for mapping in mappings[1:])
        }

    return _PythonFlowState(
        aliases=common("aliases"),
        literals=common("literals"),
        strings=common("strings"),
        mappings=common("mappings"),
    )


def _is_unofficial_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and parsed.hostname not in {
        "api.openai.com",
        "openai.com",
        "www.openai.com",
    }


class _PythonPolicyAnalyzer:
    def __init__(self) -> None:
        self.codes: set[str] = set()

    def analyze(self, tree: ast.Module) -> set[str]:
        self._analyze_block(tree.body, _PythonFlowState())
        return self.codes

    def _analyze_block(
        self,
        statements: list[ast.stmt],
        state: _PythonFlowState,
    ) -> None:
        for statement in statements:
            self._analyze_statement(statement, state)

    def _analyze_statement(self, statement: ast.stmt, state: _PythonFlowState) -> None:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            self._apply_import(statement, state)
            return
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for expression in (
                list(statement.decorator_list)
                + list(statement.args.defaults)
                + [default for default in statement.args.kw_defaults if default is not None]
            ):
                self._analyze_expression(expression, state)
            child = state.clone()
            arguments = (
                list(statement.args.posonlyargs)
                + list(statement.args.args)
                + list(statement.args.kwonlyargs)
            )
            if statement.args.vararg is not None:
                arguments.append(statement.args.vararg)
            if statement.args.kwarg is not None:
                arguments.append(statement.args.kwarg)
            for argument in arguments:
                child.forget(argument.arg)
            self._analyze_block(statement.body, child)
            state.forget(statement.name)
            return
        if isinstance(statement, ast.ClassDef):
            for expression in list(statement.decorator_list) + list(statement.bases):
                self._analyze_expression(expression, state)
            for keyword in statement.keywords:
                self._analyze_expression(keyword.value, state)
            self._analyze_block(statement.body, state.clone())
            state.forget(statement.name)
            return
        if isinstance(statement, ast.If):
            self._analyze_expression(statement.test, state)
            branches: list[_PythonFlowState] = []
            body_state = state.clone()
            self._analyze_block(statement.body, body_state)
            branches.append(body_state)
            else_state = state.clone()
            self._analyze_block(statement.orelse, else_state)
            branches.append(else_state)
            state.replace(_python_merge_flow_states(branches))
            return
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                self._analyze_expression(statement.iter, state)
                loop_state = state.clone()
                self._forget_target(statement.target, loop_state)
            else:
                self._analyze_expression(statement.test, state)
                loop_state = state.clone()
            self._analyze_block(statement.body, loop_state)
            else_state = state.clone()
            self._analyze_block(statement.orelse, else_state)
            state.replace(_python_merge_flow_states([state, loop_state, else_state]))
            return
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                self._analyze_expression(item.context_expr, state)
                if item.optional_vars is not None:
                    self._forget_target(item.optional_vars, state)
            self._analyze_block(statement.body, state)
            return
        if isinstance(statement, ast.Try):
            branches: list[_PythonFlowState] = []
            body_state = state.clone()
            self._analyze_block(statement.body, body_state)
            self._analyze_block(statement.orelse, body_state)
            branches.append(body_state)
            for handler in statement.handlers:
                handler_state = state.clone()
                if handler.type is not None:
                    self._analyze_expression(handler.type, handler_state)
                if handler.name is not None:
                    handler_state.forget(handler.name)
                self._analyze_block(handler.body, handler_state)
                branches.append(handler_state)
            merged = _python_merge_flow_states(branches)
            self._analyze_block(statement.finalbody, merged)
            state.replace(merged)
            return
        if isinstance(statement, ast.Assign):
            self._analyze_expression(statement.value, state)
            for target in statement.targets:
                self._check_credential_assignment(target, statement.value, state)
                self._check_assignment_target(target, state)
                self._bind_target(target, statement.value, state)
            return
        if isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self._analyze_expression(statement.value, state)
                self._check_credential_assignment(statement.target, statement.value, state)
                self._check_assignment_target(statement.target, state)
                self._bind_target(statement.target, statement.value, state)
            return
        if isinstance(statement, ast.AugAssign):
            self._analyze_expression(statement.value, state)
            self._check_assignment_target(statement.target, state)
            self._forget_target(statement.target, state)
            return
        if isinstance(statement, ast.Delete):
            for target in statement.targets:
                self._delete_target(target, state)
            return
        self._analyze_expression(statement, state)

    def _apply_import(
        self,
        statement: ast.Import | ast.ImportFrom,
        state: _PythonFlowState,
    ) -> None:
        if isinstance(statement, ast.Import):
            for imported in statement.names:
                if imported.name.split(".", maxsplit=1)[0].lower() == LEGACY_RUNTIME_ROOT:
                    self.codes.add("external-runtime-import")
                local_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                state.forget(local_name)
                state.aliases[local_name] = imported.name if imported.asname else local_name
            return
        if statement.module and statement.module.split(".", maxsplit=1)[0].lower() == LEGACY_RUNTIME_ROOT:
            self.codes.add("external-runtime-import")
        if statement.module is None:
            return
        for imported in statement.names:
            if imported.name == "*":
                continue
            local_name = imported.asname or imported.name
            state.forget(local_name)
            state.aliases[local_name] = f"{statement.module}.{imported.name}"

    def _analyze_expression(self, expression: ast.AST, state: _PythonFlowState) -> None:
        for node in ast.walk(expression):
            if isinstance(node, ast.Subscript) and self._is_removed_environment_target(node, state):
                self.codes.add("source-path-injection")
            if isinstance(node, ast.Dict):
                for key_node, value_node in zip(node.keys, node.values):
                    key = _python_flow_literal(key_node, state)
                    if isinstance(key, str) and _is_credential_name(key):
                        self._check_credential_value(value_node, state)
            if not isinstance(node, ast.Call):
                continue
            self._check_call(node, state)
            self._apply_mapping_call(node, state)

    def _check_call(self, node: ast.Call, state: _PythonFlowState) -> None:
        call_name = _python_flow_qualified_name(node.func, state)
        if call_name in {
            "__import__",
            "importlib.import_module",
            "importlib.util.module_from_spec",
            "importlib.util.spec_from_file_location",
        } or (call_name is not None and call_name.endswith(".exec_module")):
            self.codes.add("dynamic-import")

        if call_name is not None and call_name.startswith("sys.path."):
            if call_name.rsplit(".", maxsplit=1)[-1] in {
                "append",
                "clear",
                "extend",
                "insert",
                "pop",
                "remove",
                "reverse",
                "sort",
            }:
                self.codes.add("source-path-injection")
        if call_name == "setattr" and len(node.args) >= 2:
            owner = _python_flow_qualified_name(node.args[0], state)
            attribute = _python_flow_literal(node.args[1], state)
            if owner == "sys" and attribute == "path":
                self.codes.add("source-path-injection")
            if owner in {"openai", "OpenAI"} and attribute in {
                "api_" + "base",
                "base_" + "url",
            }:
                self.codes.add("custom-openai-endpoint")

        if call_name in {"os.putenv", "os.environ.__setitem__", "os.environb.__setitem__"}:
            if node.args and _python_flow_literal(node.args[0], state) in REMOVED_ENVIRONMENT_NAMES:
                self.codes.add("source-path-injection")
        if call_name in {
            "os.environ.get",
            "os.environ.setdefault",
            "os.environb.get",
            "os.environb.setdefault",
            "os.getenv",
            "os.getenvb",
        }:
            if node.args and _python_flow_literal(node.args[0], state) in REMOVED_ENVIRONMENT_NAMES:
                self.codes.add("source-path-injection")
        if call_name in {"os.environ.update", "os.environb.update"}:
            mapping = _python_flow_mapping(node.args[0], state) if node.args else {}
            if mapping is not None and set(mapping) & set(REMOVED_ENVIRONMENT_NAMES):
                self.codes.add("source-path-injection")

        if call_name is not None and call_name.endswith(".add_argument"):
            if any(
                _python_flow_literal(argument, state) in REMOVED_CLI_OPTIONS
                for argument in node.args
            ):
                self.codes.add("removed-source-interface")

        if call_name in {
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.run",
        }:
            for keyword in node.keywords:
                if keyword.arg != "env":
                    continue
                mapping = _python_flow_mapping(keyword.value, state)
                if mapping is not None and set(mapping) & set(REMOVED_ENVIRONMENT_NAMES):
                    self.codes.add("source-path-injection")
            strings = _python_flow_string_values(node, state)
            compact = "".join(strings).lower()
            has_external_module = any(
                re.search(r"(?:^|[.])train$", value.lower())
                and not value.lower().startswith(("midline_agent.", "scripts."))
                for value in strings
            )
            if "train.py" in compact or has_external_module:
                self.codes.add("external-training-delegation")

        endpoint_keys = {"api_" + "base", "base_" + "url", "openai_base_" + "url"}
        is_openai_constructor = (
            call_name is not None
            and call_name.rsplit(".", maxsplit=1)[-1].lower() in {"asyncopenai", "openai"}
        )
        if is_openai_constructor:
            for keyword in node.keywords:
                if keyword.arg in endpoint_keys:
                    self.codes.add("custom-openai-endpoint")
                elif keyword.arg is None:
                    mapping = _python_flow_mapping(keyword.value, state)
                    if mapping is not None and set(mapping) & endpoint_keys:
                        self.codes.add("custom-openai-endpoint")
            if any(_is_unofficial_url(value) for value in _python_flow_string_values(node, state)):
                self.codes.add("custom-openai-endpoint")

        for keyword in node.keywords:
            value = _python_flow_literal(keyword.value, state)
            if keyword.arg is not None and _is_credential_name(keyword.arg):
                self._check_credential_value(keyword.value, state)
            if (
                keyword.arg in {"provider", "relay", "service"}
                and isinstance(value, str)
                and value.lower() in RELAY_SERVICE_NAMES
            ):
                self.codes.add("relay-service")

    def _check_credential_assignment(
        self,
        target: ast.AST,
        value: ast.AST,
        state: _PythonFlowState,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                for nested_target, nested_value in zip(target.elts, value.elts):
                    self._check_credential_assignment(nested_target, nested_value, state)
            return
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        elif isinstance(target, ast.Subscript):
            key = _python_flow_literal(target.slice, state)
            name = key if isinstance(key, str) else None
        else:
            name = None
        if name is not None and _is_credential_name(name):
            self._check_credential_value(value, state)

    def _check_credential_value(
        self,
        value_node: ast.AST,
        state: _PythonFlowState,
    ) -> None:
        value = _python_flow_literal(value_node, state)
        if value is NON_LITERAL or value is None:
            return
        if not isinstance(value, str) or not _is_credential_placeholder(value):
            self.codes.add("embedded-secret")

    def _apply_mapping_call(self, node: ast.Call, state: _PythonFlowState) -> None:
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            return
        name = node.func.value.id
        if name not in state.mappings:
            return
        mapping = dict(state.mappings[name])
        if node.func.attr == "update" and node.args:
            updated = _python_flow_mapping(node.args[0], state)
            if updated is not None:
                mapping.update(updated)
        elif node.func.attr in {"setdefault", "__setitem__"} and node.args:
            key = _python_flow_literal(node.args[0], state)
            if key is not NON_LITERAL:
                value = (
                    _python_flow_literal(node.args[1], state)
                    if len(node.args) > 1
                    else NON_LITERAL
                )
                mapping.setdefault(key, value)
        elif node.func.attr == "clear" and not node.args:
            mapping.clear()
        else:
            return
        state.mappings[name] = mapping

    def _check_assignment_target(self, target: ast.AST, state: _PythonFlowState) -> None:
        if not isinstance(target, ast.Name) and _python_flow_qualified_name(target, state) == "sys.path":
            self.codes.add("source-path-injection")
        if self._is_removed_environment_target(target, state):
            self.codes.add("source-path-injection")
        if isinstance(target, (ast.Tuple, ast.List)):
            for nested in target.elts:
                self._check_assignment_target(nested, state)

    def _is_removed_environment_target(
        self,
        target: ast.AST,
        state: _PythonFlowState,
    ) -> bool:
        if not isinstance(target, ast.Subscript):
            return False
        owner = _python_flow_qualified_name(target.value, state)
        key = _python_flow_literal(target.slice, state)
        return owner in {"os.environ", "os.environb"} and key in REMOVED_ENVIRONMENT_NAMES

    def _bind_target(
        self,
        target: ast.AST,
        value: ast.AST,
        state: _PythonFlowState,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                for nested_target, nested_value in zip(target.elts, value.elts):
                    self._bind_target(nested_target, nested_value, state)
            return
        if isinstance(target, ast.Name):
            target_name = target.id
            literal = _python_flow_literal(value, state)
            strings = tuple(_python_flow_string_values(value, state))
            mapping = _python_flow_mapping(value, state)
            qualified = _python_flow_qualified_name(value, state)
            state.forget(target_name)
            if literal is not NON_LITERAL:
                state.literals[target_name] = literal
            if strings:
                state.strings[target_name] = strings
            if mapping is not None:
                state.mappings[target_name] = mapping
            if qualified is not None and qualified != target_name:
                state.aliases[target_name] = qualified
            if (
                target_name.lower()
                in {"provider", "relay", "relay_name", "service", "service_name"}
                and isinstance(literal, str)
                and literal.lower() in RELAY_SERVICE_NAMES
            ):
                self.codes.add("relay-service")
            return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            mapping = state.mappings.get(name)
            key = _python_flow_literal(target.slice, state)
            if mapping is not None and key is not NON_LITERAL:
                updated = dict(mapping)
                updated[key] = _python_flow_literal(value, state)
                state.mappings[name] = updated

    def _forget_target(self, target: ast.AST, state: _PythonFlowState) -> None:
        if isinstance(target, ast.Name):
            state.forget(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for nested in target.elts:
                self._forget_target(nested, state)

    def _delete_target(self, target: ast.AST, state: _PythonFlowState) -> None:
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            mapping = state.mappings.get(target.value.id)
            key = _python_flow_literal(target.slice, state)
            if mapping is not None and key in mapping:
                updated = dict(mapping)
                updated.pop(key)
                state.mappings[target.value.id] = updated
            return
        self._forget_target(target, state)


def _python_policy_codes(content: str) -> set[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()
    return _PythonPolicyAnalyzer().analyze(tree)


def _matches_release_path(path: Path, release_path: str, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        relative = path.absolute().relative_to(root.resolve())
    except ValueError:
        return False
    return relative.as_posix() == release_path


def _is_root_test_path(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        relative = path.absolute().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == "tests"


def _line_has_prohibition_marker(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in PROHIBITION_MARKERS)


def _is_explicit_prohibition_line(lines: list[str], line_number: int) -> bool:
    line = lines[line_number]
    if _line_has_prohibition_marker(line):
        return True
    if line.startswith((" ", "\t")) and line_number > 0:
        if _line_has_prohibition_marker(lines[line_number - 1]) and not _is_imperative_legacy_instruction(line):
            return True
    if not line.lstrip().startswith(("- ", "* ")):
        return False
    previous = line_number - 1
    while previous >= 0:
        stripped = lines[previous].strip().lower()
        if not stripped or stripped.startswith(("- ", "* ")):
            previous -= 1
            continue
        return stripped == "the repository must not:"
    return False


def _is_approved_prohibition_line(
    path: Path,
    lines: list[str],
    line_number: int,
    root: Path | None,
) -> bool:
    if not any(_matches_release_path(path, item, root) for item in APPROVED_PROHIBITION_PATHS):
        return False
    return _is_explicit_prohibition_line(lines, line_number)


def _has_legacy_runtime_path(line: str) -> bool:
    return re.search(rf"\b{re.escape(LEGACY_RUNTIME_ROOT)}/", line, re.IGNORECASE) is not None


def _is_imperative_legacy_instruction(line: str) -> bool:
    return re.search(
        r"^[ \t]*(?:[-*][ \t]+)?(?:add|clone|configure|execute|export|import|install|load|pass|run|set|source|use)\b",
        line,
        re.IGNORECASE,
    ) is not None


def _is_shell_text(path: Path, content: str) -> bool:
    if path.suffix.lower() in SHELL_SUFFIXES:
        return True
    first_line = content.splitlines()[0] if content else ""
    return re.match(
        r"^#![^\r\n]*(?:/|[ \t])(?:ba|da|k|z)?sh(?:[ \t]|$)",
        first_line,
        re.IGNORECASE,
    ) is not None


def _text_policy_codes(path: Path, content: str, root: Path | None) -> set[str]:
    is_shell = _is_shell_text(path, content)
    if (
        path.suffix.lower() not in TEXT_INTERFACE_SUFFIXES
        and not path.name.startswith(".env")
        and not is_shell
    ):
        return set()

    codes: set[str] = set()
    lines = content.splitlines()
    for line_number, line in enumerate(lines):
        has_removed_interface = any(name in line for name in REMOVED_ENVIRONMENT_NAMES) or any(
            option in line for option in REMOVED_CLI_OPTIONS
        )
        if has_removed_interface and not _is_approved_prohibition_line(
            path, lines, line_number, root
        ):
            codes.add("removed-source-interface")

        if _has_legacy_runtime_path(line):
            is_approved_reference = _matches_release_path(
                path, APPROVED_PROVENANCE_PATH, root
            ) or _matches_release_path(path, APPROVED_DESIGN_PATH, root)
            if not is_approved_reference or _is_imperative_legacy_instruction(line):
                codes.add("external-runtime-import")

    if is_shell:
        if any(re.match(r"^[ \t]*(?:source|\.)[ \t]+", line) for line in lines):
            codes.add("source-path-injection")
    return codes


def _has_private_path(path: Path, content: str, root: Path | None) -> bool:
    if path.name == ".gitignore":
        return False
    lines = content.splitlines()
    for line_number, line in enumerate(lines):
        if line.lstrip().startswith("#!"):
            continue
        without_urls = URL_PATTERN.sub("", line)
        if WINDOWS_PRIVATE_PATH_PATTERN.search(without_urls):
            return True
        for match in POSIX_ABSOLUTE_PATH_PATTERN.finditer(without_urls):
            candidate = match.group(0)
            if any(
                candidate == prefix.rstrip("/") or candidate.startswith(prefix)
                for prefix in SAFE_POSIX_PATH_PREFIXES
            ):
                continue
            if _is_root_test_path(path, root) and any(
                candidate == prefix.rstrip("/") or candidate.startswith(prefix)
                for prefix in TEST_FIXTURE_POSIX_PREFIXES
            ):
                continue
            if _is_approved_prohibition_line(path, lines, line_number, root):
                continue
            return True
    return False


def _content_violations(path: Path, content: str, root: Path | None) -> list[Violation]:
    violations: list[Violation] = []
    lowered = content.lower()
    is_audit_fixture = _is_policy_content(path, root)
    is_python = path.suffix.lower() == ".py"
    has_assigned_credential = False
    if not is_python and not is_audit_fixture:
        has_assigned_credential = any(
            _is_credential_name(match.group("name"))
            and not _is_credential_placeholder(match.group("value"))
            for match in ASSIGNMENT_PATTERN.finditer(content)
        )
    if (
        OPENAI_KEY_PATTERN.search(content)
        or any(pattern.search(content) for pattern in PROVIDER_SECRET_PATTERNS)
        or PRIVATE_KEY_PATTERN.search(content)
        or has_assigned_credential
        or _json_has_credential_literal(path, content)
    ):
        violations.append(Violation(path, "embedded-secret", "embedded credential detected"))
    if _has_private_path(path, content, root):
        violations.append(Violation(path, "private-path", "private absolute path detected"))
    if HAN_PATTERN.search(content):
        violations.append(Violation(path, "han-character", "Han character detected"))
    elif any(ord(character) > 127 for character in content):
        violations.append(Violation(path, "non-ascii", "non-ASCII text detected"))
    if any(term.lower() in lowered for term in CUSTOM_ENDPOINT_TERMS):
        violations.append(Violation(path, "custom-openai-endpoint", "custom OpenAI endpoint configuration detected"))
    if any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in RELAY_SERVICE_NAMES):
        violations.append(Violation(path, "relay-service", "OpenAI relay service name detected"))
    if not _is_policy_content(path, root) and re.search(r"\bbaseline\b", lowered):
        violations.append(Violation(path, "prohibited-term", "prohibited release term detected"))
    details = {
        "custom-openai-endpoint": "custom OpenAI endpoint configuration detected",
        "dynamic-import": "dynamic external runtime import detected",
        "embedded-secret": "embedded credential detected",
        "external-runtime-import": "external runtime import or instruction detected",
        "external-training-delegation": "delegation to an external training entrypoint detected",
        "relay-service": "OpenAI relay service name detected",
        "removed-source-interface": "removed external source interface detected",
        "source-path-injection": "Python source path injection detected",
    }
    existing_codes = {violation.code for violation in violations}
    policy_codes = _text_policy_codes(path, content, root)
    if is_python and not is_audit_fixture:
        policy_codes.update(_python_policy_codes(content))
    for code in sorted(policy_codes - existing_codes):
        violations.append(Violation(path, code, details[code]))
    return violations


def audit_paths(paths: Iterable[Path | CandidateSource], *, root: Path | None = None) -> list[Violation]:
    """Return public-release violations for a supplied candidate file list."""
    candidates = list(paths)
    audit_root = root.resolve() if root is not None else None
    if not candidates:
        return [Violation(Path("."), "empty-candidate", "release candidate contains no files")]

    violations: list[Violation] = []
    for candidate in candidates:
        source = candidate if isinstance(candidate, CandidateSource) else None
        path = source.path if source is not None else Path(candidate)
        if source.is_symlink if source is not None else path.is_symlink():
            violations.append(Violation(path, "symlink", "symlinks are not allowed in a release candidate"))
            continue
        path_is_forbidden = _has_forbidden_path(path, audit_root)
        if source is None and not path.exists():
            if path_is_forbidden:
                violations.append(Violation(path, "forbidden-path", "forbidden file or directory"))
            else:
                violations.append(Violation(path, "missing-file", "candidate file does not exist"))
            continue
        if source is None and not path.is_file():
            violations.append(Violation(path, "non-file", "candidate entry is not a regular file"))
            continue
        raw_content = source.content if source is not None else path.read_bytes()
        public_test_png_path = _is_public_test_png_path(path, audit_root)
        if path_is_forbidden and not public_test_png_path:
            violations.append(Violation(path, "forbidden-path", "forbidden file or directory"))
        if public_test_png_path:
            if not _is_valid_png(raw_content):
                violations.append(Violation(path, "binary-file", "attributed test image is not a valid PNG"))
            continue
        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(Violation(path, "binary-file", "binary files are not allowed"))
            continue
        if "\x00" in content:
            violations.append(Violation(path, "binary-file", "binary files are not allowed"))
            continue
        violations.extend(_content_violations(path, content, audit_root))
    return violations


def _boundary_violation(path: Path, root: Path) -> Violation | None:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        return Violation(path, "outside-root", "candidate path is outside the audited root")

    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return Violation(current, "symlink", "candidate path contains a symlink component")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return Violation(path, "outside-root", "candidate path resolves outside the audited root")
    return None


def _manifest_paths(manifest: Path, root: Path) -> tuple[list[Path], list[Violation]]:
    if not manifest.is_file():
        return [], [Violation(manifest, "missing-manifest", "release manifest does not exist")]

    paths: list[Path] = []
    violations: list[Violation] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        candidate = Path(entry)
        if candidate.is_absolute() or entry == "." or ".." in candidate.parts:
            violations.append(
                Violation(
                    manifest,
                    "invalid-manifest-path",
                    f"line {line_number} is not a repository-relative file path: {entry}",
                )
            )
            continue
        candidate_path = root / candidate
        boundary_violation = _boundary_violation(candidate_path, root)
        if boundary_violation is not None:
            violations.append(boundary_violation)
            continue
        paths.append(candidate_path)
    return paths, violations


def _candidate_paths(candidate_root: Path, root: Path) -> tuple[list[Path], list[Violation]]:
    boundary_violation = _boundary_violation(candidate_root, root)
    if boundary_violation is not None:
        return [], [boundary_violation]
    if candidate_root.is_symlink() or not candidate_root.exists() or candidate_root.is_file():
        return [candidate_root], []
    paths = sorted(
        (path for path in candidate_root.rglob("*") if path.is_symlink() or path.is_file()),
        key=lambda path: path.as_posix(),
    )
    return paths, []


def _index_sources(repo_root: Path, names: list[str]) -> list[CandidateSource]:
    sources: list[CandidateSource] = []
    for name in names:
        staged = subprocess.run(
            ["git", "ls-files", "--stage", "-z", "--", name],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        metadata = staged.split(b"\t", maxsplit=1)[0].split()
        mode, object_id = metadata[0].decode("ascii"), metadata[1].decode("ascii")
        content = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        sources.append(CandidateSource(repo_root / name, content, is_symlink=mode == "120000"))
    return sources


def _staged_sources(repo_root: Path) -> list[CandidateSource]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    return _index_sources(repo_root, names)


def _tracked_sources(repo_root: Path) -> list[CandidateSource]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    return _index_sources(repo_root, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="audit files listed in a release manifest")
    mode.add_argument("--candidate", type=Path, help="audit every file in a candidate tree")
    mode.add_argument("--staged", action="store_true", help="audit Git staged files")
    mode.add_argument("--tracked-only", action="store_true", help="audit tracked files")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args()
    root = args.root.resolve()

    initial_violations: list[Violation] = []
    if args.manifest is not None:
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        paths, initial_violations = _manifest_paths(manifest, root)
    elif args.candidate is not None:
        candidate = args.candidate if args.candidate.is_absolute() else root / args.candidate
        paths, initial_violations = _candidate_paths(candidate, root)
    elif args.staged:
        paths = _staged_sources(root)
    else:
        paths = _tracked_sources(root)

    violations = [*initial_violations, *audit_paths(paths, root=root)]
    count_label = "file" if len(paths) == 1 else "files"
    print(f"Scanned {len(paths)} candidate {count_label}.")
    for violation in violations:
        print(f"{violation.path}: {violation.code}: {violation.detail}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
