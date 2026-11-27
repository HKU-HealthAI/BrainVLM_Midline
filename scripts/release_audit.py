#!/usr/bin/env python3
"""Reject files that must not be included in a public source release."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass
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
RESULT_ARTIFACT_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv", ".xls", ".xlsx"}
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
PRIVATE_PATH_PATTERN = re.compile(
    rf"(?:/(?:{'user' + 'home'}|{'ho' + 'me'}|{'Us' + 'ers'})/[^\s'\"]+|[A-Za-z]:\\{'Us' + 'ers'}\\[^\s'\"]+)",
    re.IGNORECASE,
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
TEXT_INTERFACE_SUFFIXES = {".env", ".json", ".md", ".rst", ".sbatch", ".sh", ".toml", ".txt", ".yaml", ".yml"}
SHELL_SUFFIXES = {".sbatch", ".sh"}
LEGACY_RUNTIME_ROOT = "mini" + "gpt4"
PROHIBITION_MARKERS = (
    "absent",
    "delete",
    "do not",
    "does not",
    "excluded",
    "forbid",
    "must not",
    "neither",
    "no executable",
    "not accept",
    "not committed",
    "reject",
    "remove",
    "scan",
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


def _is_policy_content(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return any(normalized.endswith(suffix) for suffix in POLICY_CONTENT_PATHS)


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
    return path.suffix.lower() in RESULT_ARTIFACT_SUFFIXES and bool(FORBIDDEN_ARTIFACT_NAME.search(path.stem))


def _is_public_test_png_path(path: Path, root: Path | None) -> bool:
    if root is None or path.suffix.lower() != ".png":
        return False
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        return False
    return relative.parent == PUBLIC_TEST_IMAGE_DIRECTORY


def _is_credential_placeholder(raw_value: str) -> bool:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    lowered = value.lower()
    return (
        not value
        or lowered in {"none", "null", "placeholder", "replace_me", "change_me", "changeme"}
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


def _python_literal_value(node: ast.AST | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _python_literal_value(node.left)
        right = _python_literal_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return NON_LITERAL


def _python_assignment_pairs(target: ast.AST, value: ast.AST | None) -> list[tuple[ast.AST, ast.AST | None]]:
    if isinstance(target, (ast.List, ast.Tuple)) and isinstance(value, (ast.List, ast.Tuple)):
        if len(target.elts) == len(value.elts):
            pairs: list[tuple[ast.AST, ast.AST | None]] = []
            for nested_target, nested_value in zip(target.elts, value.elts):
                pairs.extend(_python_assignment_pairs(nested_target, nested_value))
            return pairs
    return [(target, value)]


def _python_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                aliases[local_name] = imported.name if imported.asname else local_name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                if imported.name == "*":
                    continue
                local_name = imported.asname or imported.name
                aliases[local_name] = f"{node.module}.{imported.name}"
    return aliases


def _python_qualified_name(node: ast.AST | None, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _python_qualified_name(node.value, aliases)
        return f"{owner}.{node.attr}" if owner is not None else None
    if isinstance(node, ast.Subscript):
        return _python_qualified_name(node.value, aliases)
    if isinstance(node, ast.Call) and _python_qualified_name(node.func, aliases) == "getattr":
        if len(node.args) >= 2:
            owner = _python_qualified_name(node.args[0], aliases)
            attribute = _python_literal_value(node.args[1])
            if owner is not None and isinstance(attribute, str):
                return f"{owner}.{attribute}"
    return None


def _python_string_values(node: ast.AST) -> list[str]:
    children = sorted(
        ast.walk(node),
        key=lambda child: (
            getattr(child, "lineno", -1),
            getattr(child, "col_offset", -1),
            getattr(child, "end_col_offset", -1),
        ),
    )
    values = [child.value for child in children if isinstance(child, ast.Constant) and isinstance(child.value, str)]
    for child in children:
        if isinstance(child, ast.Constant):
            continue
        value = _python_literal_value(child)
        if isinstance(value, str) and value not in values:
            values.append(value)
    return values


def _python_target_is_removed_environment(node: ast.AST, aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    owner = _python_qualified_name(node.value, aliases)
    key = _python_literal_value(node.slice)
    return owner in {"os.environ", "os.environb"} and key in REMOVED_ENVIRONMENT_NAMES


def _is_unofficial_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and parsed.hostname not in {
        "api.openai.com",
        "openai.com",
        "www.openai.com",
    }


def _python_policy_codes(content: str) -> set[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()

    aliases = _python_import_aliases(tree)
    codes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(imported.name.split(".", maxsplit=1)[0].lower() == LEGACY_RUNTIME_ROOT for imported in node.names):
                codes.add("external-runtime-import")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".", maxsplit=1)[0].lower() == LEGACY_RUNTIME_ROOT:
                codes.add("external-runtime-import")

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                qualified_target = _python_qualified_name(target, aliases)
                if qualified_target == "sys.path" or _python_target_is_removed_environment(target, aliases):
                    codes.add("source-path-injection")

        if not isinstance(node, ast.Call):
            continue
        call_name = _python_qualified_name(node.func, aliases)
        if call_name in {
            "__import__",
            "importlib.import_module",
            "importlib.util.module_from_spec",
            "importlib.util.spec_from_file_location",
        } or (call_name is not None and call_name.endswith(".exec_module")):
            codes.add("dynamic-import")

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
                codes.add("source-path-injection")
        if call_name == "setattr" and len(node.args) >= 2:
            owner = _python_qualified_name(node.args[0], aliases)
            attribute = _python_literal_value(node.args[1])
            if owner == "sys" and attribute == "path":
                codes.add("source-path-injection")
            if owner in {"openai", "OpenAI"} and attribute in {"api_" + "base", "base_" + "url"}:
                codes.add("custom-openai-endpoint")

        if call_name in {"os.putenv", "os.environ.__setitem__", "os.environb.__setitem__"}:
            if node.args and _python_literal_value(node.args[0]) in REMOVED_ENVIRONMENT_NAMES:
                codes.add("source-path-injection")
        if call_name in {"os.environ.update", "os.environb.update"}:
            if any(value in REMOVED_ENVIRONMENT_NAMES for value in _python_string_values(node)):
                codes.add("source-path-injection")

        if call_name is not None and call_name.endswith(".add_argument"):
            if any(_python_literal_value(argument) in REMOVED_CLI_OPTIONS for argument in node.args):
                codes.add("removed-source-interface")

        if call_name in {
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.run",
        }:
            strings = _python_string_values(node)
            compact = "".join(strings).lower()
            has_external_module = any(
                re.search(r"(?:^|[.])train$", value.lower())
                and not value.lower().startswith(("midline_agent.", "scripts."))
                for value in strings
            )
            if "train.py" in compact or has_external_module:
                codes.add("external-training-delegation")

        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys = {_python_literal_value(key) for key in child.keys}
                if keys & {"api_" + "base", "base_" + "url", "openai_base_" + "url"}:
                    codes.add("custom-openai-endpoint")
        if call_name is not None and call_name.rsplit(".", maxsplit=1)[-1].lower() in {
            "asyncopenai",
            "openai",
        }:
            if any(_is_unofficial_url(value) for value in _python_string_values(node)):
                codes.add("custom-openai-endpoint")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign):
            assignments = [(node.target, node.value)]
        else:
            assignments = []
        for target, value_node in assignments:
            target_name = _python_target_name(target)
            value = _python_literal_value(value_node)
            if (
                target_name is not None
                and target_name.lower() in {"provider", "relay", "relay_name", "service", "service_name"}
                and isinstance(value, str)
                and value.lower() in RELAY_SERVICE_NAMES
            ):
                codes.add("relay-service")
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                value = _python_literal_value(keyword.value)
                if (
                    keyword.arg in {"provider", "relay", "service"}
                    and isinstance(value, str)
                    and value.lower() in RELAY_SERVICE_NAMES
                ):
                    codes.add("relay-service")
    return codes


def _python_value_is_secret(node: ast.AST | None) -> bool:
    value = _python_literal_value(node)
    if value is NON_LITERAL or value is None:
        return False
    return not isinstance(value, str) or not _is_credential_placeholder(value)


def _python_target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        key = _python_literal_value(target.slice)
        return key if isinstance(key, str) else None
    return None


def _python_has_credential_literal(content: str) -> bool | None:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                key = _python_literal_value(key_node)
                if isinstance(key, str) and _is_credential_name(key):
                    if _python_value_is_secret(value_node):
                        return True
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is not None and _is_credential_name(keyword.arg):
                    if _python_value_is_secret(keyword.value):
                        return True
        if isinstance(node, ast.Assign):
            pairs = [
                pair
                for target in node.targets
                for pair in _python_assignment_pairs(target, node.value)
            ]
        elif isinstance(node, ast.AnnAssign):
            pairs = _python_assignment_pairs(node.target, node.value)
        else:
            continue
        for target, value in pairs:
            name = _python_target_name(target)
            if name is not None and _is_credential_name(name) and _python_value_is_secret(value):
                return True
    return False


def _matches_release_path(path: Path, release_path: str) -> bool:
    return path.as_posix().lower().endswith(release_path.lower())


def _is_approved_prohibition_line(path: Path, lines: list[str], line_number: int) -> bool:
    if not any(_matches_release_path(path, item) for item in APPROVED_PROHIBITION_PATHS):
        return False
    context = " ".join(lines[max(0, line_number - 8) : line_number + 1]).lower()
    return any(marker in context for marker in PROHIBITION_MARKERS)


def _has_legacy_runtime_path(line: str) -> bool:
    return re.search(rf"\b{re.escape(LEGACY_RUNTIME_ROOT)}/", line, re.IGNORECASE) is not None


def _is_imperative_legacy_instruction(line: str) -> bool:
    return re.search(
        r"\b(?:add|clone|execute|import|install|load|run|use)\b",
        line,
        re.IGNORECASE,
    ) is not None


def _text_policy_codes(path: Path, content: str) -> set[str]:
    if path.suffix.lower() not in TEXT_INTERFACE_SUFFIXES and not path.name.startswith(".env"):
        return set()

    codes: set[str] = set()
    lines = content.splitlines()
    for line_number, line in enumerate(lines):
        has_removed_interface = any(name in line for name in REMOVED_ENVIRONMENT_NAMES) or any(
            option in line for option in REMOVED_CLI_OPTIONS
        )
        if has_removed_interface and not _is_approved_prohibition_line(path, lines, line_number):
            codes.add("removed-source-interface")

        if _has_legacy_runtime_path(line):
            is_approved_reference = _matches_release_path(
                path, APPROVED_PROVENANCE_PATH
            ) or _matches_release_path(path, APPROVED_DESIGN_PATH)
            if not is_approved_reference or _is_imperative_legacy_instruction(line):
                codes.add("external-runtime-import")

    if path.suffix.lower() in SHELL_SUFFIXES:
        if any(re.match(r"^[ \t]*(?:source|\.)[ \t]+", line) for line in lines):
            codes.add("source-path-injection")
    return codes


def _content_violations(path: Path, content: str) -> list[Violation]:
    violations: list[Violation] = []
    lowered = content.lower()
    python_result = _python_has_credential_literal(content) if path.suffix.lower() == ".py" else None
    has_assigned_credential = python_result
    if has_assigned_credential is None:
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
    if PRIVATE_PATH_PATTERN.search(content):
        violations.append(Violation(path, "private-path", "private absolute path detected"))
    if HAN_PATTERN.search(content):
        violations.append(Violation(path, "han-character", "Han character detected"))
    elif any(ord(character) > 127 for character in content):
        violations.append(Violation(path, "non-ascii", "non-ASCII text detected"))
    if any(term.lower() in lowered for term in CUSTOM_ENDPOINT_TERMS):
        violations.append(Violation(path, "custom-openai-endpoint", "custom OpenAI endpoint configuration detected"))
    if any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in RELAY_SERVICE_NAMES):
        violations.append(Violation(path, "relay-service", "OpenAI relay service name detected"))
    if not _is_policy_content(path) and re.search(r"\bbaseline\b", lowered):
        violations.append(Violation(path, "prohibited-term", "prohibited release term detected"))
    details = {
        "custom-openai-endpoint": "custom OpenAI endpoint configuration detected",
        "dynamic-import": "dynamic external runtime import detected",
        "external-runtime-import": "external runtime import or instruction detected",
        "external-training-delegation": "delegation to an external training entrypoint detected",
        "relay-service": "OpenAI relay service name detected",
        "removed-source-interface": "removed external source interface detected",
        "source-path-injection": "Python source path injection detected",
    }
    existing_codes = {violation.code for violation in violations}
    policy_codes = _text_policy_codes(path, content)
    if path.suffix.lower() == ".py":
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
            if not raw_content.startswith(PNG_SIGNATURE):
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
        violations.extend(_content_violations(path, content))
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
