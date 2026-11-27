import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release_audit import audit_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "release_audit.py"


def _codes(paths: list[Path]) -> set[str]:
    return {violation.code for violation in audit_paths(paths)}


def _run_audit(*arguments: str, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), *arguments, "--root", str(root)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_audit_rejects_empty_path_collection() -> None:
    assert _codes([]) == {"empty-candidate"}


def test_manifest_mode_scans_only_listed_candidate_files(tmp_path: Path) -> None:
    included = tmp_path / "package.py"
    included.write_text("VALUE = 1\n", encoding="utf-8")
    unlisted = tmp_path / ".env"
    unlisted.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    manifest = tmp_path / "release-manifest.txt"
    manifest.write_text("package.py\n", encoding="utf-8")

    result = _run_audit("--manifest", str(manifest), root=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scanned 1 candidate file" in result.stdout


def test_release_manifest_matches_all_tracked_files() -> None:
    manifest = PROJECT_ROOT / "release-manifest.txt"
    listed = [line for line in manifest.read_text(encoding="ascii").splitlines() if line]
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = sorted(line for line in completed.stdout.splitlines() if line)

    assert listed == tracked


def test_manifest_mode_rejects_an_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "release-manifest.txt"
    manifest.write_text("# No release files yet.\n", encoding="utf-8")

    result = _run_audit("--manifest", str(manifest), root=tmp_path)

    assert result.returncode == 1
    assert "empty-candidate" in result.stdout


def test_manifest_mode_rejects_missing_and_parent_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "release-manifest.txt"
    manifest.write_text("missing.py\n../outside.py\n", encoding="utf-8")

    result = _run_audit("--manifest", str(manifest), root=tmp_path)

    assert result.returncode == 1
    assert "missing-file" in result.stdout
    assert "invalid-manifest-path" in result.stdout


@pytest.mark.parametrize("mode", ["manifest", "candidate"])
def test_candidate_sources_reject_intermediate_symlinks(tmp_path: Path, mode: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside_candidate = tmp_path / "outside" / "candidate"
    outside_candidate.mkdir(parents=True)
    (outside_candidate / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)

    if mode == "manifest":
        manifest = root / "release-manifest.txt"
        manifest.write_text("linked/candidate/module.py\n", encoding="utf-8")
        result = _run_audit("--manifest", str(manifest), root=root)
    else:
        result = _run_audit("--candidate", str(root / "linked" / "candidate"), root=root)

    assert result.returncode == 1
    assert "symlink" in result.stdout


def test_candidate_mode_scans_a_tree_and_rejects_empty_trees(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    empty_result = _run_audit("--candidate", str(candidate), root=tmp_path)
    assert empty_result.returncode == 1
    assert "empty-candidate" in empty_result.stdout

    (candidate / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    populated_result = _run_audit("--candidate", str(candidate), root=tmp_path)
    assert populated_result.returncode == 0, populated_result.stdout + populated_result.stderr
    assert "Scanned 1 candidate file" in populated_result.stdout


def test_candidate_path_policy_is_relative_to_audited_root(tmp_path: Path) -> None:
    candidate = tmp_path / "paper" / "repository"
    candidate.mkdir(parents=True)
    (candidate / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = _run_audit("--candidate", str(candidate), root=candidate)

    assert result.returncode == 0, result.stdout + result.stderr


def test_staged_mode_audits_index_blob_instead_of_working_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    settings = repo / "settings.py"
    embedded_key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(f'TOKEN = "{embedded_key}"\n', encoding="utf-8")
    subprocess.run(["git", "add", "settings.py"], cwd=repo, check=True)
    settings.write_text("VALUE = 1\n", encoding="utf-8")

    result = _run_audit("--staged", root=repo)

    assert result.returncode == 1
    assert "embedded-secret" in result.stdout


def test_tracked_mode_audits_index_blob_instead_of_working_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    settings = repo / "settings.py"
    embedded_key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(f'TOKEN = "{embedded_key}"\n', encoding="utf-8")
    subprocess.run(["git", "add", "settings.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Audit Test", "-c", "user.email=audit@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    settings.write_text("VALUE = 1\n", encoding="utf-8")

    result = _run_audit("--tracked-only", root=repo)

    assert result.returncode == 1
    assert "embedded-secret" in result.stdout


@pytest.mark.parametrize(
    "relative_path",
    [
        "paper/manuscript.tex",
        "examples/case.json",
        "scripts/run_demo.py",
        "scripts/eval_who_grade.py",
        "baselines/model.py",
    ],
)
def test_audit_rejects_non_release_source_paths(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")

    assert "forbidden-path" in _codes([path])


@pytest.mark.parametrize(
    "relative_path",
    [
        "results/tumor.json",
        "result/tumor.json",
        "predictions/tumor.json",
        "prediction/tumor.json",
        ".venv/config.txt",
        "venv/config.txt",
        "tumor_result.json",
        "who_prediction.json",
    ],
)
def test_audit_rejects_result_and_environment_artifacts(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    assert "forbidden-path" in _codes([path])


def test_audit_rejects_secret_checkpoint_private_path_and_symlink(tmp_path: Path) -> None:
    secret_file = tmp_path / ".env"
    checkpoint = tmp_path / "weights.safetensors"
    unsafe = tmp_path / "settings.py"
    private_path = "/" + "userhome/account/private"
    embedded_key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    unsafe.write_text(
        f'OPENAI_API_KEY = "{embedded_key}"\n'
        f'MODEL_ROOT = "{private_path}"\n',
        encoding="utf-8",
    )
    target = tmp_path / "target.txt"
    target.write_text("safe\n", encoding="utf-8")
    link = tmp_path / "linked.txt"
    link.symlink_to(target)

    violations = audit_paths([secret_file, checkpoint, unsafe, link])

    assert {violation.code for violation in violations} == {
        "embedded-secret",
        "forbidden-path",
        "private-path",
        "symlink",
    }


@pytest.mark.parametrize("name", [".env.local", ".env.production", ".env.secret"])
def test_audit_rejects_suffixed_env_files(tmp_path: Path, name: str) -> None:
    env_file = tmp_path / name
    env_file.write_text("VALUE=placeholder\n", encoding="utf-8")

    assert _codes([env_file]) == {"forbidden-path"}


def test_audit_distinguishes_model_token_counts_from_credentials(tmp_path: Path) -> None:
    model_config = tmp_path / "model_config.py"
    model_config.write_text('CONFIG = {"num_query_token": 32}\n', encoding="utf-8")
    credential = tmp_path / "credential.py"
    credential.write_text('ACCESS_TOKEN = "not-a-placeholder"\n', encoding="utf-8")

    assert audit_paths([model_config]) == []
    assert _codes([credential]) == {"embedded-secret"}


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("pad_token", '"$$"'),
        ("bos_token", '"[BOS]"'),
        ("eos_token", '"[EOS]"'),
        ("unk_token", '"[UNK]"'),
        ("mask_token", '"[MASK]"'),
    ],
)
def test_audit_allows_model_vocabulary_token_literals(
    tmp_path: Path,
    field_name: str,
    field_value: str,
) -> None:
    model_config = tmp_path / "model_config.py"
    model_config.write_text(f"{field_name} = {field_value}\n", encoding="utf-8")

    assert audit_paths([model_config]) == []


@pytest.mark.parametrize("field_name", ["api_token", "access_token", "auth_token"])
def test_audit_still_rejects_credential_token_literals(
    tmp_path: Path,
    field_name: str,
) -> None:
    credential = tmp_path / "credential.py"
    credential.write_text(
        f'{field_name} = "literal_abcdefghijklmnopqrstuvwxyz123456"\n',
        encoding="utf-8",
    )

    assert _codes([credential]) == {"embedded-secret"}


def test_audit_rejects_binary_content_even_with_text_suffix(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")

    assert _codes([artifact]) == {"binary-file"}


def test_audit_allows_only_valid_pngs_in_attributed_public_case_directory(tmp_path: Path) -> None:
    allowed = tmp_path / "test_data" / "utsw_glioma_bt0156" / "images" / "t1_001.png"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fixture")

    assert audit_paths([allowed], root=tmp_path) == []

    outside = tmp_path / "images" / "t1_001.png"
    outside.parent.mkdir()
    outside.write_bytes(allowed.read_bytes())
    assert _codes([outside]) == {"binary-file", "forbidden-path"}

    wrong_case = tmp_path / "test_data" / "other_case" / "images" / "t1_001.png"
    wrong_case.parent.mkdir(parents=True)
    wrong_case.write_bytes(allowed.read_bytes())
    assert _codes([wrong_case]) == {"binary-file", "forbidden-path"}

    disguised = allowed.with_name("disguised.png")
    disguised.write_bytes(b"not a png")
    assert {item.code for item in audit_paths([disguised], root=tmp_path)} == {"binary-file"}


@pytest.mark.parametrize("style", ["quoted", "dotenv", "yaml"])
def test_audit_rejects_non_placeholder_credential_assignments(tmp_path: Path, style: str) -> None:
    field_name = "ACCESS" + "_TOKEN"
    field_value = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"
    if style == "quoted":
        credential = tmp_path / "settings.py"
        assignment = f'{field_name} = "{field_value}"\n'
    elif style == "dotenv":
        credential = tmp_path / ".env.example"
        assignment = f"{field_name}={field_value}\n"
    else:
        credential = tmp_path / "settings.yaml"
        assignment = f"{field_name.lower()}: {field_value}\n"
    credential.write_text(assignment, encoding="utf-8")

    assert _codes([credential]) == {"embedded-secret"}


def test_audit_rejects_json_credential_literals(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    field_name = "api" + "_key"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(json.dumps({field_name: field_value}), encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


@pytest.mark.parametrize(
    "field_name",
    ["github_token", "HF_TOKEN", "aws_access_key_id", "aws_secret_access_key", "signing_key"],
)
def test_audit_rejects_semantic_json_credential_names(tmp_path: Path, field_name: str) -> None:
    settings = tmp_path / "settings.json"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(json.dumps({field_name: field_value}), encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


def test_audit_allows_ordinary_json_key(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"key": "display-value"}), encoding="utf-8")

    assert audit_paths([settings]) == []


def test_audit_allows_json_null_credential_placeholder(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    field_name = "api" + "_key"
    settings.write_text(json.dumps({field_name: None}), encoding="utf-8")

    assert audit_paths([settings]) == []


@pytest.mark.parametrize(
    "signature",
    [
        "ghp_" + "abcdefghijklmnopqrstuvwxyz123456",
        "hf_" + "abcdefghijklmnopqrstuvwxyz123456",
        "AKIA" + "ABCDEFGHIJKLMNOP",
    ],
)
def test_audit_rejects_provider_secret_signatures(tmp_path: Path, signature: str) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"value": signature}), encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


def test_audit_rejects_shell_reference_with_literal_default(tmp_path: Path) -> None:
    env_template = tmp_path / ".env.example"
    field_name = "API" + "_KEY"
    fallback = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    env_template.write_text(f"{field_name}=${{EXTERNAL_KEY:-{fallback}}}\n", encoding="utf-8")

    assert _codes([env_template]) == {"embedded-secret"}


def test_audit_allows_python_environment_credential_loading(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    field_name = "api" + "_key"
    environment_name = "OPENAI_API" + "_KEY"
    settings.write_text(f'{field_name} = os.environ.get("{environment_name}")\n', encoding="utf-8")

    assert audit_paths([settings]) == []


@pytest.mark.parametrize(
    "field_name",
    [
        "session_token",
        "deploy_secret",
        "database_password",
        "cloud_credential",
        "aws_access_key_id",
        "private_key",
        "signing_key",
        "github_token",
        "HF_TOKEN",
        "aws_secret_access_key",
    ],
)
def test_audit_rejects_semantic_python_credential_names(tmp_path: Path, field_name: str) -> None:
    settings = tmp_path / "settings.py"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(f'{field_name} = "{field_value}"\n', encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("GITHUB_TOKEN", "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"),
        ("HF_TOKEN", "hf_" + "abcdefghijklmnopqrstuvwxyz123456"),
        ("AWS_ACCESS_KEY_ID", "AKIA" + "ABCDEFGHIJKLMNOP"),
        ("AWS_SECRET_ACCESS_KEY", "literal_abcdefghijklmnopqrstuvwxyz123456"),
    ],
)
def test_audit_rejects_semantic_dotenv_credentials(
    tmp_path: Path, field_name: str, field_value: str
) -> None:
    env_template = tmp_path / ".env.example"
    env_template.write_text(f"{field_name}={field_value}\n", encoding="utf-8")

    assert _codes([env_template]) == {"embedded-secret"}


def test_audit_allows_dotenv_credential_references_and_placeholders(tmp_path: Path) -> None:
    env_template = tmp_path / ".env.example"
    env_template.write_text(
        "GITHUB_TOKEN=${GITHUB_TOKEN}\nHF_TOKEN=<your-token>\nAWS_ACCESS_KEY_ID=\n",
        encoding="utf-8",
    )

    assert audit_paths([env_template]) == []


def test_audit_allows_ordinary_key_and_sensitive_environment_expression(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    sensitive_name = "github" + "_token"
    settings.write_text(
        f'key = "display-value"\n{sensitive_name} = os.environ.get("GITHUB_TOKEN")\n',
        encoding="utf-8",
    )

    assert audit_paths([settings]) == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_position", "4"),
        ("token_budget", "8192"),
        ("token_count", "42"),
        ("password_policy", '"minimum 12 characters"'),
        ("token_pattern", 'r"[A-Za-z]+"'),
        ("tokenizer", '"default"'),
        ("tokenization", '"wordpiece"'),
    ],
)
def test_audit_allows_noncredential_sensitive_word_identifiers(
    tmp_path: Path, field_name: str, value: str
) -> None:
    settings = tmp_path / "settings.py"
    settings.write_text(f"{field_name} = {value}\n", encoding="utf-8")

    assert audit_paths([settings]) == []


def test_audit_rejects_python_dict_credential_literal(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    field_name = "api" + "_key"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(f'SETTINGS = {{"{field_name}": "{field_value}"}}\n', encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


def test_audit_rejects_python_call_credential_literal(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    field_name = "api" + "_key"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(f'client = Client({field_name}="{field_value}")\n', encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


def test_audit_rejects_python_subscript_credential_literal(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.py"
    field_name = "api" + "_key"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings_file.write_text(f'settings["{field_name}"] = "{field_value}"\n', encoding="utf-8")

    assert _codes([settings_file]) == {"embedded-secret"}


def test_audit_rejects_python_foldable_credential_literal(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    field_name = "api" + "_key"
    settings.write_text(f'{field_name} = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"\n', encoding="utf-8")

    assert _codes([settings]) == {"embedded-secret"}


def test_audit_allows_empty_and_explicit_credential_placeholders(tmp_path: Path) -> None:
    first_name = "ACCESS" + "_TOKEN"
    second_name = "AUTH" + "_TOKEN"
    third_name = "CLIENT" + "_SECRET"
    env_template = tmp_path / ".env.example"
    env_template.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=",
                f'{first_name}=""',
                f"{second_name}=<your-token>",
                f"{third_name}=${{{third_name}}}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    yaml_template = tmp_path / "settings.yaml"
    yaml_template.write_text("database_password: replace_me\n", encoding="utf-8")

    assert audit_paths([env_template, yaml_template]) == []


def test_audit_detects_han_characters_in_any_candidate_text(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.custom"
    text_file.write_text("English then " + chr(0x6C49) + "\n", encoding="utf-8")

    assert _codes([text_file]) == {"han-character"}


def test_audit_scans_text_even_when_its_path_is_forbidden(tmp_path: Path) -> None:
    text_file = tmp_path / "paper" / "notes.txt"
    text_file.parent.mkdir()
    text_file.write_text("English then " + chr(0x6C49) + "\n", encoding="utf-8")

    assert _codes([text_file]) == {"forbidden-path", "han-character"}


def test_audit_rejects_custom_endpoint_configuration_and_relay_names(tmp_path: Path) -> None:
    endpoint_name = "OPENAI_BASE" + "_URL"
    argument_name = "base" + "_url"
    relay_name = "".join(("open", "router"))
    settings = tmp_path / "client.py"
    settings.write_text(
        f'{endpoint_name} = "https://relay.invalid/v1"\n'
        f"client = Client({argument_name}=endpoint)\n"
        f'provider = "{relay_name}"\n',
        encoding="utf-8",
    )

    assert _codes([settings]) == {"custom-openai-endpoint", "relay-service"}


def test_audit_rejects_every_non_ascii_text_byte(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("ASCII followed by " + chr(0x00E9) + "\n", encoding="utf-8")

    assert _codes([text_file]) == {"non-ascii"}


@pytest.mark.parametrize(
    "private_path",
    [
        "/" + "userhome/account/project",
        "/" + "home/account/project",
        "/" + "Users/account/project",
        "C:" + "\\Users\\account\\project",
        "D:" + "/" + "Users/account/project",
    ],
)
def test_audit_rejects_all_private_user_roots(tmp_path: Path, private_path: str) -> None:
    settings = tmp_path / "settings.txt"
    settings.write_text(f"model_path={private_path}\n", encoding="ascii")

    assert _codes([settings]) == {"private-path"}


def test_audit_rejects_tuple_assigned_credential_literals(tmp_path: Path) -> None:
    settings = tmp_path / "settings.py"
    field_name = "api" + "_key"
    field_value = "literal_" + "abcdefghijklmnopqrstuvwxyz123456"
    settings.write_text(
        f'{field_name}, model_name = "{field_value}", "gpt-model"\n',
        encoding="ascii",
    )

    assert _codes([settings]) == {"embedded-secret"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "features/case.txt",
        "feature/case.txt",
        "reports/summary.txt",
        "report/summary.txt",
        "weights/model.txt",
        "weight/model.txt",
    ],
)
def test_audit_rejects_generated_feature_report_and_weight_trees(
    tmp_path: Path,
    relative_path: str,
) -> None:
    artifact = tmp_path / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text("generated\n", encoding="ascii")

    assert _codes([artifact]) == {"forbidden-path"}


@pytest.mark.parametrize(
    "source",
    [
        'client = OpenAI(**{"base_" + "url": "https://relay.invalid/v1"})\n',
        'setattr(openai, "api_" + "base", "https://relay.invalid/v1")\n',
        'client = OpenAI("https://relay.invalid/v1")\n',
    ],
)
def test_audit_ast_rejects_obfuscated_unofficial_openai_endpoints(
    tmp_path: Path,
    source: str,
) -> None:
    settings = tmp_path / "client.py"
    settings.write_text(source, encoding="ascii")

    assert _codes([settings]) == {"custom-openai-endpoint"}


def test_audit_ast_rejects_folded_relay_names(tmp_path: Path) -> None:
    settings = tmp_path / "client.py"
    settings.write_text('provider = "open" + "router"\n', encoding="ascii")

    assert _codes([settings]) == {"relay-service"}


def test_audit_allows_official_openai_api_without_endpoint_override(tmp_path: Path) -> None:
    settings = tmp_path / "client.py"
    settings.write_text(
        'client = OpenAI()\nofficial_documentation = "https://api.openai.com/v1"\n',
        encoding="ascii",
    )

    assert audit_paths([settings]) == []


@pytest.mark.parametrize(
    "source",
    [
        "import " + "mini" + "gpt4.models as external_model\n",
        "from " + "mini" + "gpt4.models import MiniGPT4\n",
    ],
)
def test_audit_ast_rejects_executable_minigpt4_imports(tmp_path: Path, source: str) -> None:
    module = tmp_path / "runtime.py"
    module.write_text(source, encoding="ascii")

    assert _codes([module]) == {"external-runtime-import"}


@pytest.mark.parametrize(
    "source",
    [
        'import importlib as loader\nloader.import_module("external.model")\n',
        '__import__("external.model")\n',
        (
            "from importlib import util as loader\n"
            'loader.spec_from_file_location("external", source_file)\n'
        ),
    ],
)
def test_audit_ast_rejects_dynamic_external_runtime_imports(
    tmp_path: Path,
    source: str,
) -> None:
    module = tmp_path / "runtime.py"
    module.write_text(source, encoding="ascii")

    assert _codes([module]) == {"dynamic-import"}


def test_audit_allows_static_importlib_metadata_use(tmp_path: Path) -> None:
    module = tmp_path / "version.py"
    module.write_text(
        'from importlib.metadata import version\nPACKAGE_VERSION = version("package")\n',
        encoding="ascii",
    )

    assert audit_paths([module]) == []


@pytest.mark.parametrize(
    "source",
    [
        (
            "import sys as runtime\n"
            "runtime." + "path[:0] = [str(external_source)]\n"
        ),
        (
            "from sys import " + "path as search_path\n"
            "search_path.append(str(external_source))\n"
        ),
        (
            "import sys\n"
            'getattr(sys, "pa" + "th").insert(0, str(external_source))\n'
        ),
    ],
)
def test_audit_ast_rejects_aliased_and_obfuscated_sys_path_mutations(
    tmp_path: Path,
    source: str,
) -> None:
    module = tmp_path / "runtime.py"
    module.write_text(source, encoding="ascii")

    assert _codes([module]) == {"source-path-injection"}


@pytest.mark.parametrize("variable", ["PYTHON" + "PATH", "BRAINVLM_" + "SOURCE", "BRAINVLM_SOURCE_" + "ROOT"])
def test_audit_ast_rejects_removed_environment_injection(
    tmp_path: Path,
    variable: str,
) -> None:
    module = tmp_path / "runtime.py"
    module.write_text(
        f'import os\nos.environ["{variable}"] = str(external_source)\n',
        encoding="ascii",
    )

    assert _codes([module]) == {"source-path-injection"}


def test_audit_shell_scan_rejects_removed_environment_and_source_injection(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "run.sh"
    python_path = "PYTHON" + "PATH"
    source_root = "BRAINVLM_SOURCE_" + "ROOT"
    launcher.write_text(
        f'export {python_path}="${{{source_root}}}:${{{python_path}}}"\n'
        f'. "${{{source_root}}}/activate.sh"\n',
        encoding="ascii",
    )

    assert _codes([launcher]) == {"removed-source-interface", "source-path-injection"}


@pytest.mark.parametrize(
    "option",
    [
        "--" + "source-root",
        "--" + "brainvlm-source",
        "--" + "adapter",
        "--" + "model-factory",
    ],
)
def test_audit_ast_rejects_removed_cli_options(tmp_path: Path, option: str) -> None:
    command = tmp_path / "command.py"
    command.write_text(
        f'import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument("{option}")\n',
        encoding="ascii",
    )

    assert _codes([command]) == {"removed-source-interface"}


@pytest.mark.parametrize(
    "source",
    [
        (
            "from subprocess import run as launch\n"
            'launch([python, str(external_project / "tra" + "in.py")])\n'
        ),
        (
            "import subprocess as process\n"
            'process.Popen((python, "-m", "other_project." + "train"))\n'
        ),
    ],
)
def test_audit_ast_rejects_external_training_delegation(
    tmp_path: Path,
    source: str,
) -> None:
    command = tmp_path / "command.py"
    command.write_text(source, encoding="ascii")

    assert _codes([command]) == {"external-training-delegation"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/superpowers/plans/2026-11-27-self-contained-brainvlm-midline.md",
        "docs/superpowers/specs/2026-11-27-self-contained-brainvlm-midline-design.md",
    ],
)
def test_audit_allows_removed_names_only_in_approved_prohibition_text(
    tmp_path: Path,
    relative_path: str,
) -> None:
    policy = tmp_path / relative_path
    policy.parent.mkdir(parents=True)
    python_path = "PYTHON" + "PATH"
    source_option = "--" + "source-root"
    policy.write_text(
        f"The release must not set {python_path} or expose {source_option}.\n",
        encoding="ascii",
    )
    assert audit_paths([policy]) == []

    policy.write_text(
        f"Set {python_path}=/external and pass {source_option} to the command.\n",
        encoding="ascii",
    )
    assert _codes([policy]) == {"removed-source-interface"}


def test_audit_allows_legacy_source_paths_only_in_approved_provenance(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "docs" / "provenance" / "brainvlm-midline-sources.md"
    provenance.parent.mkdir(parents=True)
    historical_path = "mini" + "gpt4/models/model.py"
    provenance.write_text(f"Historical source: `{historical_path}`.\n", encoding="ascii")
    assert audit_paths([provenance]) == []

    provenance.write_text(f"Import and execute `{historical_path}`.\n", encoding="ascii")
    assert _codes([provenance]) == {"external-runtime-import"}

    instructions = tmp_path / "docs" / "runtime.md"
    instructions.write_text(f"Install and import `{historical_path}`.\n", encoding="ascii")
    assert _codes([instructions]) == {"external-runtime-import"}


def test_audit_rejects_prohibited_term_outside_policy_files(tmp_path: Path) -> None:
    module = tmp_path / "model.py"
    prohibited_term = "base" + "line"
    module.write_text(f"# {prohibited_term} implementation\n", encoding="utf-8")

    assert _codes([module]) == {"prohibited-term"}


def test_audit_allows_env_template_and_ordinary_source(tmp_path: Path) -> None:
    env_example = tmp_path / ".env.example"
    env_example.write_text("OPENAI_API_KEY=\nOPENAI_MODEL=\n", encoding="utf-8")
    module = tmp_path / "midline_agent" / "datasets" / "annotations.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")

    assert audit_paths([env_example, module]) == []


def test_repository_env_template_has_only_official_openai_settings() -> None:
    variables = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert variables == ["OPENAI_API_KEY=", "OPENAI_MODEL="]


def test_gitignore_uses_root_runtime_paths_without_hiding_source_package() -> None:
    patterns = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/.env.*" in patterns
    assert "!/.env.example" in patterns
    assert "/data/" in patterns
    assert "/outputs/" in patterns
    assert "/output/" in patterns
    assert "/logs/" in patterns
    assert "/cache/" in patterns
    assert "/checkpoints/" in patterns
    assert "/paper/" in patterns
    assert "data/" not in patterns
    assert "datasets/" not in patterns
