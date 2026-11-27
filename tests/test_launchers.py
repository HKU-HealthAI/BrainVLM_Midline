from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL_LAUNCHERS = (
    ROOT / "scripts" / "run_tumor_prediction.sh",
    ROOT / "scripts" / "run_who_grading.sh",
)
SLURM_LAUNCHERS = (
    ROOT / "scripts" / "run_tumor_prediction.sbatch",
    ROOT / "scripts" / "run_who_grading.sbatch",
)
ALL_LAUNCHERS = SHELL_LAUNCHERS + SLURM_LAUNCHERS


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_launcher_has_valid_bash_syntax(launcher: Path) -> None:
    completed = subprocess.run(
        ["bash", "-n", str(launcher)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_launcher_is_executable_and_uses_strict_mode(launcher: Path) -> None:
    text = launcher.read_text(encoding="ascii")

    assert os.access(launcher, os.X_OK)
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_launcher_contains_no_private_or_cluster_specific_values(launcher: Path) -> None:
    text = launcher.read_text(encoding="ascii")
    forbidden = (
        r"/userhome/",
        r"/home/",
        r"CUDA_VISIBLE_DEVICES",
        r"#SBATCH\s+--(?:account|partition|mail-user)",
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"relay[_-]?service",
        r"OPENAI_API_KEY",
        r"BRAINVLM_SOURCE",
        r"PYTHONPATH",
    )

    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_tumor_shell_launcher_exposes_all_runtime_inputs() -> None:
    text = SHELL_LAUNCHERS[0].read_text(encoding="ascii")

    for variable in (
        "ANNOTATIONS",
        "IMAGE_ROOT",
        "LLAMA_MODEL",
        "TUMOR_CHECKPOINT",
        "MIL_CHECKPOINT",
        "OUTPUT_JSON",
        "DEVICE",
    ):
        assert f'${{{variable}' in text
    assert "scripts.predict_tumor" in text
    assert "BIOMEDCLIP_PRETRAINED" in text


def test_who_shell_launcher_accepts_tumor_result_or_explicit_type() -> None:
    text = SHELL_LAUNCHERS[1].read_text(encoding="ascii")

    for variable in (
        "ANNOTATIONS",
        "IMAGE_ROOT",
        "LLAMA_MODEL",
        "WHO_CHECKPOINT",
        "OUTPUT_JSON",
        "DEVICE",
    ):
        assert f'${{{variable}' in text
    assert "scripts.predict_who_grade" in text
    assert "--tumor-result" in text
    assert "--tumor-type" in text


def _capture_shell_invocation(
    tmp_path: Path,
    launcher: Path,
    environment: dict[str, str],
) -> list[str]:
    capture = tmp_path / "arguments.txt"
    python = tmp_path / "capture-python"
    python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
        encoding="ascii",
    )
    python.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON_BIN": str(python),
            "CAPTURE_PATH": str(capture),
            **environment,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return capture.read_text(encoding="ascii").splitlines()


def test_tumor_shell_launcher_forwards_internal_runtime_arguments(tmp_path: Path) -> None:
    arguments = _capture_shell_invocation(
        tmp_path,
        SHELL_LAUNCHERS[0],
        {
            "ANNOTATIONS": "/inputs/annotations.json",
            "IMAGE_ROOT": "/inputs/images",
            "LLAMA_MODEL": "/models/llama",
            "BIOMEDCLIP_PRETRAINED": "/models/biomedclip.bin",
            "TUMOR_CHECKPOINT": "/models/tumor.pth",
            "MIL_CHECKPOINT": "/models/mil.pth",
            "OUTPUT_JSON": "/run/tumor.json",
            "DEVICE": "cuda:3",
        },
    )

    assert arguments == [
        "-m",
        "scripts.predict_tumor",
        "--annotations",
        "/inputs/annotations.json",
        "--image-root",
        "/inputs/images",
        "--llama-model",
        "/models/llama",
        "--checkpoint",
        "/models/tumor.pth",
        "--mil-checkpoint",
        "/models/mil.pth",
        "--output",
        "/run/tumor.json",
        "--device",
        "cuda:3",
        "--biomedclip-pretrained",
        "/models/biomedclip.bin",
    ]


def test_who_shell_launcher_requires_tumor_input(tmp_path: Path) -> None:
    python = tmp_path / "capture-python"
    python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    python.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(SHELL_LAUNCHERS[1])],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON_BIN": str(python),
            "ANNOTATIONS": "/inputs/annotations.json",
            "IMAGE_ROOT": "/inputs/images",
            "LLAMA_MODEL": "/models/llama",
            "WHO_CHECKPOINT": "/models/who.pth",
            "OUTPUT_JSON": "/run/who.json",
            "DEVICE": "cuda:1",
            "TUMOR_RESULT": "",
            "TUMOR_TYPE": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "TUMOR_RESULT" in completed.stderr


def test_who_shell_launcher_forwards_explicit_tumor_type(tmp_path: Path) -> None:
    arguments = _capture_shell_invocation(
        tmp_path,
        SHELL_LAUNCHERS[1],
        {
            "ANNOTATIONS": "/inputs/annotations.json",
            "IMAGE_ROOT": "/inputs/images",
            "LLAMA_MODEL": "/models/llama",
            "WHO_CHECKPOINT": "/models/who.pth",
            "OUTPUT_JSON": "/run/who.json",
            "DEVICE": "cuda:2",
            "BIOMEDCLIP_PRETRAINED": "",
            "TUMOR_TYPE": "tumor_3",
            "TUMOR_RESULT": "",
        },
    )

    assert arguments == [
        "-m",
        "scripts.predict_who_grade",
        "--annotations",
        "/inputs/annotations.json",
        "--image-root",
        "/inputs/images",
        "--llama-model",
        "/models/llama",
        "--checkpoint",
        "/models/who.pth",
        "--output",
        "/run/who.json",
        "--device",
        "cuda:2",
        "--tumor-type",
        "tumor_3",
    ]


@pytest.mark.parametrize(
    ("launcher", "shell_name", "default_output"),
    (
        (SLURM_LAUNCHERS[0], "run_tumor_prediction.sh", "tumor_prediction.json"),
        (SLURM_LAUNCHERS[1], "run_who_grading.sh", "who_grading.json"),
    ),
)
def test_slurm_launcher_uses_caller_owned_run_directory(
    launcher: Path,
    shell_name: str,
    default_output: str,
) -> None:
    text = launcher.read_text(encoding="ascii")

    assert "#SBATCH --gres=gpu:1" in text
    assert '${PROJECT_ROOT:' in text
    assert '${RUN_DIR:' in text
    assert '${DEVICE:' in text
    assert 'mkdir -p "$RUN_DIR"' in text
    assert f'"$PROJECT_ROOT/scripts/{shell_name}"' in text
    assert default_output in text
    assert "#SBATCH --output" not in text
    assert "activate" not in text
    assert "source " not in text


@pytest.mark.parametrize(
    ("launcher", "shell_name"),
    (
        (SLURM_LAUNCHERS[0], "run_tumor_prediction.sh"),
        (SLURM_LAUNCHERS[1], "run_who_grading.sh"),
    ),
)
def test_slurm_launcher_replaces_invalid_inherited_tmpdir(
    tmp_path: Path,
    launcher: Path,
    shell_name: str,
) -> None:
    project_root = tmp_path / "project"
    scripts = project_root / "scripts"
    scripts.mkdir(parents=True)
    capture = tmp_path / "tmpdir.txt"
    downstream = scripts / shell_name
    downstream.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'test -d "$TMPDIR"\n'
        'test -w "$TMPDIR"\n'
        'printf \'%s\\n\' "$TMPDIR" > "$CAPTURE_PATH"\n',
        encoding="ascii",
    )
    downstream.chmod(0o755)
    run_dir = tmp_path / "run"

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PROJECT_ROOT": str(project_root),
            "RUN_DIR": str(run_dir),
            "DEVICE": "cuda:0",
            "CAPTURE_PATH": str(capture),
            "TMPDIR": str(tmp_path / "missing-inherited-tmpdir"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="ascii").strip() == str(run_dir / "tmp")
