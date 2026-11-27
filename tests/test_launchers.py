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
        r"nimabo",
        r"OPENAI_API_KEY",
    )

    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_tumor_shell_launcher_exposes_all_runtime_inputs() -> None:
    text = SHELL_LAUNCHERS[0].read_text(encoding="ascii")

    for variable in (
        "ANNOTATIONS",
        "IMAGE_ROOT",
        "BRAINVLM_SOURCE",
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
        "BRAINVLM_SOURCE",
        "LLAMA_MODEL",
        "WHO_CHECKPOINT",
        "OUTPUT_JSON",
        "DEVICE",
    ):
        assert f'${{{variable}' in text
    assert "scripts.predict_who_grade" in text
    assert "--tumor-result" in text
    assert "--tumor-type" in text


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
    assert 'mkdir -p "$RUN_DIR"' in text
    assert f'"$PROJECT_ROOT/scripts/{shell_name}"' in text
    assert default_output in text
