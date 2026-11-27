"""Distribution-level tests for the public release package."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_SDIST_FILES = {
    "PKG-INFO",
    "setup.cfg",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "checkpoints",
    "logs",
    "outputs",
    "predictions",
    "results",
}


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    work_root = tmp_path_factory.mktemp("distribution-contract")
    source_root = work_root / "source"
    source_root.mkdir()
    for relative_path in sorted(_release_manifest()):
        source = REPOSITORY_ROOT / relative_path
        assert source.is_file(), f"missing release source: {relative_path}"
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    import build

    build_tools = Path(build.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(build_tools)
    environment.pop("BRAINVLM_SOURCE", None)
    environment.pop("BRAINVLM_SOURCE_ROOT", None)
    distribution_root = work_root / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(distribution_root),
        ],
        cwd=source_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(distribution_root.glob("*.whl"))
    source_archives = list(distribution_root.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_archives) == 1
    return wheels[0], source_archives[0]


def _release_manifest() -> set[str]:
    entries = {
        line.strip()
        for line in (REPOSITORY_ROOT / "release-manifest.txt").read_text(
            encoding="ascii"
        ).splitlines()
        if line.strip()
    }
    entries.update({"MANIFEST.in", "tests/test_packaging_contract.py"})
    return entries


def _assert_no_forbidden_artifact_paths(paths: set[str]) -> None:
    violations: list[str] = []
    for value in sorted(paths):
        path = PurePosixPath(value)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(value)
            continue
        if FORBIDDEN_DIRECTORY_NAMES.intersection(path.parts):
            violations.append(value)
    assert not violations, f"forbidden distribution files: {violations}"


def test_source_distribution_contains_the_auditable_release(
    built_distributions: tuple[Path, Path],
) -> None:
    _, source_archive = built_distributions
    with tarfile.open(source_archive, mode="r:gz") as archive:
        regular_files = {member.name for member in archive if member.isfile()}

    prefixes = {name.split("/", 1)[0] for name in regular_files}
    assert len(prefixes) == 1
    prefix = prefixes.pop()
    relative_files = {
        name.removeprefix(f"{prefix}/")
        for name in regular_files
        if name.startswith(f"{prefix}/")
    }

    expected = _release_manifest()
    assert expected <= relative_files
    unexpected = {
        name
        for name in relative_files - expected
        if name not in GENERATED_SDIST_FILES
        and not name.startswith("midline_agent.egg-info/")
    }
    assert not unexpected, f"unexpected sdist files: {sorted(unexpected)}"
    _assert_no_forbidden_artifact_paths(relative_files)


def test_wheel_contains_runtime_and_complete_legal_records(
    built_distributions: tuple[Path, Path],
) -> None:
    wheel, _ = built_distributions
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    required_runtime = {
        "midline_agent/diagnostic/brainvlm_midline/model.py",
        "midline_agent/diagnostic/brainvlm_runtime.py",
        "scripts/extract_features.py",
        "scripts/predict_tumor.py",
        "scripts/predict_who_grade.py",
        "scripts/run_pipeline.py",
        "scripts/train_attention_mil.py",
        "scripts/train_brainvlm_midline.py",
    }
    assert required_runtime <= members

    license_members = {
        name.split(".dist-info/licenses/", 1)[1]
        for name in members
        if ".dist-info/licenses/" in name
    }
    assert {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "docs/provenance/brainvlm-midline-sources.md",
        "licenses/Apache-2.0.txt",
        "licenses/BSD-3-Clause-LAVIS.txt",
        "licenses/BSD-3-Clause-MiniGPT4.txt",
    } <= license_members

    assert not any(name.startswith(("configs/", "docs/", "test_data/", "tests/")) for name in members)
    _assert_no_forbidden_artifact_paths(members)
