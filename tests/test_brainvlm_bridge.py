import sys
from pathlib import Path

import numpy as np
import pytest

from midline_agent.datasets.annotations import AnnotationRecord
from midline_agent.datasets.brainvlm import (
    PREPROCESSING_VERSION,
    BrainVlmRequest,
    build_task_request,
    preprocess_slice,
)
from midline_agent.datasets.prompts import (
    COMBINED_TASK,
    KEYFRAME_TASK,
    TUMOR_TASK,
    LabelError,
    PROMPT_VERSION,
    build_task_prompt,
    label_for_record,
)
from midline_agent.diagnostic.brainvlm_adapter import (
    BrainVlmAdapter,
    ExternalSourceError,
)
from midline_agent.diagnostic.adapters import WhoGradeRequest


def test_task_labels_are_explicit_and_missing_labels_are_rejected() -> None:
    record = AnnotationRecord("patient", "slice.png", "T1", 0, "tumor_2", "grade_3", 1)

    assert label_for_record(record, KEYFRAME_TASK) == "keyframe"
    assert label_for_record(record, TUMOR_TASK) == "tumor_2"
    assert label_for_record(record, COMBINED_TASK) == "tumor_2|grade_3"
    with pytest.raises(LabelError, match="keyframe_label"):
        label_for_record(record.__class__("patient", "slice.png", "T1", 0, "tumor_2"), KEYFRAME_TASK)


@pytest.mark.parametrize(
    ("task", "value"),
    [(KEYFRAME_TASK, 2), (TUMOR_TASK, "tumor_9"), (COMBINED_TASK, "tumor_1|grade_3")],
)
def test_task_labels_reject_invalid_values(task: str, value) -> None:
    record = AnnotationRecord("patient", "slice.png", "T1", 0, "tumor_2", "grade_3", 1)
    if task == KEYFRAME_TASK:
        record = record.__class__("patient", "slice.png", "T1", 0, "tumor_2", "grade_3", value)
    elif task == TUMOR_TASK:
        record = record.__class__("patient", "slice.png", "T1", 0, value, "grade_3", 1)
    else:
        record = record.__class__("patient", "slice.png", "T1", 0, "tumor_2", "grade_4", 1)
    with pytest.raises(LabelError):
        if task == COMBINED_TASK:
            build_task_prompt(task, label=value)
        else:
            label_for_record(record, task)


def test_prompts_are_versioned_and_task_specific() -> None:
    assert "tumor_1" in build_task_prompt(TUMOR_TASK)
    assert "grade_4" in build_task_prompt(COMBINED_TASK)
    assert "keyframe" in build_task_prompt(KEYFRAME_TASK).lower()
    with pytest.raises(LabelError):
        build_task_prompt("unknown-task")


def test_preprocessing_returns_stable_metadata_and_checksum() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    first = preprocess_slice(image, size=8)
    second = preprocess_slice(image, size=8)

    assert first.image.shape == (3, 8, 8)
    assert first.metadata == second.metadata
    assert first.metadata["preprocessing_version"] == PREPROCESSING_VERSION
    assert len(first.metadata["input_sha256"]) == 64
    assert len(first.metadata["output_sha256"]) == 64


def test_build_task_request_includes_label_only_when_annotation_has_one() -> None:
    record = AnnotationRecord("patient", "slice.png", "T1", 0, "tumor_1", "grade_4", 1)
    request = build_task_request(record, np.ones((4, 4), dtype=np.float32), COMBINED_TASK)
    assert request.target_label == "tumor_1|grade_4"
    assert request.prompt_version
    assert request.metadata["preprocessing_version"] == PREPROCESSING_VERSION


def test_external_adapter_loads_only_user_supplied_source_and_checkpoint(tmp_path: Path) -> None:
    source_root = tmp_path / "external_source"
    source_root.mkdir()
    (source_root / "runner_module.py").write_text(
        "def load_model(*, checkpoint):\n"
        "    assert checkpoint.endswith('model.ckpt')\n"
        "    class Runner:\n"
        "        def predict(self, *, task, prompt, image, metadata):\n"
        "            return {'label': 'tumor_1|grade_4', 'text': '<tumor_1> <grade_4>', 'logits': [1, 0, 0, 0, 0]}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
    try:
        response = adapter.predict(
            BrainVlmRequest(
                task=COMBINED_TASK,
                prompt=build_task_prompt(COMBINED_TASK),
                image=np.zeros((3, 8, 8), dtype=np.float32),
                metadata={"preprocessing_version": PREPROCESSING_VERSION},
            )
        )
    finally:
        sys.modules.pop("runner_module", None)
    assert response.label == "tumor_1|grade_4"
    assert response.raw_text == "<tumor_1> <grade_4>"
    assert response.logits.shape == (5,)


def test_external_adapter_rejects_missing_source_or_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ExternalSourceError, match="source root"):
        BrainVlmAdapter(tmp_path / "missing", tmp_path / "model.ckpt", "runner:load_model")
    source_root = tmp_path / "external_source"
    source_root.mkdir()
    with pytest.raises(ExternalSourceError, match="checkpoint"):
        BrainVlmAdapter(source_root, tmp_path / "missing.ckpt", "runner:load_model")


def test_external_adapter_rejects_missing_model_label(tmp_path: Path) -> None:
    source_root = tmp_path / "external_source"
    source_root.mkdir()
    (source_root / "runner_module.py").write_text(
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'text': '<tumor_1> <grade_4>', 'logits': [1, 0, 0, 0, 0]}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
    try:
        with pytest.raises(ExternalSourceError, match="label"):
            adapter.predict(
                BrainVlmRequest(
                    task=COMBINED_TASK,
                    prompt=build_task_prompt(COMBINED_TASK),
                    image=np.zeros((3, 8, 8), dtype=np.float32),
                    metadata={"preprocessing_version": PREPROCESSING_VERSION},
                )
            )
    finally:
        sys.modules.pop("runner_module", None)


def test_external_adapter_rejects_missing_tumor_logits(tmp_path: Path) -> None:
    source_root = tmp_path / "external_source"
    source_root.mkdir()
    (source_root / "runner_module.py").write_text(
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'label': 'tumor_1'}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
    try:
        with pytest.raises(ExternalSourceError, match="logits"):
            adapter.predict(
                BrainVlmRequest(
                    task="tumor_classification",
                    prompt=build_task_prompt(TUMOR_TASK),
                    image=np.zeros((3, 8, 8), dtype=np.float32),
                    metadata={"preprocessing_version": PREPROCESSING_VERSION},
                )
            )
    finally:
        sys.modules.pop("runner_module", None)


def test_external_import_uses_source_file_without_sys_path_or_module_reuse(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runner_module.py").write_text(
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'label': 'tumor_1', 'logits': [1, 0, 0, 0, 0]}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    stale = type("Stale", (), {"load_model": lambda **kwargs: None})
    original_path = list(sys.path)
    sys.modules["runner_module"] = stale
    try:
        adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
        assert adapter._runner.__class__.__name__ == "Runner"
        assert sys.path == original_path
    finally:
        sys.modules.pop("runner_module", None)


def test_external_import_rejects_module_path_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    with pytest.raises(ExternalSourceError, match="module"):
        BrainVlmAdapter(source_root, checkpoint, "../outside:load_model")


def test_preprocessing_rejects_float32_overflow_and_records_input_mode() -> None:
    with pytest.raises(ValueError, match="finite"):
        preprocess_slice(np.array([[1e40]], dtype=np.float64), size=4)
    grayscale = preprocess_slice(np.array([[0.0, 2.0]], dtype=np.float32), size=4)
    rgb = preprocess_slice(np.full((2, 2, 3), 255, dtype=np.uint16), size=4)
    rgb_unit = preprocess_slice(np.full((2, 2, 3), 0.5, dtype=np.float32), size=4)
    assert grayscale.metadata["normalization"] == "grayscale_minmax"
    assert rgb.metadata["normalization"] == "rgb_scale_255"
    assert rgb_unit.metadata["normalization"] == "rgb_clip_0_1"
    assert grayscale.metadata["input_dtype"] == "float32"


def test_brainvlm_request_requires_supported_task_positive_shape_and_consistent_target() -> None:
    with pytest.raises(ValueError, match="supported"):
        BrainVlmRequest("unknown", "prompt", np.zeros((3, 2, 2), dtype=np.float32), {"preprocessing_version": PREPROCESSING_VERSION})
    with pytest.raises(ValueError, match="positive"):
        BrainVlmRequest(TUMOR_TASK, "prompt", np.zeros((3, 0, 2), dtype=np.float32), {"preprocessing_version": PREPROCESSING_VERSION})
    with pytest.raises(ValueError, match="target"):
        BrainVlmRequest(
            TUMOR_TASK,
            build_task_prompt(TUMOR_TASK),
            np.zeros((3, 2, 2), dtype=np.float32),
            {"preprocessing_version": PREPROCESSING_VERSION},
            target_label="tumor_9",
        )


def test_predict_who_grade_requires_brainvlm_prompt_version_and_preserves_it(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runner_module.py").write_text(
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            assert kwargs['metadata']['prompt_version'] == 'brainvlm-prompts-v1'\n"
        "            return {'label': 'tumor_1|grade_4', 'text': '<tumor_1> <grade_4>', 'logits': [1, 0, 0, 0, 0]}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
    try:
        with pytest.raises(ExternalSourceError, match="prompt version"):
            adapter.predict_who_grade(WhoGradeRequest("slice", np.zeros((3, 4, 4)), "who-grade-v1"))
        result = adapter.predict_who_grade(WhoGradeRequest("slice", np.zeros((3, 4, 4)), PROMPT_VERSION))
    finally:
        sys.modules.pop("runner_module", None)
    assert result.prompt_version == PROMPT_VERSION


def test_cli_scripts_use_validated_brainvlm_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    extract_source = (root / "scripts/extract_features.py").read_text(encoding="utf-8")
    pipeline_source = (root / "scripts/run_pipeline.py").read_text(encoding="utf-8")
    assert "load_brainvlm_adapter" in extract_source
    assert "load_external_adapter" not in extract_source
    assert "load_brainvlm_adapter" in pipeline_source
    assert "load_external_adapter" not in pipeline_source


def test_external_import_supports_sibling_module_and_restores_preloaded_name(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "helper.py").write_text("VALUE = 'source-helper'\n", encoding="utf-8")
    (source_root / "runner_module.py").write_text(
        "from helper import VALUE\n"
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'label': 'tumor_1', 'logits': [1, 0, 0, 0, 0], 'value': VALUE}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    stale_helper = type("StaleHelper", (), {"VALUE": "stale-helper"})
    original_path = list(sys.path)
    sys.modules["helper"] = stale_helper
    try:
        adapter = BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
        assert adapter._runner.predict(task="tumor_classification", prompt="", image=None, metadata={})["value"] == "source-helper"
        assert sys.modules["helper"] is stale_helper
        assert sys.path == original_path
    finally:
        sys.modules.pop("helper", None)


def test_external_import_supports_package_relative_module_and_cleans_modules(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    package_root = source_root / "brain_package"
    package_root.mkdir(parents=True)
    (package_root / "helper.py").write_text("VALUE = 'package-helper'\n", encoding="utf-8")
    (package_root / "__init__.py").write_text(
        "from .helper import VALUE\n"
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'label': 'tumor_1', 'logits': [1, 0, 0, 0, 0], 'value': VALUE}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    before = set(sys.modules)
    adapter = BrainVlmAdapter(source_root, checkpoint, "brain_package:load_model")
    assert adapter._runner.predict(task="tumor_classification", prompt="", image=None, metadata={})["value"] == "package-helper"
    assert set(sys.modules) == before


def test_external_import_supports_relative_import_from_nested_package_module(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    package_root = source_root / "brain_package"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "helper.py").write_text("VALUE = 'nested-helper'\n", encoding="utf-8")
    (package_root / "runner.py").write_text(
        "from .helper import VALUE\n"
        "def load_model(*, checkpoint):\n"
        "    class Runner:\n"
        "        def predict(self, **kwargs):\n"
        "            return {'label': 'tumor_1', 'logits': [1, 0, 0, 0, 0], 'value': VALUE}\n"
        "    return Runner()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    before = set(sys.modules)
    adapter = BrainVlmAdapter(source_root, checkpoint, "brain_package.runner:load_model")
    assert adapter._runner.predict(task="tumor_classification", prompt="", image=None, metadata={})["value"] == "nested-helper"
    assert set(sys.modules) == before


def test_external_import_rejects_factory_symlink_outside_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def load_model(*, checkpoint): return object()\n", encoding="utf-8")
    (source_root / "runner_module.py").symlink_to(outside)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    with pytest.raises(ExternalSourceError, match="source root"):
        BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")


def test_external_import_rejects_sibling_dependency_resolved_outside_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "helper.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    (source_root / "helper.py").symlink_to(outside)
    (source_root / "runner_module.py").write_text(
        "from helper import VALUE\n"
        "def load_model(*, checkpoint):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"external checkpoint")
    with pytest.raises(ExternalSourceError, match="outside source root"):
        BrainVlmAdapter(source_root, checkpoint, "runner_module:load_model")
