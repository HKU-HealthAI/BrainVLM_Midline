# Self-Contained BrainVLM-Midline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship checkpoint-compatible BrainVLM-Midline tumor prediction, feature extraction, WHO grading, and focused training entirely from the MidlineAgent repository, then prove the BT0156 example works without another project-source checkout.

**Architecture:** Add a focused `midline_agent.diagnostic.brainvlm_midline` package containing the model, prompt, vision, dataset, collation, and training responsibilities extracted from the authors' modified MiniGPT-4 working tree. Keep the existing standalone prediction services and released logits MIL implementation as the patient-level service layer, but make every CLI and the Diagnostic workflow construct the internal runtime directly. Preserve private pre-migration reference artifacts outside Git and compare the installed release package against them with strict tensor and output parity checks.

**Tech Stack:** Python 3.9, PyTorch 2.1.2, torchvision 0.16.2, Transformers 4.46.3, PEFT 0.2.0, OpenCLIP 2.24.0, OmegaConf 2.3.0, NumPy 1.26.4, Pillow 11.3.0, safetensors, pytest, Slurm, setuptools.

---

## File Map

The implementation uses these focused units:

- `midline_agent/diagnostic/brainvlm_midline/config.py`: validated model and training configuration records.
- `midline_agent/diagnostic/brainvlm_midline/prompts.py`: canonical tumor and WHO instructions, templates, labels, and age context.
- `midline_agent/diagnostic/brainvlm_midline/vision.py`: BiomedCLIP construction, slice preprocessing, final-norm hook, and low-resolution encoding.
- `midline_agent/diagnostic/brainvlm_midline/llama.py`: tokenizer setup, PEFT wrapping, prompt embedding, LM loss, hidden/logit extraction, and generation.
- `midline_agent/diagnostic/brainvlm_midline/model.py`: checkpoint-compatible module assembly and stable model methods.
- `midline_agent/diagnostic/brainvlm_midline/datasets.py`: strict tumor and WHO training records, filtering, upsampling, and seeded mixing.
- `midline_agent/diagnostic/brainvlm_midline/collate.py`: validated task batch construction.
- `midline_agent/diagnostic/brainvlm_midline/training.py`: optimizer groups, scheduler, AMP/accumulation/DDP loop, save, and resume.
- `midline_agent/diagnostic/brainvlm_runtime.py`: public internal runtime, strict checkpoint load, image inference, and `SliceFeatures`.
- `midline_agent/diagnostic/feature_store.py`: versioned `hidden_and_logits` persistence and metadata validation.
- `midline_agent/diagnostic/standalone.py`: unchanged public result schemas plus shared tumor/WHO services.
- `scripts/train_brainvlm_midline.py`: focused `--task tumor|who` training command.
- `scripts/extract_features.py`, `scripts/predict_tumor.py`, `scripts/predict_who_grade.py`, `scripts/run_pipeline.py`: self-contained model asset arguments and shared service wiring.
- `scripts/run_*.sh` and `scripts/run_*.sbatch`: direct and Slurm launchers without source-root variables.
- `configs/brainvlm_tumor.example.yaml`, `configs/brainvlm_who.example.yaml`: path-free training templates.
- `requirements/diagnostic-reference.txt`: exact tested diagnostic dependency versions except platform-specific CUDA wheels.
- `docs/brainvlm-midline.md`: internal runtime, training, annotation, and asset contract.
- `docs/provenance/brainvlm-midline-sources.md`: file-level derivation and license inventory.
- `licenses/`: complete third-party license texts required by copied or derived source.
- `tests/diagnostic/`: focused internal-model, prompt, dataset, training, runtime, and integration tests.
- `tests/diagnostic/ddp_smoke.py`: two-process CPU `torchrun` proof for the public distributed training path.
- `tests/acceptance/run_installed_diagnostic.py`: deterministic installed-wheel Diagnostic orchestration acceptance driver.

Obsolete external bridge files and documents are removed only after their replacements pass tests.

### Task 1: Capture the pre-migration numerical reference

**Files:**
- Create outside Git: `.runtime/runs/self-contained-reference/capture_reference.py`
- Create outside Git: `.runtime/runs/self-contained-reference/reference.safetensors`
- Create outside Git: `.runtime/runs/self-contained-reference/reference.json`
- Inspect: `midline_agent/diagnostic/brainvlm_runtime.py`
- Inspect: `midline_agent/diagnostic/logits_mil.py`

- [x] **Step 1: Write the private capture program**

Write a temporary program outside the publication clone that loads the current external runtime and the authoritative tumor, WHO, and MIL checkpoints. It must process BT0156 in canonical annotation order and record:

```python
tensors = {
    "slice_logits": torch.stack(slice_logits).float().cpu(),       # [30, 5]
    "slice_hidden": torch.stack(slice_hidden).float().cpu(),       # [30, 4096]
    "aligned_mil_input": aligned.float().cpu(),                    # [20, 5]
    "patient_logits": patient_logits.float().cpu(),                # [5]
    "patient_probabilities": probabilities.float().cpu(),          # [5]
}
metadata = {
    "patient_id": "BT0156",
    "ordered_slices": ordered_relative_paths,
    "who_generations": generations,
    "who_votes": vote_details,
    "tumor_type": tumor_type,
    "who_grade": who_grade,
    "asset_sha256": asset_digests,
    "environment": package_versions,
}
```

Use `safetensors.torch.save_file` for tensors and ASCII JSON for metadata. The temporary program and outputs must remain beneath the root repository's ignored `.runtime/runs` directory and must never be added to Git.

- [x] **Step 2: Submit the reference job**

Run the capture program in the known working `MiniGPT-4` environment on one indexed CUDA device through Slurm. Expected: exit code 0 and both reference files exist.

- [x] **Step 3: Validate the reference envelope**

Run a CPU validation script that asserts exact tensor shapes, finite values, 30 unique ordered slices, checkpoint hashes from the approved specification, `tumor_4`, and `grade_3`.

- [x] **Step 4: Record the reference location without committing artifacts**

Add the absolute private reference location to the local implementation log only. Run `git status --short` in the publication clone and verify no reference file appears.

### Task 2: Define prompts, labels, configuration, and checkpoint manifest

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_midline/__init__.py`
- Create: `midline_agent/diagnostic/brainvlm_midline/config.py`
- Create: `midline_agent/diagnostic/brainvlm_midline/prompts.py`
- Create: `tests/diagnostic/test_brainvlm_prompts.py`
- Create: `tests/diagnostic/test_brainvlm_config.py`
- Create: `docs/provenance/brainvlm-midline-sources.md`
- Create: `licenses/BSD-3-Clause-MiniGPT4.txt`
- Create: `licenses/BSD-3-Clause-LAVIS.txt`
- Create: `licenses/Apache-2.0.txt`
- Modify: `midline_agent/diagnostic/brainvlm_runtime.py`

- [x] **Step 1: Complete the pre-copy provenance and ownership gate**

Before creating or copying any derived implementation, record every planned derived release path, its exact local source path and SHA-256, identifiable upstream repository and immutable revision, modification statement, copyright holder, and license. Add the complete license texts needed by the planned extraction. If authorship, permission, or a compatible license cannot be established for a planned file or block, exclude it and reimplement only from documented behavior; do not commit uncertain derived code. Commit this gate before the first derived implementation:

```bash
git add docs/provenance/brainvlm-midline-sources.md licenses
git commit -m "docs: record BrainVLM-Midline source provenance"
```

- [x] **Step 2: Write failing prompt and configuration tests**

Cover exact tumor inference prompt suffix `ASSISTANT:<tumor_`, the three tumor training instructions, the four WHO instructions, child/adult age sentences, canonical token vocabularies, fixed-grade constraints, strict unknown-label rejection, local asset validation, indexed CUDA syntax, and immutable expected checkpoint key/shape metadata.

```python
def test_tumor_inference_prompt_preserves_checkpoint_suffix():
    prompt = build_tumor_inference_prompt(age=None)
    assert prompt.endswith("ASSISTANT:<tumor_")

def test_age_context_uses_training_boundary():
    assert age_context(18) == "This is a child patient. "
    assert age_context(19) == "This is an adult patient. "
```

- [x] **Step 3: Run the tests and verify they fail**

Run: `pytest -q tests/diagnostic/test_brainvlm_prompts.py tests/diagnostic/test_brainvlm_config.py`

Expected: FAIL because the internal package does not exist.

- [x] **Step 4: Implement the focused constants and records**

Add frozen dataclasses for foundation assets, model settings, and training settings. Centralize prompt text and label conversions. Define the 146-entry checkpoint manifest from the authoritative checkpoint keys, without loading weight data into source.

- [x] **Step 5: Run focused tests**

Run the command from Step 2. Expected: PASS.

- [x] **Step 6: Recheck provenance and commit the package contract**

Compare every derived block actually added with the inventory. Update the modification statement and released-file digest. Stop before committing if any derived content is not covered by an established ownership/license record.

```bash
git add midline_agent/diagnostic/brainvlm_midline tests/diagnostic/test_brainvlm_prompts.py tests/diagnostic/test_brainvlm_config.py docs/provenance/brainvlm-midline-sources.md
git commit -m "feat: define internal BrainVLM-Midline contract"
```

### Task 3: Implement BiomedCLIP preprocessing and vision encoding

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_midline/vision.py`
- Create: `tests/diagnostic/test_brainvlm_vision.py`
- Modify: `midline_agent/datasets/brainvlm.py`

- [x] **Step 1: Write failing preprocessing tests**

Test bicubic 224-pixel RGB conversion, BiomedCLIP mean/std normalization, CHW float32 output, finite-value checks, local pretrained override handling, the final normalization hook, CLS removal, 196-patch mean pooling, and the 768-dimensional low-resolution result. Inject a tiny fake OpenCLIP trunk so CPU tests do not download weights.

- [x] **Step 2: Verify failure**

Run: `pytest -q tests/diagnostic/test_brainvlm_vision.py`

Expected: FAIL because `vision.py` is missing.

- [x] **Step 3: Implement the minimal vision unit**

Construct `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` through installed `open_clip`, register and remove the final-norm hook safely, validate the captured shape, and expose `preprocess_image()` plus `encode_low_resolution()`.

- [x] **Step 4: Verify focused and existing image tests**

Run: `pytest -q tests/diagnostic/test_brainvlm_vision.py tests/test_mri_dataset.py`

Expected: PASS.

- [x] **Step 5: Commit vision support**

```bash
git add midline_agent/diagnostic/brainvlm_midline/vision.py midline_agent/datasets/brainvlm.py tests/diagnostic/test_brainvlm_vision.py
git commit -m "feat: add BrainVLM-Midline vision encoder"
```

### Task 4: Implement checkpoint-compatible Llama adaptation and model assembly

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_midline/llama.py`
- Create: `midline_agent/diagnostic/brainvlm_midline/model.py`
- Create: `tests/diagnostic/test_brainvlm_model.py`
- Create: `tests/diagnostic/test_brainvlm_checkpoint.py`
- Modify: `midline_agent/diagnostic/checkpoints.py`

- [x] **Step 1: Write failing construction and state-layout tests**

Use injected tiny tokenizer, causal LM, and vision stubs. Assert special-token insertion order, pad token `$$`, embedding resize, LoRA rank 8/alpha 32/dropout 0.05 on `q_proj` and `v_proj`, frozen foundations, trainable module names, 768-to-4096-to-4096 projector, 768-to-1408 compatibility layer, cross-attention keys, inputs-embeds forward behavior, final-position hidden/logit extraction, and deterministic generation arguments. Also assert checkpoint-compatible supervised sample construction: image placeholder splitting, prompt and target token concatenation, BOS placement, attention masks, `-100` masking for image/prompt/padding positions, unmasked answer tokens including the end symbol, maximum-length truncation, causal-LM label alignment, scalar mean reduction, and unreduced per-sample loss behavior.

- [x] **Step 2: Verify failure**

Run: `pytest -q tests/diagnostic/test_brainvlm_model.py tests/diagnostic/test_brainvlm_checkpoint.py`

Expected: FAIL because the model modules are missing.

- [x] **Step 3: Implement the smallest compatible model**

Prefer a thin adaptation around pinned Transformers `LlamaForCausalLM`. Implement the source-compatible supervised `forward(samples, reduction="mean")` path as well as inference. It must form image/prompt/answer embeddings and masks exactly as characterized by the tests, pass labels to the causal LM, and implement only `mean` and `none` reductions with explicit validation. Preserve released state names exactly, including:

```text
llama_model.base_model.model.model.layers.*.*_proj.lora_[AB].weight
visual_encoder.norm.{weight,bias}
llama_proj_mlps.{0,2}.{weight,bias}
v_q_project.{weight,bias}
cross_attention.{Wq,Wk,Wv,output_linear,norm}.{weight,bias}
```

If the reference parity job later demonstrates that the pinned Transformers path changes forward or generation behavior, replace only the required Llama internals with the cleaned Apache-2.0 implementation and keep this public interface unchanged.

- [x] **Step 4: Implement strict checkpoint loading**

Accept only a mapping containing `model`; reject unknown keys, missing required manifest keys, incompatible shapes, non-tensors, and non-finite floating tensors. Permit foundation parameters absent from the trainable-only checkpoint. Return the sorted loaded key tuple and verify all 146 expected trainable tensors were loaded.

- [x] **Step 5: Run focused tests**

Run the command from Step 2. Expected: PASS.

- [x] **Step 6: Inspect real checkpoint manifests on CPU**

Load only metadata/state tensors for both authoritative checkpoints and assert exactly 146 matching keys and shapes. Expected: PASS without constructing Llama.

- [x] **Step 7: Recheck provenance and commit the internal model**

Update the provenance inventory for `llama.py` and `model.py`, including final release digests and the precise distinction between behaviorally reimplemented project code and any retained licensed code. Do not commit until all included derived material is covered.

```bash
git add midline_agent/diagnostic/brainvlm_midline/llama.py midline_agent/diagnostic/brainvlm_midline/model.py midline_agent/diagnostic/checkpoints.py tests/diagnostic/test_brainvlm_model.py tests/diagnostic/test_brainvlm_checkpoint.py
git commit -m "feat: add checkpoint-compatible BrainVLM-Midline model"
```

### Task 5: Replace the external runtime with the internal feature runtime

**Files:**
- Modify: `midline_agent/diagnostic/brainvlm_runtime.py`
- Modify: `midline_agent/diagnostic/__init__.py`
- Modify: `tests/test_brainvlm_runtime.py`
- Create: `tests/diagnostic/test_slice_features.py`
- Delete: `midline_agent/diagnostic/brainvlm_adapter.py`
- Delete: `tests/test_brainvlm_bridge.py`
- Delete: `tests/test_diagnostic_adapter.py`

- [x] **Step 1: Rewrite runtime tests against an internal factory**

Require `BrainVlmRuntime.load(llama_model, checkpoint, device, biomedclip_pretrained=None)` with no source root. Test `SliceFeatures(logits, hidden)`, one shared deterministic forward pass, exactly five class-token logits, 4096 hidden values, generation formatting, image decode errors, CUDA validation, and strict asset checks. Add a static test that forbids `sys.path` mutation and `minigpt4` imports in executable source.

- [x] **Step 2: Verify the rewritten tests fail**

Run: `pytest -q tests/test_brainvlm_runtime.py tests/diagnostic/test_slice_features.py`

Expected: FAIL because the current runtime requires an external source.

- [x] **Step 3: Implement the internal runtime**

Delete dynamic imports and source checkout validation. Construct `BrainVlmMidline` directly, load the strict trainable state, expose:

```python
def predict_slice_features(self, image_path, metadata) -> SliceFeatures: ...
def predict_tumor_logits(self, image_path, metadata) -> torch.Tensor:
    return self.predict_slice_features(image_path, metadata).logits
def predict_who_generation(self, image_path, metadata, tumor_type, age) -> str: ...
```

Keep `from_model()` for isolated contract tests.

- [ ] **Step 4: Remove the generic external adapter**

Delete the adapter implementation and bridge-specific tests. Update exports so callers can only use the internal runtime contract.

- [x] **Step 5: Run runtime and standalone service tests**

Run: `pytest -q tests/test_brainvlm_runtime.py tests/diagnostic/test_slice_features.py tests/test_standalone_diagnostic.py`

Expected: PASS.

- [ ] **Step 6: Commit the runtime replacement**

```bash
git add -A midline_agent/diagnostic tests/test_brainvlm_runtime.py tests/diagnostic tests/test_brainvlm_bridge.py tests/test_diagnostic_adapter.py
git commit -m "refactor: use internal BrainVLM-Midline runtime"
```

### Task 6: Extract strict tumor and WHO training datasets

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_midline/datasets.py`
- Create: `midline_agent/diagnostic/brainvlm_midline/collate.py`
- Create: `tests/diagnostic/test_brainvlm_datasets.py`
- Create: `tests/fixtures/brainvlm_training/annotations.json`

- [ ] **Step 1: Write failing seeded fixture tests**

Cover JSON object and JSONL inputs; `figure`; `who_class`/`WHO_class`; `who_grade`; `age`/`Age`; path containment; decodable images; finite age; canonical labels; WHO filtering to tumors 2, 4, and 5; three/four instruction pools; deterministic prompt selection; 224-pixel preprocessing; threshold upsampling counts; consistent patient labels; and reproducible 80:20 WHO/tumor stream schedules.

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/diagnostic/test_brainvlm_datasets.py`

Expected: FAIL because task datasets are missing.

- [ ] **Step 3: Implement strict record parsing and datasets**

Extract only the valid classification behavior from `tcia_classification_who` and `just_who_grade`. Do not copy recursive corrupt-image fallback, silent defaults, caption paths, segmentation, unrelated datasets, or private paths. Return typed task samples consumed by the collator.

- [ ] **Step 4: Implement deterministic mixing and collation**

Build seeded samplers for class upsampling and WHO 80:20 auxiliary mixing. The collator must reject mixed tensor shapes, missing targets, and non-finite images before model invocation.

- [ ] **Step 5: Run focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 6: Commit training data support**

```bash
git add midline_agent/diagnostic/brainvlm_midline/datasets.py midline_agent/diagnostic/brainvlm_midline/collate.py tests/diagnostic/test_brainvlm_datasets.py tests/fixtures/brainvlm_training
git commit -m "feat: add focused BrainVLM-Midline datasets"
```

### Task 7: Implement focused training, save, and resume

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_midline/training.py`
- Create: `tests/diagnostic/test_brainvlm_training.py`
- Create: `tests/diagnostic/ddp_smoke.py`
- Create: `scripts/train_brainvlm_midline.py`
- Create: `configs/brainvlm_tumor.example.yaml`
- Create: `configs/brainvlm_who.example.yaml`
- Delete: `scripts/train_brainvlm.sh`
- Delete: `configs/brainvlm_training.example.yaml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing component training tests**

Inject a tiny compatible model and dataset. Test parameter freezing, AdamW decay/no-decay grouping, linear warmup plus cosine decay, gradient accumulation, AMP scaler plumbing, seeded epochs, one optimizer step, trainable-only model state, checkpoint envelope fields, resume at the next epoch, and strict inference loading of the saved model envelope. Parameterize every approved configuration field with valid, missing, wrong-type, out-of-range, and unknown-field cases. Require the CLI `--task` value and configuration task to agree.

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/diagnostic/test_brainvlm_training.py`

Expected: FAIL because focused training is missing.

- [ ] **Step 3: Implement the training core**

Train only LoRA adapters, `visual_encoder.norm`, `llama_proj_mlps`, `v_q_project`, and `cross_attention`. Support single-GPU and single-node `torchrun` DDP, AMP, accumulation, deterministic sampling, and rank-zero checkpoint writing. Reject inference-only checkpoints passed to `--resume`.

- [ ] **Step 4: Implement the CLI and templates**

Add `midline-train-brainvlm = "scripts.train_brainvlm_midline:main"`. Accept `--task tumor|who --config PATH`. Parse only approved configuration fields, require caller-owned data/output paths, and reject unknown fields. Templates must contain placeholders and no server paths.

- [ ] **Step 5: Run a real two-process CPU DDP smoke test**

Implement `tests/diagnostic/ddp_smoke.py` as a small executable using injected foundation stubs, a two-record dataset, the `gloo` backend, and the public training entry point. Each rank writes an atomic rank marker to a temporary directory; only rank zero may write the checkpoint. Run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/diagnostic/ddp_smoke.py --output /tmp/midlineagent-ddp-smoke
```

Expected: exit code 0, two rank markers, one checkpoint, one optimizer step per rank, identical reduced loss, and no non-zero-rank checkpoint.

- [ ] **Step 6: Run component and CLI tests**

Run: `pytest -q tests/diagnostic/test_brainvlm_training.py tests/test_cli_entrypoints.py`

Expected: PASS.

- [ ] **Step 7: Commit focused training**

```bash
git add -A midline_agent/diagnostic/brainvlm_midline/training.py scripts/train_brainvlm_midline.py scripts/train_brainvlm.sh configs pyproject.toml tests/diagnostic/test_brainvlm_training.py tests/diagnostic/ddp_smoke.py tests/test_cli_entrypoints.py
git commit -m "feat: add self-contained BrainVLM-Midline training"
```

### Task 8: Make feature extraction use the internal runtime

**Files:**
- Modify: `midline_agent/diagnostic/feature_store.py`
- Modify: `scripts/extract_features.py`
- Modify: `tests/test_feature_store.py`
- Create: `tests/diagnostic/test_feature_extraction_cli.py`
- Modify: `docs/data-schema.md`

- [ ] **Step 1: Write failing feature-contract tests**

Require `hidden_and_logits`, logit dimension 5, hidden dimension 4096, ordered slice identity, four-value modality map, checkpoint digest, prompt version, preprocessing version, and label-free/labeled modes. Label-free input must reject diagnostic labels; labeled mode must require one consistent canonical tumor label and store its integer class.

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/test_feature_store.py tests/diagnostic/test_feature_extraction_cli.py`

Expected: FAIL because the current command uses a source root and adapter.

- [ ] **Step 3: Rewrite the feature command**

Accept `--annotations`, `--image-root`, `--llama-model`, optional `--biomedclip-pretrained`, `--checkpoint`, `--device`, `--output`, and `--labeled`. Construct `BrainVlmRuntime` directly and call `predict_slice_features()` once per slice.

- [ ] **Step 4: Update feature schemas and documentation**

Version the feature metadata and document how the same artifact feeds AttentionMIL and the Survival Sub-Agent without bundling generated features.

- [ ] **Step 5: Run feature tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 6: Commit internal feature extraction**

```bash
git add midline_agent/diagnostic/feature_store.py scripts/extract_features.py tests/test_feature_store.py tests/diagnostic/test_feature_extraction_cli.py docs/data-schema.md
git commit -m "feat: extract BrainVLM-Midline features internally"
```

### Task 9: Rewire standalone tumor and WHO commands and launchers

**Files:**
- Modify: `scripts/predict_tumor.py`
- Modify: `scripts/predict_who_grade.py`
- Modify: `scripts/run_tumor_prediction.sh`
- Modify: `scripts/run_who_grading.sh`
- Modify: `scripts/run_tumor_prediction.sbatch`
- Modify: `scripts/run_who_grading.sbatch`
- Modify: `tests/test_cli_entrypoints.py`
- Modify: `tests/test_launchers.py`
- Modify: `tests/test_standalone_diagnostic.py`

- [ ] **Step 1: Rewrite failing CLI tests**

Assert both commands accept Llama, optional local BiomedCLIP, checkpoint, indexed CUDA, input, and output assets; neither accepts a source root or model factory. Verify tumor calls the released `LogitsMilPredictor`, WHO reuses the shared runtime, fixed-grade rules skip checkpoint construction, and all public result metadata remains stable.

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/test_cli_entrypoints.py tests/test_launchers.py tests/test_standalone_diagnostic.py`

Expected: FAIL while external-source arguments remain.

- [ ] **Step 3: Rewire commands and launchers**

Construct the internal runtime through one helper. Replace source variables with `LLAMA_MODEL`, optional `BIOMEDCLIP_PRETRAINED`, checkpoint variables, and `DEVICE`. Keep caller-owned output paths and Slurm logs outside the repository.

- [ ] **Step 4: Run standalone tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit standalone inference**

```bash
git add scripts/predict_tumor.py scripts/predict_who_grade.py scripts/run_tumor_prediction.sh scripts/run_who_grading.sh scripts/run_tumor_prediction.sbatch scripts/run_who_grading.sbatch tests/test_cli_entrypoints.py tests/test_launchers.py tests/test_standalone_diagnostic.py
git commit -m "refactor: run diagnostic checkpoints from internal source"
```

### Task 10: Rewire the Diagnostic workflow and orchestrator

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `midline_agent/diagnostic/workflow.py`
- Modify: `midline_agent/orchestrator.py`
- Modify: `tests/test_diagnostic_workflow.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/diagnostic/test_pipeline_diagnostic_assets.py`
- Create: `tests/acceptance/run_installed_diagnostic.py`
- Create: `test_data/utsw_glioma_bt0156/case.json`

- [ ] **Step 1: Write failing shared-service workflow tests**

Assert `midline-run` constructs the same `TumorPredictionService`, `WhoGradePredictionService`, internal runtime, and `LogitsMilPredictor` as standalone inference, then converts their validated outputs to `DiagnosticResult`. Assert Survival remains wired as currently released and Treatment uses the existing deterministic client in tests. No AttentionMIL adapter or external factory may be constructed. Add a label-free public `case.json` containing only the BT0156 case identifier and an empty public clinical/MRI manifest sufficient to exercise the public Diagnostic CLI; reject diagnostic labels in this fixture. Create `tests/acceptance/run_installed_diagnostic.py` as an executable asset-driven runner with explicit arguments for annotations, images, foundation assets, three checkpoints, device, output, and expected labels. It must assert all `midline_agent` module origins are beneath the active environment, inject deterministic non-network Survival/Treatment components, execute the public orchestrator, and validate the privacy-minimized trace.

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/test_diagnostic_workflow.py tests/test_orchestrator.py tests/diagnostic/test_pipeline_diagnostic_assets.py`, then run `python tests/acceptance/run_installed_diagnostic.py --help`.

Expected: FAIL because `run_pipeline.py` still uses the generic adapter path.

- [ ] **Step 3: Implement shared diagnostic orchestration**

Accept the same diagnostic model assets as standalone commands, invoke tumor then WHO services, preserve trace minimization, and surface missing assets as an explicit unavailable state only when the Diagnostic Sub-Agent was not requested/configured. Never silently substitute a different predictor.

- [ ] **Step 4: Run workflow tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit orchestration changes**

```bash
git add scripts/run_pipeline.py midline_agent/diagnostic/workflow.py midline_agent/orchestrator.py tests/test_diagnostic_workflow.py tests/test_orchestrator.py tests/diagnostic/test_pipeline_diagnostic_assets.py tests/acceptance/run_installed_diagnostic.py test_data/utsw_glioma_bt0156/case.json
git commit -m "refactor: share diagnostic checkpoint services in orchestrator"
```

### Task 11: Add exact dependencies, provenance, licenses, and user documentation

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements/diagnostic-reference.txt`
- Create: `docs/brainvlm-midline.md`
- Create: `docs/provenance/brainvlm-midline-sources.md`
- Create: `licenses/BSD-3-Clause-MiniGPT4.txt`
- Create: `licenses/BSD-3-Clause-LAVIS.txt`
- Create: `licenses/Apache-2.0.txt`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `README.md`
- Modify: `.env.example`
- Delete: `docs/external-brainvlm-adapter.md`
- Delete: `docs/superpowers/specs/2026-11-27-checkpoint-inference-design.md`
- Delete: `docs/superpowers/plans/2026-11-27-checkpoint-inference.md`

- [ ] **Step 1: Reconcile the pre-copy source provenance inventory**

Verify the Task 2 pre-copy inventory now matches every final derived release file. Record final release digests and any additional local source digest, identifiable upstream repository and immutable revision, modifications, holder, and license. When local Git lineage is unavailable, state that fact and retain the immutable local digest instead of inventing a revision. Remove any code whose ownership or compatible license still cannot be established.

- [ ] **Step 2: Add dependency-contract tests**

Extend CLI and release tests to assert the diagnostic extra includes compatible Transformers, PEFT, OpenCLIP, OmegaConf, einops, Pillow, NumPy, safetensors, PyTorch, and torchvision constraints. Assert the exact reference versions are recorded separately and CUDA wheel selection is documented as platform-specific.

- [ ] **Step 3: Update README from the MidlineAgent perspective**

Explain BrainVLM-Midline as an internally implemented Diagnostic Sub-Agent model derived from related work. Keep the supplied Google Drive checkpoint links, Llama and BiomedCLIP preparation, BT0156 commands, feature extraction, training, Survival, Treatment private-evidence boundary, official OpenAI-only API use, and acknowledgements. Remove all instructions to clone another source repository.

- [ ] **Step 4: Add complete notices and runtime documentation**

Document annotation fields, model assets, training scope, checkpoint hashes, feature schema, supported commands, and intentional exclusions. Include full license texts and precise attribution without presenting unrelated comparison implementations.

- [ ] **Step 5: Remove obsolete external-boundary documents**

Delete the old external adapter and external-checkout checkpoint plan/spec after their replacements exist. The approved self-contained design and this implementation plan remain as historical engineering documents.

- [ ] **Step 6: Run documentation and packaging tests**

Run: `pytest -q tests/test_cli_entrypoints.py tests/test_release_audit.py`

Expected: PASS.

- [ ] **Step 7: Commit documentation and provenance**

```bash
git add -A README.md pyproject.toml requirements docs licenses THIRD_PARTY_NOTICES.md .env.example tests
git commit -m "docs: publish self-contained diagnostic runtime contract"
```

### Task 12: Harden the public release audit

**Files:**
- Modify: `scripts/release_audit.py`
- Modify: `tests/test_release_audit.py`
- Modify: `release-manifest.txt`

- [ ] **Step 1: Write failing policy tests**

Add fixtures that reject every non-ASCII byte in released text files, private absolute paths, credentials, generated results, unapproved binary weights/data, custom OpenAI endpoints, relay names, executable `minigpt4` imports, `sys.path`/`PYTHONPATH` source injection, dynamic model factories, removed source-root arguments/environment variables, and delegation to another project's training entrypoint. Binary formats are handled by the existing explicit allowlist (the attributed BT0156 PNG directory is the only data exception). Allow removed names only in the two approved policy/design documents where they appear as explicit prohibitions.

- [ ] **Step 2: Verify policy-test failure**

Run: `pytest -q tests/test_release_audit.py`

Expected: FAIL until the new source-boundary rules are implemented.

- [ ] **Step 3: Implement narrow source-boundary scanning**

Use AST inspection for imports, `sys.path` mutation, dynamic import calls, subprocess training delegation, and CLI options. Use text scanning only for configuration and shell variables. Keep exemptions path- and rule-specific.

- [ ] **Step 4: Regenerate the manifest from tracked files**

Run: `git ls-files | LC_ALL=C sort > /tmp/midlineagent-release-manifest.txt`, then update `release-manifest.txt` through `apply_patch` so the manifest exactly matches the intended tracked release after staged deletions/additions.

- [ ] **Step 5: Run the complete audit**

Run: `python scripts/release_audit.py --tracked-only --root .`

Expected: exit code 0 and no violations.

- [ ] **Step 6: Commit release guards**

```bash
git add scripts/release_audit.py tests/test_release_audit.py release-manifest.txt
git commit -m "test: enforce self-contained public source boundary"
```

### Task 13: Run the complete CPU suite and build isolated distributions

**Files:**
- Modify as failures require: only files already owned by Tasks 2-12
- Create outside Git: `.runtime/dist/`
- Create outside Git: `.runtime/venvs/self-contained-acceptance/`

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass. Investigate each failure with `superpowers:systematic-debugging`; do not weaken assertions to obtain a green run.

- [ ] **Step 2: Build wheel and source distribution**

Run: `python -m build --outdir ../dist`

Expected: one wheel and one source distribution created outside the publication tree.

- [ ] **Step 3: Inspect distribution contents**

Run: `python -m zipfile -l ../dist/midline_agent-*.whl` and inspect the sdist file list.

Expected: internal BrainVLM-Midline source, scripts, and required notices are present; checkpoints, outputs, caches, and external source are absent.

- [ ] **Step 4: Create an isolated acceptance environment**

Create the environment outside the checkout, unset `PYTHONPATH` and removed source variables, install the built wheel plus declared dependencies, and do not add the checkout or MiniGPT-4 directory to import search paths.

- [ ] **Step 5: Audit module origins**

Run a script that imports all `midline_agent` and project model modules and asserts their resolved origins are inside the isolated environment's site-packages. Third-party imports must resolve to installed distributions.

- [ ] **Step 6: Run installed-package CPU contract tests**

Invoke every console command with `--help`, validate bundled schema readers against BT0156 annotations, and run stubbed Diagnostic orchestration from the installed package.

- [ ] **Step 7: Commit any packaging corrections**

If corrections were needed, rerun Steps 1-6 and commit only the scoped packaging fixes:

```bash
git add pyproject.toml midline_agent scripts tests docs
git commit -m "fix: complete isolated MidlineAgent installation"
```

### Task 14: Run real-checkpoint BT0156 parity and iterate to exact acceptance

**Files:**
- Create outside Git: `.runtime/runs/self-contained-acceptance/`
- Modify as parity failures require: internal diagnostic files and their tests only

- [ ] **Step 0: Define the isolated acceptance assets**

From a caller-owned directory outside the publication checkout, define and validate every path explicitly:

```bash
export REPOSITORY_ROOT=/path/to/MidlineAgent
export ACCEPTANCE_ROOT=/path/to/midlineagent-acceptance
export VENV=/path/to/midlineagent-acceptance-venv
export ANNOTATIONS="$REPOSITORY_ROOT/test_data/utsw_glioma_bt0156/annotations.json"
export IMAGE_ROOT="$REPOSITORY_ROOT/test_data/utsw_glioma_bt0156"
export CASE_JSON="$REPOSITORY_ROOT/test_data/utsw_glioma_bt0156/case.json"
export LLAMA_MODEL=/path/to/Llama-3.1-8B-Instruct
export BIOMEDCLIP_PRETRAINED=/path/to/biomedclip-pretrained.bin
export TUMOR_CHECKPOINT=/path/to/brainvlm-midline.pth
export MIL_CHECKPOINT=/path/to/mil.pth
export WHO_CHECKPOINT=/path/to/who-grading.pth
export DEVICE=cuda:0
mkdir -p "$ACCEPTANCE_ROOT/logs"
unset PYTHONPATH BRAINVLM_SOURCE BRAINVLM_SOURCE_ROOT
```

Assert the repository, virtual environment, and acceptance root resolve to three distinct directories; all model assets have the expected filesystem type; and the three released checkpoint SHA-256 values match the approved specification.

- [ ] **Step 1: Submit installed-wheel tumor and feature jobs**

Using the isolated environment, authoritative Llama/BiomedCLIP/tumor/MIL assets, and no project-source checkout on the import path, run:

```bash
cd "$ACCEPTANCE_ROOT"
"$VENV/bin/midline-extract-features" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$TUMOR_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$ACCEPTANCE_ROOT/bt0156-features.safetensors"
"$VENV/bin/midline-predict-tumor" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$TUMOR_CHECKPOINT" \
  --mil-checkpoint "$MIL_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$ACCEPTANCE_ROOT/tumor.json"
```

Expected: both jobs exit 0; the feature artifact contains 30-by-5 logits and 30-by-4096 hidden vectors; tumor output is `tumor_4`.

- [ ] **Step 2: Compare tumor tensors to the private reference**

Assert identical ordered slices and shapes. Compare slice logits, hidden vectors, aligned 20-by-5 input, patient logits, and probabilities with `torch.testing.assert_close(rtol=1e-4, atol=1e-4)`. Compare final label exactly.

- [ ] **Step 3: Debug any parity failure at the earliest divergent boundary**

Use `superpowers:systematic-debugging`. Compare in this order: normalized pixels, vision hook output, pooled 768-vector, projected image embedding, token IDs, prompt embeddings, final hidden state, class logits, aligned MIL input, patient logits. If the pinned Transformers subclass is the cause, first add the proposed Apache-derived path, local source digest, upstream revision, holder, license, and modification statement to the provenance inventory and verify the full Apache-2.0 text is present. Exclude the code if that gate cannot be satisfied. Only then migrate the minimum required behavior, update the final released-file digest, add a regression test, and rerun parity.

- [ ] **Step 4: Submit installed-wheel WHO job**

Run from `ACCEPTANCE_ROOT`:

```bash
"$VENV/bin/midline-predict-who-grade" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$WHO_CHECKPOINT" \
  --tumor-result "$ACCEPTANCE_ROOT/tumor.json" \
  --device "$DEVICE" \
  --output "$ACCEPTANCE_ROOT/who.json"
```

Expected: exit code 0 and validated output `grade_3`.

- [ ] **Step 5: Compare WHO output to the private reference**

Require exact per-slice deterministic generations, accepted/rejected votes, invalid count, vote counts, and final grade. Debug prompt tokenization, stopping, and generation arguments at the first difference.

- [ ] **Step 6: Submit installed-wheel Diagnostic orchestration**

First run the installed public command with only the Diagnostic stage configured:

```bash
cd "$ACCEPTANCE_ROOT"
"$VENV/bin/midline-run" \
  --case-json "$CASE_JSON" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --tumor-checkpoint "$TUMOR_CHECKPOINT" \
  --mil-checkpoint "$MIL_CHECKPOINT" \
  --who-checkpoint "$WHO_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$ACCEPTANCE_ROOT/case-trace.json"
```

Then run the installed acceptance driver by absolute repository path while retaining `ACCEPTANCE_ROOT` as the working directory:

```bash
cd "$ACCEPTANCE_ROOT"
"$VENV/bin/python" "$REPOSITORY_ROOT/tests/acceptance/run_installed_diagnostic.py" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --case-json "$CASE_JSON" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --tumor-checkpoint "$TUMOR_CHECKPOINT" \
  --mil-checkpoint "$MIL_CHECKPOINT" \
  --who-checkpoint "$WHO_CHECKPOINT" \
  --device "$DEVICE" \
  --expected-tumor tumor_4 \
  --expected-grade grade_3 \
  --output "$ACCEPTANCE_ROOT/installed-orchestration-trace.json"
```

The driver must assert every `midline_agent` module originates inside `VENV`, inject the released deterministic Survival/Treatment components without network or private evidence, and validate the privacy-minimized trace. Expected: both routes exit 0; Diagnostic completes with `tumor_4` and `grade_3`; unavailable non-Diagnostic stages are explicit; the deterministic route emits a valid trace.

- [ ] **Step 7: Run launchers exactly as documented**

Run both shell launchers from the publication repository with the variables from Step 0 plus `PROJECT_ROOT="$REPOSITORY_ROOT"` and `OUTPUT_JSON` set beneath `ACCEPTANCE_ROOT`. Assert each exit code and validate its JSON before continuing.

Submit and monitor the Slurm launchers with caller-owned logs:

```bash
export PROJECT_ROOT="$REPOSITORY_ROOT"
export OUTPUT_JSON="$ACCEPTANCE_ROOT/slurm-tumor.json"
tumor_job=$(sbatch --parsable --output="$ACCEPTANCE_ROOT/logs/tumor-%j.log" "$REPOSITORY_ROOT/scripts/run_tumor_prediction.sbatch")
while squeue -h -j "$tumor_job" | grep -q .; do sleep 10; done
sacct -n -X -j "$tumor_job" --format=State,ExitCode
test "$(sacct -n -X -j "$tumor_job" --format=State,ExitCode | awk 'NF {print $1, $2; exit}')" = "COMPLETED 0:0"

export TUMOR_RESULT="$ACCEPTANCE_ROOT/slurm-tumor.json"
export OUTPUT_JSON="$ACCEPTANCE_ROOT/slurm-who.json"
who_job=$(sbatch --parsable --output="$ACCEPTANCE_ROOT/logs/who-%j.log" "$REPOSITORY_ROOT/scripts/run_who_grading.sbatch")
while squeue -h -j "$who_job" | grep -q .; do sleep 10; done
sacct -n -X -j "$who_job" --format=State,ExitCode
test "$(sacct -n -X -j "$who_job" --format=State,ExitCode | awk 'NF {print $1, $2; exit}')" = "COMPLETED 0:0"
```

Expected: both jobs reach `COMPLETED 0:0`; schema validation passes; the Slurm outputs exactly match standalone `tumor_4` and `grade_3`. Any `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, missing accounting row, non-zero exit code, or missing output is a failed acceptance step and must be debugged before proceeding.

- [ ] **Step 8: Commit parity fixes only after all jobs pass**

For each discovered defect, first add a focused regression test, then commit the minimal fix. Do not commit generated features, predictions, Slurm logs, or reference tensors.

### Task 15: Final verification and release readiness review

**Files:**
- Modify: `release-manifest.txt` only if the final tracked set changed
- Inspect: all tracked files

- [ ] **Step 1: Run verification commands from a clean worktree**

```bash
pytest -q
python scripts/release_audit.py --tracked-only --root .
git diff --check
git status --short
```

Expected: tests and audit pass, no whitespace errors, and the worktree is clean.

- [ ] **Step 2: Run targeted forbidden-content searches**

Search tracked text files for any non-ASCII byte, and search all tracked candidates for private path roots, credentials, source-root flags, source-injection variables, dynamic MiniGPT-4 imports, custom OpenAI endpoints, relay names, and checkpoint/data/result artifacts. Expected: all released text is English ASCII; no executable or user-instruction violation exists; only explicit prohibition text in approved policy documents may match removed terms.

- [ ] **Step 3: Rebuild and reinstall the final wheel**

Delete only the prior temporary acceptance environment and distributions, rebuild, reinstall, rerun module-origin checks, and repeat BT0156 tumor, WHO, feature, and Diagnostic workflow acceptance from the final commit.

- [ ] **Step 4: Request independent code review**

Use `superpowers:requesting-code-review` with the approved specification, this plan, the full local commit range, CPU test output, release audit output, build inspection, module-origin report, and GPU parity evidence. Resolve every High or Medium finding with tests and repeat review.

- [ ] **Step 5: Confirm release boundary**

Verify the repository contains source, documentation, configuration templates, notices, and the attributed BT0156 images only. It must not contain private treatment evidence, internal databases, credentials, checkpoints, generated features, predictions, result files, comparison implementations, or any dependency on another project-source checkout.

- [ ] **Step 6: Leave remote publication to the user-controlled handoff**

Report the final local branch and commit range. Do not push until the user explicitly requests the GitHub update after their README edits.
