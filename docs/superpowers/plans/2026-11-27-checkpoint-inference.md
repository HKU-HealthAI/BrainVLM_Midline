# Checkpoint-Backed Diagnostic Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently runnable tumor-prediction and WHO-grading commands that execute the released BrainVLM-Midline, MIL, and WHO checkpoints on an attributed public MRI case.

**Architecture:** Keep BrainVLM model source and all weights external. Add a first-party runtime bridge for the external checkout, a label-free inference annotation contract, a narrow loader for the released learnable-logits MIL checkpoint, and two standalone services and CLIs. Run tumor and WHO models in separate processes so only one 8B model occupies the GPU at a time.

**Tech Stack:** Python 3.9+ runtime compatibility, PyTorch, Pillow, NumPy, external BrainVLM/MiniGPT-4, pytest, POSIX shell, Slurm, JSON, TCIA UTSW-Glioma CC BY 4.0 data.

---

## File Map

- `midline_agent/datasets/inference_annotations.py`: label-free single-patient MRI input validation.
- `midline_agent/diagnostic/logits_mil.py`: exact released MIL alignment, checkpoint loading, and patient prediction.
- `midline_agent/diagnostic/standalone.py`: framework-independent tumor and WHO stage services and result schemas.
- `midline_agent/diagnostic/brainvlm_runtime.py`: external BrainVLM construction and real slice inference.
- `scripts/predict_tumor.py`: tumor CLI.
- `scripts/predict_who_grade.py`: WHO CLI.
- `scripts/run_tumor_prediction.sh`, `scripts/run_who_grading.sh`: portable launchers.
- `scripts/run_tumor_prediction.sbatch`, `scripts/run_who_grading.sbatch`: Slurm launchers.
- `test_data/utsw_glioma_bt0156/`: attributed label-free public smoke-test input.
- `tests/test_inference_annotations.py`: inference input tests.
- `tests/test_logits_mil.py`: alignment, arithmetic, and checkpoint tests.
- `tests/test_standalone_diagnostic.py`: stage orchestration and result tests.
- `tests/test_brainvlm_runtime.py`: runtime bridge tests without allocating an 8B model.
- `tests/test_public_case.py`: public case integrity and attribution tests.
- `tests/test_launchers.py`: shell and Slurm interface tests.
- `README.md`, `docs/data-schema.md`, `docs/external-brainvlm-adapter.md`, `THIRD_PARTY_NOTICES.md`: public setup and provenance documentation.
- `scripts/release_audit.py`, `tests/test_release_audit.py`, `release-manifest.txt`, `.gitignore`: narrowly scoped public-data exception and release controls.

### Task 1: Add the label-free inference input contract

**Files:**
- Create: `midline_agent/datasets/inference_annotations.py`
- Create: `tests/test_inference_annotations.py`
- Modify: `midline_agent/datasets/__init__.py`

- [ ] **Step 1: Write failing tests for valid single-patient annotations**

Test that relative PNG paths, normalized modalities, slice indices, and one
consistent finite age produce an immutable `InferenceCase` ordered by modality
and slice index.

- [ ] **Step 2: Write failing tests for unsafe or ambiguous inputs**

Cover absolute paths, path traversal, missing files, duplicate slice identity,
multiple patients, inconsistent ages, unsupported modalities, non-finite age,
and an undecodable image.

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `pytest tests/test_inference_annotations.py -q`

Expected: collection or import failure because the new loader is absent.

- [ ] **Step 4: Implement the minimal loader**

Use a frozen `InferenceSlice` dataclass and frozen `InferenceCase` dataclass.
Require `patient_id`, `image_path`, `modality`, and `slice_index`; make `age`
optional. Resolve each image beneath `image_root`, verify it with Pillow, and
reject rather than substitute invalid inputs.

- [ ] **Step 5: Run focused and existing annotation tests**

Run: `pytest tests/test_inference_annotations.py tests/test_annotations.py tests/test_mri_dataset.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add midline_agent/datasets tests/test_inference_annotations.py
git commit -m "feat: validate standalone diagnostic inputs"
```

### Task 2: Support the released MIL checkpoint exactly

**Files:**
- Create: `midline_agent/diagnostic/logits_mil.py`
- Create: `tests/test_logits_mil.py`
- Modify: `midline_agent/diagnostic/__init__.py`

- [ ] **Step 1: Write failing tests for modality alignment**

Specify `T1`, `T1CE`, `T2`, `FLAIR` order, five instances per modality,
center selection when more than five are present, cyclic repetition when one
to four are present, and failure when a modality is absent.

- [ ] **Step 2: Write failing tests for exact MIL arithmetic**

For a 20 by 5 logit matrix, assert:

```python
patient_logits = (slice_logits * class_weights).mean(dim=0) / temperature
probabilities = torch.softmax(patient_logits, dim=0)
```

Also assert canonical `tumor_1` through `tumor_5` mapping.

- [ ] **Step 3: Write failing checkpoint-safety tests**

Use temporary `.pth` fixtures and require `torch.load(..., weights_only=True)`.
Reject missing keys, extra model keys, wrong shapes, non-finite tensors,
non-positive temperature, and non-file paths.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `pytest tests/test_logits_mil.py -q`

Expected: import failure because `logits_mil.py` does not exist.

- [ ] **Step 5: Implement alignment and `LogitsMilPredictor`**

Keep this loader separate from `AttentionMilPredictor`; do not reinterpret the
released two-parameter checkpoint as the self-attention architecture.

- [ ] **Step 6: Run focused and existing MIL tests**

Run: `pytest tests/test_logits_mil.py tests/test_attention_mil.py tests/test_mil_training.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add midline_agent/diagnostic tests/test_logits_mil.py
git commit -m "feat: load released logits MIL checkpoint"
```

### Task 3: Add standalone tumor and WHO services

**Files:**
- Create: `midline_agent/diagnostic/standalone.py`
- Create: `tests/test_standalone_diagnostic.py`
- Modify: `midline_agent/diagnostic/who_grade.py`

- [ ] **Step 1: Write failing tumor-service tests with a fake runtime**

The fake runtime returns known five-value slice logits. Assert one call per
slice, exact MIL prediction, deterministic JSON-compatible fields, SHA-256
checkpoint fingerprints, and no local path disclosure.

- [ ] **Step 2: Write failing WHO rule tests**

Assert that `tumor_1` and `tumor_3` return `grade_4` without constructing or
calling a WHO runtime.

- [ ] **Step 3: Write failing WHO inference and voting tests**

For `tumor_2`, `tumor_4`, and `tumor_5`, assert one generation per slice,
strict constrained parsing, rejection of mismatched tumor outputs, explicit
failure when no matching vote remains, and deterministic grade tie handling.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `pytest tests/test_standalone_diagnostic.py -q`

Expected: import failure because the services are absent.

- [ ] **Step 5: Implement immutable result schemas and services**

Define protocols for `predict_tumor_logits(image, metadata)` and
`predict_who_generation(image, metadata, tumor_type, age)`. Keep model details
outside the services. Exclude raw paths and raw generated text from persisted
results while retaining aggregate provenance counts.

- [ ] **Step 6: Run focused and existing WHO/workflow tests**

Run: `pytest tests/test_standalone_diagnostic.py tests/test_who_grade.py tests/test_diagnostic_workflow.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add midline_agent/diagnostic tests/test_standalone_diagnostic.py
git commit -m "feat: add independent diagnostic stage services"
```

### Task 4: Implement the external BrainVLM runtime bridge

**Files:**
- Create: `midline_agent/diagnostic/brainvlm_runtime.py`
- Create: `tests/test_brainvlm_runtime.py`
- Modify: `docs/external-brainvlm-adapter.md`

- [ ] **Step 1: Write failing source and configuration tests**

Reject missing or escaped source roots, missing Llama paths, missing
checkpoints, CPU-only execution, unsupported device identifiers, and external
packages resolved outside the supplied source checkout.

- [ ] **Step 2: Write failing checkpoint compatibility tests**

Use a small fake external model. Assert weights-only loading, rejection of
unrecognized checkpoint state keys or incompatible shapes, and acceptance of
the expected `model` state mapping without reading embedded training config.

- [ ] **Step 3: Write failing inference tests with a fake model**

Assert CLIP mean/std normalization, five tumor-token logits in canonical
order, finite outputs, deterministic WHO generation arguments, and explicit
failure on empty or malformed model output.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `pytest tests/test_brainvlm_runtime.py -q`

Expected: import failure because the concrete runtime is absent.

- [ ] **Step 5: Implement model construction**

Temporarily import `minigpt4` from `source_root`, construct `MiniGPT4` with an
explicit Llama path and 2D settings, optionally override BiomedCLIP pretrained
resolution during construction, load only the checkpoint `model` state, move
to the requested CUDA device, set evaluation mode, and restore import/process
state after construction.

- [ ] **Step 6: Implement tumor logits and WHO generation**

Use the public fixed prompts, the model's image encoder/projector/tokenizer,
and greedy decoding. Keep one runtime instance bound to one task checkpoint.

- [ ] **Step 7: Run bridge and adapter tests**

Run: `pytest tests/test_brainvlm_runtime.py tests/test_brainvlm_bridge.py tests/test_diagnostic_adapter.py -q`

Expected: PASS without allocating the real 8B model.

- [ ] **Step 8: Commit**

```bash
git add midline_agent/diagnostic/brainvlm_runtime.py tests/test_brainvlm_runtime.py docs/external-brainvlm-adapter.md
git commit -m "feat: add concrete BrainVLM inference bridge"
```

### Task 5: Add independent command-line interfaces

**Files:**
- Create: `scripts/predict_tumor.py`
- Create: `scripts/predict_who_grade.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Write failing CLI help and validation tests**

Require explicit annotations, image root, BrainVLM source, Llama path,
checkpoint, MIL checkpoint where applicable, output, and CUDA device. WHO must
require exactly one of tumor-result JSON or explicit tumor type.

- [ ] **Step 2: Write failing service-wiring tests**

Monkeypatch the runtime and services, execute each `main`, and assert atomic
JSON output, non-zero failure behavior, and no OpenAI configuration dependency.

- [ ] **Step 3: Run CLI tests and verify RED**

Run: `pytest tests/test_cli_entrypoints.py -q`

Expected: failures because the new entry points do not exist.

- [ ] **Step 4: Implement both CLIs and package entry points**

Write output to a temporary file in the destination directory, validate JSON
serialization with `allow_nan=False`, then replace the destination only after
the stage completes.

- [ ] **Step 5: Run CLI tests**

Run: `pytest tests/test_cli_entrypoints.py tests/test_standalone_diagnostic.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/predict_tumor.py scripts/predict_who_grade.py pyproject.toml tests/test_cli_entrypoints.py
git commit -m "feat: expose tumor and WHO prediction commands"
```

### Task 6: Add and license the public test case

**Files:**
- Create: `test_data/utsw_glioma_bt0156/annotations.json`
- Create: `test_data/utsw_glioma_bt0156/ATTRIBUTION.md`
- Create: `test_data/utsw_glioma_bt0156/images/*.png`
- Create: `tests/test_public_case.py`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `scripts/release_audit.py`
- Modify: `tests/test_release_audit.py`

- [ ] **Step 1: Write failing public-case integrity tests**

Require exactly 30 listed images, ASCII relative paths, all four modalities,
successful image decoding, no ground-truth or generated-result fields, and a
complete TCIA citation, DOI, collection URL, CC BY 4.0 URL, and modification
notice.

- [ ] **Step 2: Write a failing release-audit allowlist test**

Prove that PNG is rejected everywhere except the exact attributed test-case
image directory, and that other binary or patient-data paths remain rejected.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/test_public_case.py tests/test_release_audit.py -q`

Expected: failures because the case and narrow exception are absent.

- [ ] **Step 4: Copy and normalize the 30 selected BT0156 slices**

Build annotations from the source keyframe JSON, rename files to short ASCII
names while retaining modality and slice index, and omit source labels and
precomputed predictions.

- [ ] **Step 5: Add attribution and narrow audit exception**

Attribute Reddy et al., TCIA DOI `10.7937/DFAE-1B86`, link CC BY 4.0, and
state that files are selected 2D PNG derivatives.

- [ ] **Step 6: Run public-case and release-audit tests**

Run: `pytest tests/test_public_case.py tests/test_release_audit.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add test_data tests/test_public_case.py THIRD_PARTY_NOTICES.md scripts/release_audit.py tests/test_release_audit.py
git commit -m "data: add attributed UTSW diagnostic test case"
```

### Task 7: Add shell and Slurm launchers

**Files:**
- Create: `scripts/run_tumor_prediction.sh`
- Create: `scripts/run_who_grading.sh`
- Create: `scripts/run_tumor_prediction.sbatch`
- Create: `scripts/run_who_grading.sbatch`
- Create: `tests/test_launchers.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing launcher tests**

Run `bash -n` on all launchers. Assert strict shell mode, quoted variables,
explicit required-variable checks, separate output paths, no private absolute
paths, no credentials, no custom endpoints, no email address, and no forced
`CUDA_VISIBLE_DEVICES` conflicting with Slurm.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_launchers.py -q`

Expected: failures because the launchers are absent.

- [ ] **Step 3: Implement portable shell launchers**

Accept documented environment variables and forward them to the installed
CLI. The WHO launcher reads the tumor JSON by default but permits an explicit
canonical tumor class.

- [ ] **Step 4: Implement separate Slurm launchers**

Request one GPU, appropriate memory and time, create caller-owned log/run
directories, and invoke the corresponding shell launcher. Do not embed
machine-specific project paths.

- [ ] **Step 5: Run launcher tests**

Run: `pytest tests/test_launchers.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_*prediction* scripts/run_who_grading* tests/test_launchers.py .gitignore
git commit -m "feat: add portable diagnostic launchers"
```

### Task 8: Document the reproducible workflow and release boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/data-schema.md`
- Modify: `docs/external-brainvlm-adapter.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `release-manifest.txt`
- Modify: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Write failing documentation assertions**

Require the three checkpoint download links, external BrainVLM checkout,
Llama and BiomedCLIP preparation, public-case citation, independent tumor and
WHO commands, shell and Slurm usage, WHO rule behavior, expected BT0156 labels,
and a statement that no generated result is bundled.

- [ ] **Step 2: Run documentation tests and verify RED**

Run: `pytest tests/test_cli_entrypoints.py tests/test_public_case.py -q`

Expected: failures for missing public instructions.

- [ ] **Step 3: Update the public documentation**

Replace the placeholder external-factory instructions for the supported
checkpoint path with concrete commands. Keep the generic factory contract for
custom adapters. Correct the release boundary to acknowledge the attributed
public case and describe the released MIL architecture accurately.

- [ ] **Step 4: Regenerate the explicit release manifest**

List every intended tracked file, including all 30 images, launchers, tests,
plans, and specifications. Do not add generated outputs or logs.

- [ ] **Step 5: Run documentation and audit tests**

Run: `pytest tests/test_cli_entrypoints.py tests/test_public_case.py tests/test_release_audit.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md docs THIRD_PARTY_NOTICES.md release-manifest.txt tests
git commit -m "docs: publish checkpoint inference workflow"
```

### Task 9: Verify real checkpoints on Slurm and stabilize the case

**Files:**
- Modify as needed: runtime, services, launchers, tests, and selected
  `test_data/` case only.
- Do not commit: run JSON, feature artifacts, logs, checkpoints, or scheduler
  metadata.

- [ ] **Step 1: Submit the tumor prediction job**

Use the existing BrainVLM environment and local source checkout with:

- tumor checkpoint `checkpoint_1039.pth`;
- MIL checkpoint `best_model.pth`;
- Llama 3.1-8B Instruct local snapshot; and
- the candidate public case.

- [ ] **Step 2: Monitor the job to a terminal state**

Inspect scheduler state and logs. A successful exit without a valid result is
not sufficient. Confirm 30 actual slice calls, a loaded MIL checkpoint, finite
patient probabilities, and `tumor_4` for BT0156.

- [ ] **Step 3: Diagnose any failure systematically**

For source incompatibility, device placement, dependency, memory, prompt, or
checkpoint mismatch, add a reproducing test before changing production code.
Repeat submission until the tumor command succeeds honestly.

- [ ] **Step 4: Submit WHO grading after tumor success**

Use the tumor result as input and WHO checkpoint `checkpoint_1052.pth`.

- [ ] **Step 5: Monitor and validate WHO output**

Confirm the WHO model executes on all intended slices, no fallback values are
present, valid matching votes remain, and the patient result is `grade_3`.

- [ ] **Step 6: Apply the approved fallback case policy if necessary**

This fallback was not required. Final deterministic inference selected BT0156
after the tumor stage predicted `tumor_4` and the WHO stage predicted
`grade_3`. The committed case remains label-free and contains no generated
results.

- [ ] **Step 7: Add regression coverage for every discovered defect**

Run focused tests after each red-green cycle, then commit only code, tests, and
the final attributed input case.

### Task 10: Complete release verification

**Files:**
- Modify only if verification identifies a release defect.

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run the release audit and manifest check**

Run: `python scripts/release_audit.py --root . --manifest release-manifest.txt`

Expected: exit 0 and no violations.

- [ ] **Step 3: Scan all tracked files for Han characters and non-ASCII policy violations**

Use `rg` for Han ranges and a byte-level ASCII scan where appropriate. The
repository must contain no Chinese text, comments, filenames, private paths,
credentials, custom OpenAI endpoints, relay service names, or comparison-model
code.

- [ ] **Step 4: Verify tracked-file size and checkpoint exclusion**

Use `git ls-files`, suffix checks, and size reporting. Confirm that only the
30 licensed test PNGs are tracked binaries and no model artifact is present.

- [ ] **Step 5: Review the complete branch diff**

Run: `git diff --check origin/main...HEAD` and inspect
`git diff --stat origin/main...HEAD` plus the full security-sensitive diff.

- [ ] **Step 6: Commit any final release fixes**

```bash
git add <verified-files-only>
git commit -m "chore: finalize diagnostic inference release"
```

- [ ] **Step 7: Publish only after explicit final evidence**

Push the feature branch or fast-forward `main` according to the agreed
repository workflow. Report exact test counts, Slurm job IDs, real checkpoint
outputs, selected case, audit results, and the published commit hash.
