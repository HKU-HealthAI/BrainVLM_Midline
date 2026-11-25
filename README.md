# BrainVLM Midline

BrainVLM Midline is research code for a traceable midline glioma workflow. It
connects a BrainVLM slice model, an AttentionMIL patient-level classifier, a
tumor-conditioned WHO-grade vote, a multimodal Survival Agent, and an
evidence-grounded Treatment Agent. This release provides the complete source
code for the Diagnostic, Survival, and Treatment Agents. A hybrid orchestrator
enforces the Diagnostic -> Survival -> Treatment order in Python and uses the
official OpenAI Responses API for validated handoffs and report synthesis.

## Table of Contents

- [Release boundary](#release-boundary)
- [Installation](#installation)
- [Prepare model weights](#prepare-model-weights)
- [Prepare test data files](#prepare-test-data-files)
- [Diagnostic workflow](#diagnostic-workflow)
- [Survival workflow](#survival-workflow)
- [Treatment and orchestration](#treatment-and-orchestration)
- [Contracts](#contracts)
- [Acknowledgement](#acknowledgement)

## Release boundary

This repository contains code and configuration templates only. It does not
contain patient data, annotation records, extracted features, model
checkpoints, API credentials, reports, evaluation outputs, or comparison-model
implementations. BrainVLM source and all project checkpoints are distributed
separately.

This software is for research use. It is not a medical device and must not be
used to make autonomous diagnostic or treatment decisions.

## Installation

Python 3.10 or newer is required. Install the complete public workflow with:

```bash
python -m pip install -e ".[all]"
```

The `diagnostic`, `survival`, and `treatment` extras can instead be installed
separately.

## Prepare model weights

BrainVLM Midline uses the same Llama and vision-encoder foundations as the
related BrainVLM project, together with project-specific diagnostic weights
and a patient-level MIL model. Download the required assets separately; model
weights are not stored in this repository.

| Model | Purpose | Download | Configuration |
| --- | --- | --- | --- |
| Llama 3.1-8B Instruct | Language-model backbone used by BrainVLM | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Configure in the external BrainVLM checkout as described by the [BrainVLM project](https://github.com/HKU-HealthAI/BrainVLM). |
| BiomedCLIP | Vision encoder used by BrainVLM | [Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Loaded by the external BrainVLM implementation. |
| BrainVLM-Midline | Midline glioma diagnosis and slice-feature extraction | [Google Drive](https://drive.google.com/file/d/168Bp-JgH9CheV9QYj0MLoolSG_ONFuDD/view?usp=drive_link) | Pass to the external adapter with `--checkpoint`. |
| WHO Grading | Tumor-conditioned WHO-grade prediction | [Google Drive](https://drive.google.com/file/d/1eWBI0DTWuwhQv5kErNfcDzMCmdSplwKL/view?usp=drive_link) | Use with a compatible external BrainVLM adapter or the original WHO-grade evaluation pipeline. The public CLI does not expose a separate WHO-checkpoint argument. |
| MIL | Patient-level aggregation of slice predictions | [Google Drive](https://drive.google.com/file/d/15fch44MAuqOE4g4vuPegwAaFZn2EDk-M/view?usp=drive_link) | Supply through `--attention-mil-checkpoint`; the artifact must satisfy the public checkpoint contract in `docs/data-schema.md`. |

Access to Llama 3.1 and BiomedCLIP remains subject to their respective license
and model-access terms. The external BrainVLM source checkout is also required
for `--source-root` and must provide the adapter described in
`docs/external-brainvlm-adapter.md`.

## Prepare test data files

The related BrainVLM project supports volumetric `.nii.gz` and `.npy` inputs.
This repository instead operates on preselected 2D MRI slices and does not
provide a 3D-to-2D conversion command. Convert or select slices externally
before running the public workflow; no patient data or example cases are
bundled here.

Place de-identified slice files beneath a caller-owned image root. Supported
inputs are RGB PNG/JPEG files and two-dimensional NumPy arrays. The annotation
reader supports the `T1`, `T1CE` (or `T1C`), `T2`, and `FLAIR` modality labels;
the loader does not require every modality to be present. Pass the image root
with `--image-root` and provide annotations as either a top-level JSON list or
a JSONL file. Every annotation record requires:

- `patient_id`: an opaque patient identifier;
- `image_path`: a relative path that remains beneath `--image-root`;
- `modality`: one of the supported modality labels;
- `slice_index`: the integer slice position; and
- `tumor_type`: one of `tumor_1` through `tumor_5`.

The case JSON passed through `--case-json` requires `case_id` and may contain
`clinical_facts` and `mri_manifest`. The `case_id` must match a `patient_id` in
the annotation file. It must contain 1-128 ASCII characters, begin with an
alphanumeric character, and otherwise use only alphanumeric characters,
underscores, periods, colons, or hyphens. Optional fields must contain finite,
JSON-compatible values, and every `mri_manifest` entry must be an object.

## Diagnostic workflow

The external BrainVLM integration uses a narrow factory contract and never
copies the external source or checkpoint into this repository. See
`docs/external-brainvlm-adapter.md` for the required interface.

Extract versioned slice features into a safetensors artifact and JSON sidecar:

```bash
midline-extract-features \
  --annotations ANNOTATIONS \
  --image-root IMAGE_ROOT \
  --source-root BRAINVLM_SOURCE \
  --checkpoint BRAINVLM_CHECKPOINT \
  --adapter PACKAGE.MODULE:FACTORY \
  --output FEATURE_ARTIFACT
```

Train AttentionMIL with explicit, disjoint patient-level train and validation
split files:

```bash
midline-train-attention-mil \
  --feature-file FEATURE_ARTIFACT \
  --train-split TRAIN_SPLIT \
  --validation-split VALIDATION_SPLIT \
  --output ATTENTION_MIL_CHECKPOINT
```

The production inference path requires an AttentionMIL checkpoint and does not
silently replace it with mean aggregation. WHO-grade votes are conditioned on
the patient-level tumor class, and rejected mismatched votes remain in
provenance.

External BrainVLM training can be launched with `scripts/train_brainvlm.sh`.
The caller must supply `BRAINVLM_SOURCE_ROOT`, `BRAINVLM_CHECKPOINT`,
`BRAINVLM_TRAIN_SCRIPT`, and `BRAINVLM_TRAIN_CONFIG`.

## Survival workflow

The Survival Agent is implemented in `midline_agent/survival`. It pools
patient-level BrainVLM hidden features using either all valid slices or T1CE
keyframes, compresses the 4096-dimensional representation with a regularized
autoencoder, estimates risk and survival functions with a random survival
forest, and calibrates median overall survival with an accelerated failure time
model.

`MidlineOrchestrator` accepts a configured `survival_workflow` callable or
object with a `run` method. The workflow receives the case input and validated
diagnostic result and returns a `SurvivalResult` containing a finite risk score,
positive median survival time, and a non-increasing survival curve. The result
is validated before it is supplied to the Treatment Agent or persisted in the
privacy-minimized trace. Fitted model checkpoints remain external to this
repository and must be supplied by the deployment.

## Treatment and orchestration

Set `OPENAI_API_KEY` and `OPENAI_MODEL` for the official OpenAI service. Custom
API endpoints are not supported. Run the integrated workflow with:

```bash
midline-run \
  --case-json CASE_INPUT \
  --annotations ANNOTATIONS \
  --image-root IMAGE_ROOT \
  --source-root BRAINVLM_SOURCE \
  --checkpoint BRAINVLM_CHECKPOINT \
  --adapter PACKAGE.MODULE:FACTORY \
  --attention-mil-checkpoint ATTENTION_MIL_CHECKPOINT \
  --output CASE_TRACE
```

The orchestrator executes each configured sub-agent in order and does not
silently replace failed predictions. If a runtime model or checkpoint is not
configured, that step is recorded as unavailable. The Treatment Agent accepts
only injected evidence providers. No guideline
corpus, case database, or web cache is bundled. The optional OpenAI web-search
provider uses the same official Responses client. Reports validate source IDs
against collected evidence before rendering.

Persisted traces hash the case identifier and omit MRI paths, raw clinical
text, evidence excerpts, and URLs. Deployments remain responsible for local
privacy review and retention controls.

## Contracts

- `docs/data-schema.md` defines annotations, split files, artifacts, and traces.
- `docs/external-brainvlm-adapter.md` defines the external model boundary.
- `docs/survival-agent.md` defines the Survival workflow contract.
- `docs/treatment-providers.md` defines evidence-provider requirements.
- `THIRD_PARTY_NOTICES.md` records the external dependency boundary.

## Acknowledgement

The related [BrainVLM](https://github.com/HKU-HealthAI/BrainVLM) project builds
on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and uses
[Llama 3.1-8B Instruct](#prepare-model-weights) and
[BiomedCLIP](#prepare-model-weights). BrainVLM Midline integrates BrainVLM
through the external adapter boundary defined in this repository. Orchestration
and treatment-report synthesis use the official OpenAI Python SDK and
[OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses).
We thank the developers and maintainers of these projects and resources.
