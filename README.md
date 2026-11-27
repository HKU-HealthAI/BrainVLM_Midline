# MidlineAgent

MidlineAgent is a multi-agent research framework for traceable clinical
decision support in midline glioma. It coordinates Diagnostic, Survival, and
Treatment Sub-Agents through a hybrid LLM Orchestrator. The Diagnostic
Sub-Agent integrates BrainVLM-Midline, AttentionMIL, and tumor-conditioned
WHO-grade prediction; the Survival Sub-Agent performs multimodal prognosis
modelling; and the Treatment Sub-Agent produces structured treatment reports
through deployment-injected evidence providers.

This release provides the complete public source code for the Diagnostic,
Survival, and Treatment Sub-Agents, including public orchestration and report
validation. It does not include the private evidence sources, database schemas,
or deployment-owned retrieval code used in the authors' internal environment.
The orchestrator enforces the Diagnostic -> Survival -> Treatment order in
Python and uses the official OpenAI Responses API for validated handoffs and
report synthesis.

## Table of Contents

- [Release boundary](#release-boundary)
- [Installation](#installation)
- [Prepare model weights](#prepare-model-weights)
- [Prepare test data files](#prepare-test-data-files)
- [Diagnostic Sub-Agent](#diagnostic-sub-agent)
- [Survival Sub-Agent](#survival-sub-agent)
- [Treatment Sub-Agent and LLM Orchestrator](#treatment-sub-agent-and-llm-orchestrator)
- [Contracts](#contracts)
- [Acknowledgement](#acknowledgement)

## Release boundary

This repository contains source code, configuration templates, and one
attributed public test case comprising 30 selected 2D MRI slices. It does not
contain diagnostic labels for that case, extracted features, model
checkpoints, API credentials, generated reports, prediction outputs, or
comparison-model implementations. BrainVLM source and all project checkpoints
are distributed separately.

The internal case database, local knowledge sources, physician-authored rules,
private retrieval implementation, and derived records are not distributed.
They are not required to inspect or test the public interfaces, but the public
release cannot reproduce private-database retrieval or the report quality
obtained with those unavailable sources.

This software is for research use. It is not a medical device and must not be
used to make autonomous diagnostic or treatment decisions.

## Installation

Python 3.9 or newer is required. Install the complete public workflow with:

```bash
python -m pip install -e ".[all]"
```

The `diagnostic`, `survival`, and `treatment` extras can instead be installed
separately.

## Prepare model weights

The MidlineAgent Diagnostic Sub-Agent uses Llama 3.1-8B Instruct and BiomedCLIP
foundations through the related BrainVLM project, together with the
BrainVLM-Midline diagnostic model, tumor-conditioned WHO Grading, and
AttentionMIL for patient-level aggregation. Download the required assets
separately; model weights are not stored in this repository.

| Model | Purpose | Download | Configuration |
| --- | --- | --- | --- |
| Llama 3.1-8B Instruct | Language-model backbone used by BrainVLM | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Configure in the external BrainVLM checkout as described by the [BrainVLM project](https://github.com/HKU-HealthAI/BrainVLM). |
| BiomedCLIP | Vision encoder used by BrainVLM | [Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Loaded by the external BrainVLM implementation. |
| BrainVLM-Midline | Five-class slice inference for midline tumors | [Google Drive](https://drive.google.com/file/d/168Bp-JgH9CheV9QYj0MLoolSG_ONFuDD/view?usp=drive_link) | Set `TUMOR_CHECKPOINT` or pass `--checkpoint` to `midline-predict-tumor`. |
| WHO Grading | Tumor-conditioned WHO-grade prediction | [Google Drive](https://drive.google.com/file/d/1eWBI0DTWuwhQv5kErNfcDzMCmdSplwKL/view?usp=drive_link) | Set `WHO_CHECKPOINT` or pass `--checkpoint` to `midline-predict-who-grade`. |
| MIL | Learnable-logits patient aggregation | [Google Drive](https://drive.google.com/file/d/15fch44MAuqOE4g4vuPegwAaFZn2EDk-M/view?usp=drive_link) | Set `MIL_CHECKPOINT` or pass `--mil-checkpoint` to `midline-predict-tumor`. |

Access to Llama 3.1 and BiomedCLIP remains subject to their respective license
and model-access terms. Clone the external source separately:

```bash
git clone https://github.com/HKU-HealthAI/BrainVLM.git /path/to/BrainVLM
```

The released checkpoint runtime receives this checkout through
`--brainvlm-source`. It never copies the checkout or weights into MidlineAgent.
The external environment must provide BrainVLM's own runtime dependencies,
including `transformers`, `open_clip`, `peft`, and `omegaconf`.

## Prepare test data files

The related BrainVLM project supports volumetric `.nii.gz` and `.npy` inputs.
This repository's checkpoint inference commands instead operate on preselected
2D MRI slices and do not provide a 3D-to-2D conversion command.

An attributed test case is included at `test_data/utsw_glioma_bt0156/`. It
contains 30 selected axial PNG slices from subject BT0156 in the UTSW-Glioma
collection: 7 T1, 8 T1CE, 7 T2, and 8 FLAIR slices. The source citation,
modification notice, and CC BY 4.0 terms are recorded in the case-level
`ATTRIBUTION.md` and at https://doi.org/10.7937/DFAE-1B86. The label-free input
file is `test_data/utsw_glioma_bt0156/annotations.json`. No generated
prediction is bundled. The adjacent `tumor_label_definitions.json` records the
current WHO name and the original checkpoint class description for each model
label without disclosing the case label. `tumor_2` is explicitly identified as
a checkpoint superclass because it does not map one-to-one to a single WHO
entity.

For other cases, place de-identified PNG or JPEG slices beneath a caller-owned
image root. The standalone inference reader supports the `T1`, `T1CE` (or
`T1C`), `T2`, and `FLAIR` modality labels. Every inference record requires:

- `patient_id`: an opaque patient identifier;
- `image_path`: a relative path that remains beneath `--image-root`;
- `modality`: one of the supported modality labels; and
- `slice_index`: the integer slice position.

An optional finite `age` value must be consistent across the patient. Labels
such as `tumor_type` and `who_grade` are rejected by the standalone inference
loader. Tumor prediction requires at least one slice from each of T1, T1CE,
T2, and FLAIR so that the released MIL input can be aligned to five slices per
modality.

The case JSON passed through `--case-json` requires `case_id` and may contain
`clinical_facts` and `mri_manifest`. The `case_id` must match a `patient_id` in
the annotation file. It must contain 1-128 ASCII characters, begin with an
alphanumeric character, and otherwise use only alphanumeric characters,
underscores, periods, colons, or hyphens. When present, `clinical_facts` must be
a JSON object and `mri_manifest` must be a JSON array of objects. Their values
must be finite and JSON-compatible.

## Diagnostic Sub-Agent

### Standalone released-checkpoint inference

Set paths to the external dependencies and downloaded checkpoints. The
BiomedCLIP override is optional when the BrainVLM checkout can resolve its
configured pretrained asset itself.

```bash
export ANNOTATIONS="$PWD/test_data/utsw_glioma_bt0156/annotations.json"
export IMAGE_ROOT="$PWD/test_data/utsw_glioma_bt0156"
export BRAINVLM_SOURCE=/path/to/BrainVLM
export LLAMA_MODEL=/path/to/Llama-3.1-8B-Instruct
export BIOMEDCLIP_PRETRAINED=/path/to/biomedclip-pretrained.bin
export TUMOR_CHECKPOINT=/path/to/brainvlm-midline.pth
export MIL_CHECKPOINT=/path/to/mil.pth
export WHO_CHECKPOINT=/path/to/who-grading.pth
export DEVICE=cuda:0
export OUTPUT_JSON=/path/to/run/tumor_prediction.json

bash scripts/run_tumor_prediction.sh

export TUMOR_RESULT="$OUTPUT_JSON"
export OUTPUT_JSON=/path/to/run/who_grading.json
bash scripts/run_who_grading.sh
```

The first command writes patient-level tumor logits, probabilities, modality
counts, preprocessing metadata, and checkpoint SHA-256 digests. The second
command validates the patient ID in the tumor result before WHO inference. For
`tumor_1` and `tumor_3`, WHO grade is fixed to `grade_4` by the public clinical
rule and the model checkpoint is not loaded. Other tumor classes use strict
tumor-matched slice voting; mismatched and invalid model generations are
counted rather than silently relabelled.

The reference labels for BT0156 are `tumor_4` (Astrocytoma, IDH-mutant; the
checkpoint description is IDH-mutated astrocytoma) and `grade_3`. These values
are provided only as verification targets.
No generated prediction is bundled in the repository. Both standalone result
files record `tumor_type`, the corresponding `tumor_name`, and the original
`checkpoint_class_description`. Current entity names follow the WHO
Classification of Tumours, 5th edition; the exact legacy checkpoint text is
retained separately for reproducibility. See the official
[IARC classification structure](https://whobluebooks.iarc.who.int/structures/central-nervous-system-tumours/).

The same two stages can be submitted independently on Slurm. Create a
caller-owned run directory first and pass its log path to `sbatch`; all other
exported variables above are inherited.

```bash
export PROJECT_ROOT="$PWD"
export RUN_DIR=/path/to/run
mkdir -p "$RUN_DIR/logs"

sbatch --output="$RUN_DIR/logs/tumor-%j.log" scripts/run_tumor_prediction.sbatch
# Submit the WHO stage after the tumor job completes successfully.
sbatch --output="$RUN_DIR/logs/who-%j.log" scripts/run_who_grading.sbatch
```

Use `midline-predict-tumor --help` and `midline-predict-who-grade --help` for
the equivalent direct CLI arguments.

### Generic adapter and training interfaces

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

## Survival Sub-Agent

The Survival Sub-Agent is implemented in `midline_agent/survival`. It pools
patient-level BrainVLM hidden features using either all valid slices or T1CE
keyframes, compresses the 4096-dimensional representation with a regularized
autoencoder, estimates risk and survival functions with a random survival
forest, and calibrates median overall survival with an accelerated failure time
model.

`MidlineOrchestrator` accepts a configured `survival_workflow` callable or
object with a `run` method. The workflow receives the case input and validated
diagnostic result and returns a `SurvivalResult` containing a finite risk score,
positive median survival time, and a non-increasing survival curve. The result
is validated before it is supplied to the Treatment Sub-Agent or persisted in
the privacy-minimized trace. Fitted model checkpoints remain external to this
repository and must be supplied by the deployment.

## Treatment Sub-Agent and LLM Orchestrator

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

The LLM Orchestrator executes each configured Sub-Agent in order and does not
silently replace failed predictions. If a runtime model or checkpoint is not
configured, that step is recorded as unavailable. The Treatment Sub-Agent
accepts only injected evidence providers. No guideline corpus, case database,
or web cache is bundled. The optional OpenAI web-search provider uses the same
official Responses client. Reports validate source IDs against collected
evidence before rendering.

Private treatment evidence is not distributed with this repository. Keep any
internal case database, local knowledge base, and physician-authored rules
outside the Git checkout in a deployment-controlled location, such as
`/secure/midlineagent/treatment_evidence/`. The example path is illustrative;
access control, encryption, retention, and privacy review remain deployment
responsibilities.

Deployment-owned code must implement the `EvidenceProvider` protocol, return a
bounded tuple of `EvidenceItem` values with non-identifying opaque source IDs,
and inject the provider through
`MidlineOrchestrator(treatment_providers=(provider,))`. The public
`midline-run` command does not automatically discover or load a private
database, and the repository does not define or infer a private database
format. See `docs/treatment-providers.md` for the complete trust, persistence,
and privacy boundary.

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
[BiomedCLIP](#prepare-model-weights).
MidlineAgent integrates BrainVLM through the external adapter boundary defined
in this repository. Orchestration and treatment-report synthesis use the
official OpenAI Python SDK and
[OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses).
We thank the developers and maintainers of these projects and resources.
