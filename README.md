# MidlineAgent

MidlineAgent is a multi-agent research framework for traceable clinical
decision support in midline glioma. It coordinates Diagnostic, Survival, and
Treatment Sub-Agents through a hybrid LLM Orchestrator. BrainVLM-Midline is one
internal Diagnostic Sub-Agent model; it is not the title or complete scope of
the MidlineAgent system.

This repository includes the modified training and inference source for
BrainVLM-Midline, its patient-level MIL aggregation, tumor-conditioned WHO
grading, the public Survival implementation, Treatment interfaces, and
orchestration. The Diagnostic runtime imports this repository's Python package
and does not call another source checkout. Model weights remain separate
downloads.

The orchestrator enforces the Diagnostic -> Survival -> Treatment order in
Python and uses the official OpenAI Responses API for configured LLM handoffs
and report synthesis. This software is for research use, is not a medical
device, and must not be used for autonomous clinical decisions.

## Release boundary

The release contains source, configuration templates, and an attributed,
label-free test case with 30 selected 2D MRI slices. It does not contain model
weights, extracted features, API credentials, generated reports, prediction
outputs, or comparison-model implementations. No comparison code is included.

The private internal Treatment database is not distributed. Its schemas,
physician-authored rules, retrieval implementation, credentials, and derived
evidence are not released. Users may implement and inject an `EvidenceProvider` from a
deployment-controlled location outside the Git checkout. The public code does
not automatically discover a database or infer its format. Without the
unreleased evidence and deployment logic, this release cannot reproduce
private-database retrieval or the report quality obtained with those sources;
it does not reproduce the internal treatment report quality.

## Installation

Python 3.9 or newer is required. Install the complete public workflow or a
single Sub-Agent dependency group:

```bash
python -m pip install -e ".[all]"
# Alternatives: .[diagnostic], .[survival], or .[treatment]
```

The compatible diagnostic constraints are in `pyproject.toml`. The exact
historical package set used for checkpoint reference capture is in
`requirements/diagnostic-reference.txt`. PyTorch CPU and CUDA wheels are
platform-specific; select an official PyTorch distribution appropriate for the
host instead of assuming one CUDA package index works everywhere.

## Model assets

Prepare Llama 3.1-8B Instruct and BiomedCLIP in the same roles used by the
related BrainVLM work. Llama access is subject to Meta's license and model
terms. BiomedCLIP access is subject to its model card and applicable terms.
No additional project source repository is needed.

| Asset | Role | Location |
| --- | --- | --- |
| Llama 3.1-8B Instruct | Language backbone | [Hugging Face model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| BiomedCLIP | Vision encoder | [Hugging Face model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| BrainVLM-Midline | Five-class slice model | [Google Drive](https://drive.google.com/file/d/168Bp-JgH9CheV9QYj0MLoolSG_ONFuDD/view?usp=drive_link) |
| WHO_Grading | Tumor-conditioned WHO model | [Google Drive](https://drive.google.com/file/d/1eWBI0DTWuwhQv5kErNfcDzMCmdSplwKL/view?usp=drive_link) |
| MIL | Patient-level logits aggregation | [Google Drive](https://drive.google.com/file/d/15fch44MAuqOE4g4vuPegwAaFZn2EDk-M/view?usp=drive_link) |

Use a local Llama directory for `--llama-model`. The BiomedCLIP argument is
optional: without it, the runtime resolves
`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` through
OpenCLIP; with it, pass an authorized local pretrained file or OpenCLIP
identifier. See `docs/brainvlm-midline.md` for checkpoint hashes and the full
asset contract.

## Label-free BT0156 data

`test_data/utsw_glioma_bt0156/` contains 30 selected axial PNG slices from the
UTSW-Glioma BT0156 subject: 7 T1, 8 T1CE, 7 T2, and 8 FLAIR. The files are
modified derivatives attributed in the case-level `ATTRIBUTION.md` and at
https://doi.org/10.7937/DFAE-1B86 under CC BY 4.0.

`test_data/utsw_glioma_bt0156/annotations.json` is a label-free inference
manifest. No diagnostic labels or results are bundled. The adjacent
`tumor_label_definitions.json` records the five canonical tumor names and the
historical checkpoint class descriptions, but it does not annotate BT0156.

Each inference record requires `patient_id`, a relative `image_path`, a
supported `modality` (`T1`, `T1CE`/`T1C`, `T2`, or `FLAIR`), and a
non-negative integer `slice_index`. A finite, patient-consistent `age` is
optional. The inference loader rejects tumor and grade labels. The case file
used by `midline-run` requires `case_id` and may contain `clinical_facts`
and `mri_manifest`.

## Diagnostic Sub-Agent

The following commands use caller-owned model paths and write all generated
artifacts outside the repository:

```bash
export ANNOTATIONS="$PWD/test_data/utsw_glioma_bt0156/annotations.json"
export IMAGE_ROOT="$PWD/test_data/utsw_glioma_bt0156"
export CASE_JSON="$PWD/test_data/utsw_glioma_bt0156/case.json"
export LLAMA_MODEL=/path/to/Llama-3.1-8B-Instruct
export BIOMEDCLIP_PRETRAINED=/path/to/biomedclip-pretrained.bin
export TUMOR_CHECKPOINT=/path/to/brainvlm-midline.pth
export MIL_CHECKPOINT=/path/to/mil.pth
export WHO_CHECKPOINT=/path/to/who-grading.pth
export DEVICE=cuda:0
export RUN_ROOT=/path/to/caller-owned/run
mkdir -p "$RUN_ROOT"
```

### Tumor prediction

```bash
midline-predict-tumor \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$TUMOR_CHECKPOINT" \
  --mil-checkpoint "$MIL_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$RUN_ROOT/tumor.json"
```

The result contains patient-level tumor logits and probabilities, canonical
class names, modality counts, preprocessing metadata, and checkpoint SHA-256
digests.

The environment-variable launcher is equivalent:

```bash
export OUTPUT_JSON="$RUN_ROOT/tumor.json"
bash scripts/run_tumor_prediction.sh
```

For Slurm, also set `PROJECT_ROOT="$PWD"` and `RUN_DIR="$RUN_ROOT"`, then use
`scripts/run_tumor_prediction.sbatch`. The paired WHO launcher is
`scripts/run_who_grading.sbatch`.

### WHO grading

```bash
midline-predict-who-grade \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$WHO_CHECKPOINT" \
  --tumor-result "$RUN_ROOT/tumor.json" \
  --device "$DEVICE" \
  --output "$RUN_ROOT/who.json"
```

The command validates the tumor-result patient ID. For `tumor_1` and
`tumor_3`, the public rule fixes WHO grade to `grade_4` without loading the
WHO model. Other classes use strict tumor-matched slice voting and record
rejected or invalid generations rather than silently relabelling them.

The environment-variable launcher accepts `TUMOR_RESULT` (or an explicit
`TUMOR_TYPE`):

```bash
export TUMOR_RESULT="$RUN_ROOT/tumor.json"
export OUTPUT_JSON="$RUN_ROOT/who.json"
bash scripts/run_who_grading.sh
```

### Feature extraction and MIL training

Extract label-free slice logits and 4096-dimensional hidden features:

```bash
midline-extract-features \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --checkpoint "$TUMOR_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$RUN_ROOT/bt0156-features.safetensors"
```

For caller-owned labelled cohorts, add `--labeled`. Train AttentionMIL with
explicit, disjoint patient split files:

```bash
midline-train-attention-mil \
  --feature-file /path/to/labelled-features.safetensors \
  --train-split /path/to/train-patients.json \
  --validation-split /path/to/validation-patients.json \
  --output /path/to/caller-owned/run/attention-mil.safetensors
```

### Focused BrainVLM-Midline training

Copy either path-free configuration template, replace every caller-owned data,
asset, and output placeholder, then launch the matching task:

```bash
midline-train-brainvlm \
  --task tumor \
  --config /path/to/caller-owned/brainvlm-tumor.yaml

midline-train-brainvlm \
  --task who \
  --config /path/to/caller-owned/brainvlm-who.yaml
```

The templates are `configs/brainvlm_tumor.example.yaml` and
`configs/brainvlm_who.example.yaml`. Training requires indexed CUDA and writes
an inference-compatible `checkpoint_last.pth` beneath the configured output
directory.

### Integrated diagnostic workflow

`midline-run` executes the shipped tumor and WHO services through the same
Diagnostic workflow used by the orchestrator and writes a privacy-minimized
trace:

```bash
midline-run \
  --case-json "$CASE_JSON" \
  --annotations "$ANNOTATIONS" \
  --image-root "$IMAGE_ROOT" \
  --llama-model "$LLAMA_MODEL" \
  --biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED" \
  --tumor-checkpoint "$TUMOR_CHECKPOINT" \
  --mil-checkpoint "$MIL_CHECKPOINT" \
  --who-checkpoint "$WHO_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$RUN_ROOT/case-trace.json"
```

If a Survival workflow or Treatment client is not injected, its step is
recorded as unavailable. The command never substitutes a different predictor.

## Survival Sub-Agent

The public Survival implementation is in `midline_agent.survival`. It exposes
feature aggregation, regularized autoencoder compression, random survival
forest fitting/prediction, and accelerated failure-time calibration. Fitted
models and patient-level training data are caller-owned and remain outside the
repository.

There is no public Survival command that guesses a deployment's data schema or
fitted-model locations. Construct a callable or object with a `run` method and
inject it as `MidlineOrchestrator(survival_workflow=workflow)`. The callable
receives the validated case and Diagnostic result and returns a
`midline_agent.schema.SurvivalResult`. See `docs/survival-agent.md` for the
public API boundary.

## Treatment Sub-Agent

Set only `OPENAI_API_KEY` and `OPENAI_MODEL` when using the official OpenAI
service. Custom API endpoints are not supported. `OpenAiResponsesClient` uses
the official OpenAI Responses API with structured output, bounded retries,
request timeout, and `store=false`.

The public Treatment boundary contains no private internal database or
evidence. Deployment-owned code implements `EvidenceProvider`, returns a
bounded tuple of `EvidenceItem` values with non-identifying opaque source IDs,
and injects it through
`MidlineOrchestrator(treatment_providers=(provider,))`. The public command
does not automatically load private evidence. See `docs/treatment-providers.md` for
the trust, privacy, persistence, and report-validation contracts.

## Documentation

- `docs/brainvlm-midline.md`: model, assets, annotations, features, and training.
- `docs/data-schema.md`: shared annotations, artifacts, split files, and traces.
- `docs/survival-agent.md`: Survival workflow API.
- `docs/treatment-providers.md`: deployment-injected evidence providers.
- `docs/provenance/brainvlm-midline-sources.md`: source derivation inventory.
- `THIRD_PARTY_NOTICES.md`: data, source-work, library, and model attribution.

## Acknowledgements

The internal BrainVLM-Midline implementation derives task behavior and
checkpoint compatibility from related BrainVLM and MiniGPT-4 work. MiniGPT-4
builds on LAVIS. The runtime uses the installed Transformers, PEFT, OpenCLIP,
and BiomedCLIP ecosystems, and Llama 3.1-8B Instruct subject to its license and
model-access terms. Orchestration uses the official OpenAI Python SDK. See
`THIRD_PARTY_NOTICES.md` and `licenses/` for the precise notices and license
texts. These acknowledgements do not imply a dependency on another project
source tree at runtime.
