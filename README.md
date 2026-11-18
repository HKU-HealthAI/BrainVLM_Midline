# BrainVLM Midline

BrainVLM Midline is research code for a traceable midline glioma workflow. It
connects a BrainVLM slice model, an AttentionMIL patient-level classifier, a
tumor-conditioned WHO-grade vote, and an evidence-grounded Treatment Agent.
A hybrid orchestrator enforces the workflow order in Python and uses the
official OpenAI Responses API for validated handoffs and report synthesis.

The Survival Agent is intentionally unavailable in this release. The state
machine records that omission and continues to Treatment only through an
explicit handoff.

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

The `diagnostic` and `treatment` extras can instead be installed separately.

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

The Treatment Agent accepts only injected evidence providers. No guideline
corpus, case database, or web cache is bundled. The optional OpenAI web-search
provider uses the same official Responses client. Reports validate source IDs
against collected evidence before rendering.

Persisted traces hash the case identifier and omit MRI paths, raw clinical
text, evidence excerpts, and URLs. Deployments remain responsible for local
privacy review and retention controls.

## Contracts

- `docs/data-schema.md` defines annotations, split files, artifacts, and traces.
- `docs/external-brainvlm-adapter.md` defines the external model boundary.
- `docs/treatment-providers.md` defines evidence-provider requirements.
- `THIRD_PARTY_NOTICES.md` records the external dependency boundary.
