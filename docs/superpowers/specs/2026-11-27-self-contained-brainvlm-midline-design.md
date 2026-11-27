# Self-Contained BrainVLM-Midline Design

**Date:** 2026-11-27

**Status:** Approved

## Objective

Make the MidlineAgent publication repository self-contained with respect to
project source code. A user may install standard third-party Python packages
and download model weights separately, but must not clone BrainVLM,
MiniGPT-4, or any other project-source repository to train or run the released
Diagnostic Sub-Agent.

The released BT0156 example must run tumor-type prediction with patient-level
MIL aggregation and tumor-conditioned WHO grading using only this repository,
installed dependencies, and the separately downloaded model assets.

## Dependency Boundary

The repository may depend on packaged libraries such as PyTorch,
Transformers, PEFT, OpenCLIP, OmegaConf, Pillow, NumPy, and safetensors. Llama
3.1-8B Instruct, BiomedCLIP, BrainVLM-Midline, WHO Grading, and MIL weights
remain separately downloaded assets and are not committed to Git.

The repository must not:

- require a BrainVLM or MiniGPT-4 source checkout;
- add an external project directory to `sys.path` or `PYTHONPATH`;
- dynamically import a model factory from caller-owned source;
- delegate training to an external `train.py`; or
- expose `BRAINVLM_SOURCE`, `--brainvlm-source`, `--source-root`, or an
  external model-factory option.

## Source Architecture

The project-specific implementation will live under
`midline_agent/diagnostic/brainvlm_midline/`:

```text
brainvlm_midline/
|-- model.py
|-- llama.py
|-- vision.py
|-- prompts.py
|-- datasets.py
|-- collate.py
|-- training.py
`-- config.py
```

The exact file boundaries may be adjusted during implementation when that
reduces coupling, but the package must retain these responsibilities:

- construct the checkpoint-compatible BrainVLM-Midline model;
- contain the project-specific Llama and BiomedCLIP adaptations;
- define the tumor-type and WHO-grade prompts in one canonical location;
- provide focused tumor-type and WHO-grade training datasets;
- build validated training batches and configurations; and
- run focused fine-tuning and write inference-compatible state dictionaries.

`BrainVlmRuntime` will construct the internal model directly. The current
external import bridge and generic caller-supplied factory will be removed or
replaced with an internal stable interface. Tumor prediction, feature
extraction, WHO grading, and the orchestrated Diagnostic Sub-Agent must use
the same internal runtime.

## Source Extraction Scope

The implementation will be extracted from the modified project source in the
local MiniGPT-4 working tree. Checkpoint tensor names, tensor shapes, prompt
semantics, image normalization, and MIL inputs are compatibility constraints.

The complete historical `caption_datasets.py` will not be copied. Only the
logic used by the released midline tumor-type and WHO-grade tasks will be
retained, including:

- 2D MRI loading and BiomedCLIP normalization;
- canonical tumor and grade label handling;
- tumor-type and WHO-grade prompt construction;
- optional age-category context;
- task-specific class filtering; and
- the class-balancing behavior used for training.

Segmentation, survival prediction, unrelated tumor baselines, unused dataset
classes, private paths, experimental outputs, debug code, and historical
configuration are outside this migration.

All released files must contain English and ASCII text only unless a file
format intrinsically requires otherwise. Derived MiniGPT-4 or LAVIS code will
retain the applicable copyright and license notices. Third-party provenance
will be recorded in `THIRD_PARTY_NOTICES.md`.

## Inference Data Flow

Tumor prediction follows this path:

```text
30 MRI slices
  -> internal BrainVLM-Midline
  -> five-class logits per slice
  -> MIL checkpoint
  -> patient-level tumor type
```

WHO grading follows this path:

```text
MRI slices + patient-level tumor type
  -> internal WHO Grading model
  -> validated slice generations
  -> fixed-grade rules or strict tumor-matched voting
  -> patient-level WHO grade
```

The standalone tumor CLI accepts annotation and image paths, the Llama and
BiomedCLIP assets, BrainVLM-Midline and MIL checkpoints, an indexed CUDA
device, and an output path. The standalone WHO CLI accepts the corresponding
foundation assets, the WHO checkpoint, a tumor result or explicit canonical
tumor type, an indexed CUDA device, and an output path.

Outputs retain the current schema: canonical labels and names, logits and
probabilities where applicable, modality counts, preprocessing metadata,
checkpoint SHA-256 digests, and WHO voting diagnostics.

## Training Data Flow

Training consumes caller-owned MRI files and annotation JSON through the
internal task-specific datasets, builds the checkpoint-compatible model, and
writes a state-dictionary checkpoint accepted by public inference.

The repository will provide separate tumor-type and WHO-grade configuration
templates. Dataset and output locations must be supplied by the caller. No
training data, absolute server path, W&B credential, email address, historical
result directory, or private checkpoint path may be included.

The release retains the technical core of training rather than every
historical experiment. It must document the expected annotation fields and
provide a launch command, but publication verification does not require a
complete retraining run because the training data are not released.

## Error Handling

The runtime fails before inference when a required model asset is missing or
has the wrong filesystem type. Checkpoint loading rejects unknown parameters,
shape mismatches, non-tensor values, non-finite floating tensors, and malformed
containers.

The input layer rejects paths outside the declared image root, missing MRI
modalities, label-bearing standalone inference annotations, inconsistent
patient identifiers, and non-finite metadata.

WHO generations that do not match the canonical tumor-and-grade format are
invalid votes. They are never silently relabelled. A case with no valid vote
must return an explicit failure. Fixed-grade rules for tumor classes 1 and 3
remain explicit and do not load the WHO checkpoint unnecessarily.

## Verification

Verification has four layers:

1. Unit tests cover internal model construction, image preprocessing, task
   datasets, prompts, labels, checkpoint loading, and error cases.
2. CLI and workflow tests prove that standalone inference, feature extraction,
   MIL aggregation, WHO grading, and orchestration use the internal runtime.
3. Release audits scan all tracked files for Han characters, credentials,
   private filesystem paths, private database contents, generated results,
   external-source arguments, dynamic MiniGPT-4 imports, and bundled weights.
4. GPU jobs run BT0156 tumor prediction plus MIL and WHO grading with the real
   released checkpoints. The jobs must not set a source-root variable or add
   `/projects/MiniGPT-4` to Python import paths. Each command must exit
   successfully and its JSON output must pass the public schema validators.

The real-case results will also be compared with the previously validated
outputs to detect model, prompt, preprocessing, or aggregation drift. Generated
predictions remain outside the publication repository.

## Documentation Changes

The README will describe BrainVLM-Midline as an internal Diagnostic Sub-Agent
model implementation derived from the related BrainVLM and MiniGPT-4 work. It
will no longer instruct users to clone either source repository. Weight
download links remain in the model preparation section.

The external-adapter document and external-source contract will be removed or
replaced by an internal runtime document. The acknowledgement and third-party
notices will credit BrainVLM, MiniGPT-4, LAVIS, Llama, and BiomedCLIP without
implying that runtime source is fetched from those repositories.

## Completion Criteria

The migration is complete only when:

- no runtime or training command requires source outside this repository;
- the focused project training code and modified datasets are present;
- all tracked release files pass the language and privacy audits;
- the automated test suite passes; and
- both released BT0156 checkpoint workflows complete successfully using the
  repository-internal implementation.
