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

The reference environment used to create the released checkpoints contains
Python 3.9, PyTorch 2.1.2 with CUDA 11.8, torchvision 0.16.2, Transformers
4.46.3, PEFT 0.2.0, OpenCLIP 2.24.0, OmegaConf 2.3.0, einops 0.8.1, Pillow
11.3.0, and NumPy 1.26.4. The diagnostic dependency set will constrain model
libraries to compatible versions and will publish a reproducibility
requirements file with these exact tested versions. CUDA-specific PyTorch
wheels remain a platform installation choice and are documented separately.

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

### Internal runtime contract

The runtime exposes these project interfaces:

```python
predict_slice_features(image_path, metadata) -> SliceFeatures
predict_tumor_logits(image_path, metadata) -> Tensor[5]
predict_who_generation(image_path, metadata, tumor_type, age) -> str
```

`SliceFeatures` contains a five-value float32 tumor logit vector and a
4096-value float32 hidden vector. Both are produced in one deterministic
forward pass with the canonical tumor prompt ending immediately after
`ASSISTANT:<tumor_`. The hidden vector is the final Llama layer state at that
last prompt position. The logits are the language-model head values at that
same position for the single-token encodings of `1`, `2`, `3`, `4`, and `5`.
`predict_tumor_logits` returns the logit member of this result and may share an
in-memory result within one workflow, but it must not change the calculation.

The feature extraction command writes `hidden_and_logits` feature stores with
`logit_dimension=5`, `hidden_dimension=4096`, the canonical four-value modality
map, prompt and preprocessing versions, checkpoint digest, and per-slice
identity metadata. Its default label-free mode consumes the same strict
annotation schema as standalone inference and omits patient labels. An
explicit `--labeled` mode consumes training annotations with a required
canonical tumor label and writes the integer patient label used by MIL
training. Label-free input rejects diagnostic labels; labelled input rejects
missing or inconsistent patient labels.

The current generic adapter protocol is replaced by this runtime contract.
`midline-extract-features` receives the Llama path, optional local BiomedCLIP
asset, BrainVLM-Midline checkpoint, CUDA device, annotations, image root, and
output path directly. `midline-run` receives those same model assets plus the
released MIL and WHO checkpoints. Its Diagnostic workflow invokes the same
`TumorPredictionService` and `WhoGradePredictionService` used by the standalone
commands and converts their validated outputs into `DiagnosticResult`; it no
longer builds an adapter or a separate AttentionMIL inference path.

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

### Source-to-release mapping

The released tumor dataset is extracted from
`minigpt4/datasets/datasets/caption_datasets.py::tcia_classification_who` and
supports its classification branch only. It preserves the three-instruction
pool, optional child/adult age sentence, 224-pixel bicubic RGB conversion,
BiomedCLIP normalization, five-class prompt, and threshold-based class
upsampling. Caption generation and report defaults are excluded.

The released WHO dataset is extracted from
`caption_datasets.py::just_who_grade`. It preserves filtering to tumor classes
2, 4, and 5; the four-instruction pool; optional child/adult age sentence;
combined tumor-and-grade target; identical image preprocessing; and
threshold-based upsampling by combined label. WHO training also retains the
auxiliary tumor-classification stream from `tcia_classification_who`, mixed at
the original relative sample weights of 80 WHO examples to 20 tumor examples.

`caption_datasets.py::classification_with_who_grade` did not produce the
released WHO checkpoint and is not migrated. Its differing missing-label
defaults are not part of the release contract.

Training annotation files are JSON objects mapping relative image paths to
records, or JSON Lines records with a required `figure` relative path. Tumor
training requires `who_class` or the legacy spelling `WHO_class`. WHO training
requires that tumor field plus `who_grade`. Optional age may use `age` or
legacy `Age`. Canonical values are `<tumor_1>` through `<tumor_5>` and
`<grade_2>` through `<grade_4>`. The release intentionally replaces legacy
silent defaults, recursive corrupt-image fallback, and malformed-line skipping
with path containment, decodability, finite-age, required-label, and canonical
vocabulary validation. Seeded fixture tests must prove parity for valid-record
filtering, age context, prompt selection, preprocessing, upsampling counts, and
the 80:20 WHO stream schedule.

All released files must contain English and ASCII text only unless a file
format intrinsically requires otherwise. Derived MiniGPT-4 or LAVIS code will
retain the applicable copyright and license notices. Third-party provenance
will be recorded in `THIRD_PARTY_NOTICES.md`.

### Checkpoint-compatible model manifest

Both released BrainVLM checkpoints contain 146 tensors and use the same state
key and shape manifest. The internal model retains these registered names:

- 32 layers of PEFT LoRA A/B tensors below
  `llama_model.base_model.model.model.layers`, with rank 8, alpha 32, dropout
  0.05, no bias tensors, and `q_proj` and `v_proj` targets;
- `visual_encoder.norm.weight` and `.bias` from the BiomedCLIP vision trunk;
- the two-layer 768-to-4096-to-4096 GELU projector under
  `llama_proj_mlps.0` and `.2`;
- the 768-to-1408 compatibility layer under `v_q_project`; and
- query, key, value, output, normalization, and bias tensors under
  `cross_attention`.

Model construction uses Llama 3.1-8B Instruct, adds `<box>`, `<Img>`, `</Img>`,
and `<t>` as tokenizer special tokens, sets `$$` as the pad token, resizes token
embeddings as in the training source, and wraps Llama with PEFT before loading
the released state. The vision encoder is
`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; the runtime
uses its vision trunk, captures the output of its final normalization layer,
drops the CLS token, and mean-pools 196 patch tokens for low-resolution slice
encoding. The preprocessing mean is `(0.48145466, 0.4578275, 0.40821073)` and
the standard deviation is `(0.26862954, 0.26130258, 0.27577711)`.

The internal Llama adaptation must preserve the state layout, inputs-embeds
forward path, reduction behavior used during training, and deterministic
generation behavior of the reference environment. It may subclass the pinned
Transformers implementation when parity tests pass; otherwise the minimal
required Apache-2.0 implementation is migrated and cleaned rather than copying
unrelated model heads.

The authoritative local checkpoint digests are:

| Artifact | SHA-256 |
| --- | --- |
| BrainVLM-Midline `checkpoint_1039.pth` | `0e3eff840fb5bd5eb9be0ef27b03d17f2fb7aa1aa9dca248300a1f23d7bfab18` |
| WHO Grading `checkpoint_1052.pth` | `96c6637b1ae41edaaccc0a32e8b845abeb3995eec232cdcc48e1507b152bc0ca` |
| MIL `best_model.pth` | `bd3f7b8d5606e649adef490dddb2ec85e37ed1ab8dfdc4304dc185f9723d4eca` |

Compatibility tests load both 313 MB BrainVLM checkpoints with no unknown or
shape-incompatible tensors and compare their complete 146-entry manifests.

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

### Training contract

The public entry point accepts `--task tumor` or `--task who` and one validated
configuration. Both tasks construct the same checkpoint-compatible model.
Trainable parameters are the LoRA adapters, `visual_encoder.norm`,
`llama_proj_mlps`, `v_q_project`, and `cross_attention`; all other foundation
parameters are frozen. The optimizer is AdamW. Parameters with rank below two
or names containing bias, normalization, or batch-normalization markers receive
zero weight decay; other trainable tensors use configured weight decay.

The retained configuration fields are dataset paths and image root, task,
batch size, dataset mixing weights, seed, output directory, initial and minimum
learning rates, warmup learning rate and steps, weight decay, epochs,
iterations per epoch, worker count, gradient accumulation, AMP, indexed CUDA
device, optional distributed launch settings, and optional resume checkpoint.
The learning-rate schedule is linear warmup followed by cosine decay. AMP uses
a gradient scaler. Gradient accumulation and single-node torchrun DDP are
supported. Seeded shuffling and the 80:20 WHO dataset mixture are reproducible.

Checkpoints contain `model`, `optimizer`, `config`, `scaler`, and `epoch`.
Only trainable model tensors are saved, matching the released checkpoint
envelope. Resume restores all present training state and continues at the next
epoch. A checkpoint intended only for inference may omit optimizer and scaler,
but cannot be used with `--resume`.

Historical W&B integration, opportunistic free-GPU discovery, subprocess
evaluation, webdataset support, segmentation, report captioning, 3D encoders,
Q-Former training, unrelated datasets, and hard-coded output roots are
intentionally unsupported.

A component smoke test runs one forward/backward/optimizer/scheduler step with
small injected foundation stubs, saves a trainable-only checkpoint, resumes it,
and loads its model envelope through the same strict checkpoint loader used by
inference. Real foundation weights are covered by the BT0156 acceptance jobs,
not by CPU unit tests.

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

Before migration, the current external implementation writes private reference
artifacts beneath `.runtime/runs/self-contained-reference/`: ordered per-slice
five-class logits, 4096-dimensional hidden vectors, deterministic WHO
generations, accepted/rejected vote details, aligned 20-by-5 MIL input, patient
logits and probabilities, final labels, environment versions, and asset
digests. These generated artifacts are never committed.

After migration, the same records are produced by an installed wheel in an
isolated virtual environment. Slice order and tensor shapes must match exactly;
floating tensors must satisfy `rtol=1e-4` and `atol=1e-4`; deterministic WHO
generations and vote decisions must match exactly; and final tumor type and WHO
grade must match. Any tolerance failure is investigated rather than hidden by
checking only the final class.

The isolated environment is created outside the checkout, unsets `PYTHONPATH`
and every removed source variable, installs the built wheel plus declared
dependencies, and runs all console entry points while the local MiniGPT-4 tree
is absent from import search paths. A module-origin audit requires all
`midline_agent` and project model modules to resolve inside the installed wheel;
other imports must resolve to installed distributions. Isolated acceptance
covers standalone tumor plus MIL, WHO grading, feature extraction, and the
Diagnostic portion of orchestration. Treatment API network calls are replaced
by the existing deterministic test client for this local acceptance test.

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

The obsolete `docs/external-brainvlm-adapter.md`, external training example,
external bridge module and tests, and the prior external-checkout checkpoint
specification and plan are removed. Active launchers, README examples, console
arguments, environment templates, release manifest, and test names are
rewritten for internal runtime assets. Policy and migration documents may name
removed options only in explicit prohibition or historical-removal passages;
the audit exemption is limited to those exact documentation paths and does not
apply to executable files, configuration templates, README instructions, or
tests.

## License and Provenance

Before a derived source file is committed, a file-level provenance inventory
records its released path, local source path and SHA-256, upstream repository
and exact revision when identifiable, modification statement, copyright
holder, and license. Missing local Git history does not permit inventing an
upstream revision: the inventory records the immutable local source digest and
the independently verified upstream base revision or states the local project
lineage explicitly. BrainVLM-Midline-specific changes authored by this project
are identified as HKU-HealthAI project code rather than misattributed to the
public BrainVLM or MiniGPT-4 repositories. Any source whose ownership cannot be
established is excluded until the maintainers resolve it.

MiniGPT-4-derived files retain the BSD-3-Clause notice for Copyright 2023 Deyao
Zhu. LAVIS-derived files retain the separate BSD-3-Clause notice for Copyright
2022 Salesforce, Inc. Llama implementation code copied from Transformers
retains its Hugging Face and EleutherAI attribution and Apache-2.0 header.
Project-owned modifications carry the repository's project copyright notice.
Complete copies of both BSD-3-Clause licenses and Apache License 2.0, plus all
required notices and modification statements, ship with the source. The
current statement that no third-party model source is vendored is replaced by
this inventory and license boundary.

## Completion Criteria

The migration is complete only when:

- no runtime or training command requires source outside this repository;
- the focused project training code and modified datasets are present;
- all tracked release files pass the language and privacy audits;
- the automated test suite passes;
- wheel and source distributions contain every required project module and no
  prohibited artifact;
- isolated-install tumor, WHO, feature, and orchestration acceptance passes;
- both released BT0156 checkpoint workflows complete successfully using the
  repository-internal implementation; and
- per-slice and patient-level reference parity passes at the stated tolerances.
