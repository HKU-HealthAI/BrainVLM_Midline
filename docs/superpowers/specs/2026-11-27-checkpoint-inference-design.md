# Checkpoint-Backed Diagnostic Inference Design

## Objective

Provide two independently runnable public commands for the Diagnostic
Sub-Agent:

1. patient-level tumor prediction from MRI slices using the
   BrainVLM-Midline checkpoint followed by the released MIL checkpoint; and
2. tumor-conditioned WHO grading using the released WHO Grading checkpoint.

The commands must run without OpenAI credentials or the LLM Orchestrator. They
must use external model weights and an external BrainVLM source checkout, fail
explicitly on invalid inputs or inference errors, and support both shell and
Slurm execution. The release will include one attributed public test case that
is verified with the final commands and real checkpoints.

## Release Boundary

MidlineAgent will keep the existing external BrainVLM source boundary. The
repository will not vendor BrainVLM or MiniGPT-4 model source, Llama weights,
BiomedCLIP weights, project checkpoints, generated predictions, or job logs.
This avoids placing third-party-derived model source with unclear repository
licensing under MidlineAgent's MIT license.

The repository will contain a first-party runtime bridge that imports a
caller-owned BrainVLM checkout, constructs the expected model architecture,
loads a caller-supplied project checkpoint, and exposes the exact slice-level
operations required by the two public commands. The bridge will accept
explicit paths for the BrainVLM source, Llama backbone, BiomedCLIP cache or
model identifier, and project checkpoint. It will not read private paths from
the training checkpoint's embedded configuration.

## Runtime Architecture

### Tumor prediction

`midline-predict-tumor` will:

1. load and validate an inference annotation file and all referenced images;
2. construct one BrainVLM model and load the BrainVLM-Midline checkpoint;
3. preprocess every selected slice deterministically;
4. extract the five constrained tumor-token logits for each slice;
5. group slices in `T1`, `T1CE`, `T2`, `FLAIR` order;
6. select the centered five slices per modality, repeating available slices
   only when a modality contains fewer than five slices;
7. load the released MIL checkpoint with PyTorch's weights-only loader;
8. apply the checkpoint's learned class weights and temperature to the aligned
   20 by 5 logit tensor; and
9. write a versioned JSON result containing the case identifier, predicted
   tumor class, patient-level logits and probabilities, slice counts,
   checkpoint fingerprints, and preprocessing metadata.

The released MIL checkpoint is the `LogitsMIL` learnable aggregation selected
by the original training run. Its state contains a five-value class-weight
parameter and a scalar temperature. It is not compatible with the current
public self-attention `AttentionMIL` architecture. The new loader will support
the released format honestly and narrowly; the existing safetensors
`AttentionMilPredictor` remains available for users who train the public
architecture.

### WHO grading

`midline-predict-who-grade` will accept either a tumor result JSON from the
first command or an explicit canonical tumor class. It will:

1. validate the case identifier and tumor input;
2. return the defined grade 4 rule for `tumor_1` and `tumor_3` without loading
   the WHO model;
3. for `tumor_2`, `tumor_4`, or `tumor_5`, construct one BrainVLM model and
   load the WHO Grading checkpoint;
4. generate one constrained `<tumor_X> <grade_Y>` response per slice using a
   fixed prompt and deterministic decoding;
5. strictly parse every generation and retain provenance for invalid or
   tumor-mismatched outputs;
6. vote only among valid generations matching the supplied patient-level
   tumor class; and
7. write a versioned JSON result containing the final grade, vote counts,
   rejected-vote counts, checkpoint fingerprint, and preprocessing metadata.

The two commands run in separate processes so the 8B tumor and WHO models are
never resident on the GPU at the same time.

## Input Contract

Inference annotations will use the existing top-level JSON list format but
will not require ground-truth tumor or grade labels. Each record requires:

- `patient_id`;
- a relative `image_path` contained under the supplied image root;
- canonical modality `T1`, `T1CE`, `T2`, or `FLAIR`;
- integer `slice_index`; and
- optional finite numeric `age`.

All records in one invocation must belong to one patient and must agree on
age when age is present. Every image must exist and decode successfully.
Absolute paths, path traversal, empty bags, unsupported modalities, duplicate
slice identifiers, and silently substituted blank images are rejected.

Training annotations retain their current required-label contract. A separate
inference loader prevents weakening validation for feature extraction and
training.

## Public Test Case

The first candidate is `BT0323` from the UTSW-Glioma collection. The release
will contain its 30 selected, de-identified 2D PNG slices and a label-free
inference annotation file. Filenames will be normalized to short ASCII names.
No generated result file will be committed.

UTSW-Glioma is distributed by The Cancer Imaging Archive under CC BY 4.0. The
repository will include:

- the collection name and DOI `10.7937/DFAE-1B86`;
- the required dataset citation;
- a link to the TCIA collection page;
- the CC BY 4.0 license notice; and
- a statement that the repository files are selected 2D PNG derivatives of
  the original MRI data.

The final runtime will be tested with the real tumor, MIL, and WHO checkpoints.
`BT0323` will remain in the release only if the final tumor command predicts
`tumor_4` and the final WHO command predicts `grade_3`. If deterministic final
inference changes either result, the same verification will be applied to the
pre-screened UTSW candidates in order, and only a case with both correct
patient-level outputs will be published.

## Execution Interfaces

The package will expose:

- `midline-predict-tumor`;
- `midline-predict-who-grade`; and
- an optional lightweight `midline-run-diagnostic` coordinator that invokes
  the two stages sequentially without OpenAI.

Repository launchers will include separate POSIX shell scripts and separate
Slurm submission scripts for tumor prediction and WHO grading. All local
paths will be supplied through documented command-line options or named
environment variables. Scripts will create caller-owned run directories and
logs, which remain git-ignored.

## Error Handling and Security

The public runtime will never invent a tumor class, WHO grade, zero image, or
synthetic logit vector after an error. It will terminate non-zero with a clear
message when a required dependency, source file, checkpoint tensor, image,
model output, or constrained token is missing or invalid.

Checkpoint loading will use `weights_only=True`. The loaders will validate
required state keys, tensor shapes, dtypes, finite values, and positive
temperature. Output files will be written only after complete validation.
Checkpoint paths, image paths, raw generations, and private training
configuration will not be copied into committed files.

## Verification

Automated tests will cover annotation safety, deterministic modality
alignment, exact MIL arithmetic, checkpoint rejection, WHO rules and strict
voting, CLI argument contracts, launcher syntax, public-data attribution, and
release-audit exceptions limited to the attributed test images.

Integration verification will occur in three levels:

1. CPU tests with small fake runtimes for command behavior and result schemas;
2. model-construction and checkpoint-load checks in the existing BrainVLM
   environment; and
3. Slurm GPU jobs using the real three checkpoints and the selected public
   case.

The GPU jobs will be monitored through completion. Logs and result JSON files
will be inspected for actual model execution, correct patient-level outputs,
and absence of fallback behavior. The full test suite, release audit, secret
scan, non-ASCII/Han scan, and tracked-file manifest check must pass before the
branch is published.
