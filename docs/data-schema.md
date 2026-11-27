# Data Schema

The repository defines the following data contracts. It includes one
attributed, label-free public inference case but no training records, model
weights, extracted features, or generated predictions.

## Label-free inference records

Standalone checkpoint inference accepts a single top-level JSON list. Every
record contains exactly `patient_id`, `image_path`, `modality`, `slice_index`,
and optional `age`. Image paths are relative and must remain beneath the
caller-provided image root. All records must describe one patient. Supported
modalities are `T1`, `T1CE` (with `T1C` accepted as an alias), `T2`, and
`FLAIR`; modality and slice-index pairs must be unique.

The loader verifies each PNG or JPEG with Pillow before inference. Any `age`
must be finite, non-negative, and consistent across records. Diagnostic fields
including `tumor_type`, `who_grade`, generated text, logits, and probabilities
are not accepted by the inference loader. Tumor inference subsequently
requires every supported modality and deterministically aligns five slices per
modality for the released 20-slice MIL input.

## Training annotation records

The training annotation reader accepts a caller-owned JSON array or JSONL file.
Each record contains `patient_id`, an image path relative to a caller-provided
image root, `modality`, `slice_index`, and `tumor_type`. These training labels
are not accepted by the inference loader. Supported modalities are `T1`,
`T1CE`, `T2`, and `FLAIR`. `who_grade`, `keyframe_label`, and `report_text` are
task-dependent optional training fields.

Supported image inputs are RGB PNG/JPEG files and two-dimensional NumPy arrays.
Paths are resolved beneath the explicit image root. Missing labels required by
a selected training task are rejected instead of replaced with defaults.

## Patient splits

AttentionMIL training requires separate train and validation split files. Each
file is a JSON list of opaque patient identifiers. The loader rejects empty
splits, duplicate identifiers, overlap between splits, and identifiers absent
from the feature store. The checkpoint records the SHA-256 digest of each split
file. Test cohorts are not accepted by the training command.

## Feature artifacts

Feature stores use a `.safetensors` tensor file bound to a JSON sidecar by an
artifact identifier and SHA-256 digest. Release extraction always writes the
complete schema `1.0` `hidden_and_logits` profile: five float32 logits and 4096
float32 hidden values per slice, the exact modality map `T1=0`, `T1CE=1`,
`T2=2`, `FLAIR=3`, and the BrainVLM-Midline checkpoint SHA-256, prompt version,
and preprocessing version. Slice rows are ordered by that modality order and
then by slice index. Each row has the identity
`patient_id:modality:slice_index`, a valid-mask value, and metadata containing
the modality, slice index, and source-image SHA-256. Filesystem paths,
generations, and prediction results are not stored.

The default extraction mode accepts the strict label-free inference schema and
does not write a patient label. The explicit labeled mode accepts one
caller-owned patient with one consistent canonical label from `tumor_1`
through `tumor_5` and stores the corresponding zero-based class integer.
Tensor shapes, dtypes, finite values, labels, ordered identities, checksums,
masks, and metadata are validated on write and read. Pickle-based feature files
are not supported.

The same caller-generated feature artifact is the input to AttentionMIL and
the Survival Sub-Agent. AttentionMIL consumes its slice logits and hidden
features; Survival consumes the hidden features and aligned slice metadata.
Generated feature files contain caller data and model outputs, so none are
bundled in this repository or its release artifacts.

AttentionMIL checkpoints use the same bound safetensors-plus-JSON design. The
sidecar records the exact model configuration, feature contract, modality map,
split digests, best epoch, and validation loss. Loading is strict: unexpected,
missing, non-finite, or shape-incompatible tensors are rejected.

The separately distributed MIL artifact used by the released standalone tumor
command is a compact learnable-logits checkpoint. It is loaded with PyTorch's
`weights_only=True` mode and must contain `args.model_type = "logits"`,
`args.aggregation = "learnable"`, a finite five-value `model.weights` tensor,
and a finite positive scalar `model.temperature`. The command aligns the slice
logits to T1, T1CE, T2, and FLAIR order with five slices per modality before
applying this checkpoint. It is distinct from the hidden-feature AttentionMIL
training interface above.

## Standalone diagnostic results

Tumor and WHO-grade commands write schema version `1.2` JSON atomically. Both
results contain the opaque `patient_id`, canonical `tumor_type`, current
`tumor_name`, and the exact legacy `checkpoint_class_description`. The latter
preserves the class text used to train the released checkpoint. `tumor_2` is a
checkpoint superclass and is explicitly marked as such because it does not
map one-to-one to one WHO entity. Current entity names follow the WHO
Classification of Tumours, 5th edition.

Tumor results additionally contain five logits and probabilities, slice and
modality counts, preprocessing metadata, and SHA-256 digests for the BrainVLM
and MIL checkpoints. WHO results contain the grade, method, strict vote counts,
rejected and invalid vote counts, preprocessing metadata, and the WHO
checkpoint digest when model voting is used. Generated text and filesystem
paths are never persisted in either result.

## Case input and trace

The integrated pipeline accepts a caller-owned case JSON containing an opaque
`case_id`, optional `clinical_facts`, and an optional `mri_manifest`. In-memory
workflow state is immutable. The persisted `CaseTrace` contains exactly three
ordered step records: Diagnostic, Survival, and Treatment.

Completed Survival results contain a finite risk score, a positive median
survival time, and paired time/probability vectors. Time points are finite,
non-negative, and strictly increasing; survival probabilities are finite,
bounded by zero and one, and non-increasing. A step without a configured model
is represented explicitly as unavailable rather than replaced with a fallback
prediction.

The trace hashes the case identifier and excludes raw clinical facts, MRI
paths, generated slice text, report Markdown, evidence excerpts, and evidence
URLs. Failure reasons are reduced to generic step status text before
persistence.
