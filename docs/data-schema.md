# Data Schema

The repository defines contracts but includes no data records or generated
artifacts.

## Annotation records

The annotation reader accepts a caller-owned JSON array or JSONL file. Each
record contains `patient_id`, an image path relative to a caller-provided image
root, `modality`, `slice_index`, and `tumor_type`. Supported modalities are
`T1`, `T1CE`, `T2`, and `FLAIR`. `who_grade`, `keyframe_label`, and
`report_text` are task-dependent optional fields.

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
artifact identifier and SHA-256 digest. The sidecar records schema and prompt
versions, feature dimensions, modality indices, patient labels, masks, slice
metadata, and tensor keys. Tensor shapes, dtypes, finite values, masks, and
modality ranges are validated on write and read. Pickle-based feature files are
not supported.

AttentionMIL checkpoints use the same bound safetensors-plus-JSON design. The
sidecar records the exact model configuration, feature contract, modality map,
split digests, best epoch, and validation loss. Loading is strict: unexpected,
missing, non-finite, or shape-incompatible tensors are rejected.

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
