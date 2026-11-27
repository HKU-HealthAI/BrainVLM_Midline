# BrainVLM-Midline Diagnostic Model

BrainVLM-Midline is one model inside the MidlineAgent Diagnostic Sub-Agent.
Its modified training and inference implementation is shipped in
`midline_agent/diagnostic/brainvlm_midline/`. Tumor prediction, WHO grading,
feature extraction, and focused training use that internal package directly.
No other project-source checkout is required or called.

## Model assets

The runtime requires separately obtained model assets:

- a local Llama 3.1-8B Instruct directory;
- BiomedCLIP, resolved by OpenCLIP as
  `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, or an
  authorized local pretrained asset;
- the BrainVLM-Midline tumor checkpoint;
- the WHO Grading checkpoint for learned grade prediction; and
- the MIL checkpoint for patient-level tumor-logit aggregation.

Access to Llama and BiomedCLIP remains subject to their respective licenses,
model cards, and access terms. The runtime checks checkpoint tensor names and
shapes strictly. The approved checkpoint SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| BrainVLM-Midline tumor | `0e3eff840fb5bd5eb9be0ef27b03d17f2fb7aa1aa9dca248300a1f23d7bfab18` |
| WHO Grading | `96c6637b1ae41edaaccc0a32e8b845abeb3995eec232cdcc48e1507b152bc0ca` |
| MIL | `bd3f7b8d5606e649adef490dddb2ec85e37ed1ab8dfdc4304dc185f9723d4eca` |

Verify downloaded files before use. A different digest is not an approved
release artifact even if its filename is similar.

## Tumor classes

Outputs use canonical labels and current entity names while preserving the
historical checkpoint class description as a separate field:

| Label | Canonical tumor name | Historical checkpoint class description |
| --- | --- | --- |
| `tumor_1` | Diffuse midline glioma, H3 K27-altered | H3K27M-mutated midline glioma |
| `tumor_2` | Paediatric-type diffuse glioma (checkpoint superclass) | Pediatric-type diffuse glioma |
| `tumor_3` | Glioblastoma, IDH-wildtype | IDH-wildtype glioblastoma |
| `tumor_4` | Astrocytoma, IDH-mutant | IDH-mutated astrocytoma |
| `tumor_5` | Oligodendroglioma, IDH-mutant and 1p/19q-codeleted | IDH-mutated oligodendroglioma |

The `tumor_2` checkpoint output is explicitly a historical superclass rather
than a one-to-one current WHO entity. Keeping both name fields avoids rewriting
the meaning of the trained checkpoint.

Grade labels are `grade_2`, `grade_3`, and `grade_4`. The public rules fix
`tumor_1` and `tumor_3` to `grade_4`; `tumor_5` permits grades 2 and 3;
and `tumor_2` and `tumor_4` permit grades 2, 3, and 4.

## Inference annotations

Label-free inference annotations are a non-empty JSON list. Every record has:

- `patient_id`: one non-empty patient identifier;
- `image_path`: a relative path contained beneath the image root;
- `modality`: `T1`, `T1CE` (or `T1C`), `T2`, or `FLAIR`; and
- `slice_index`: a non-negative integer unique within the modality.

A finite non-negative `age` is optional and must agree across all records for
the patient. Diagnostic label fields are rejected. Tumor prediction requires
all four modalities because the MIL service aligns up to five slices from each
modality.

The attributed BT0156 manifest at
`test_data/utsw_glioma_bt0156/annotations.json` is label-free. Its neighboring
`tumor_label_definitions.json` describes the complete output vocabulary; it
does not supply a BT0156 tumor or grade annotation.

## Inference commands

Use the exact CLI flags below and place outputs in caller-owned storage:

```bash
midline-predict-tumor \
  --annotations /path/to/label-free-annotations.json \
  --image-root /path/to/images \
  --llama-model /path/to/Llama-3.1-8B-Instruct \
  --biomedclip-pretrained /path/to/biomedclip-pretrained.bin \
  --checkpoint /path/to/brainvlm-midline.pth \
  --mil-checkpoint /path/to/mil.pth \
  --device cuda:0 \
  --output /path/to/run/tumor.json

midline-predict-who-grade \
  --annotations /path/to/label-free-annotations.json \
  --image-root /path/to/images \
  --llama-model /path/to/Llama-3.1-8B-Instruct \
  --biomedclip-pretrained /path/to/biomedclip-pretrained.bin \
  --checkpoint /path/to/who-grading.pth \
  --tumor-result /path/to/run/tumor.json \
  --device cuda:0 \
  --output /path/to/run/who.json
```

An explicit `--tumor-type tumor_N` may replace `--tumor-result` for WHO
grading. The result-file form also checks that the patient identifier matches.

The integrated Diagnostic workflow uses the same services:

```bash
midline-run \
  --case-json /path/to/case.json \
  --annotations /path/to/label-free-annotations.json \
  --image-root /path/to/images \
  --llama-model /path/to/Llama-3.1-8B-Instruct \
  --biomedclip-pretrained /path/to/biomedclip-pretrained.bin \
  --tumor-checkpoint /path/to/brainvlm-midline.pth \
  --mil-checkpoint /path/to/mil.pth \
  --who-checkpoint /path/to/who-grading.pth \
  --device cuda:0 \
  --output /path/to/run/case-trace.json
```

## Feature schema

`midline-extract-features` accepts the same annotations, image root, Llama,
BiomedCLIP, tumor checkpoint, and indexed device arguments. It writes a
safetensors file bound to a JSON sidecar:

```bash
midline-extract-features \
  --annotations /path/to/annotations.json \
  --image-root /path/to/images \
  --llama-model /path/to/Llama-3.1-8B-Instruct \
  --biomedclip-pretrained /path/to/biomedclip-pretrained.bin \
  --checkpoint /path/to/brainvlm-midline.pth \
  --device cuda:0 \
  --output /path/to/run/features.safetensors
```

The extraction metadata uses schema version `1.0`, feature kind
`hidden_and_logits`, `logit_dimension=5`, `hidden_dimension=4096`, and
the canonical modality map `T1=0`, `T1CE=1`, `T2=2`, `FLAIR=3`. It also
records the prompt version, preprocessing version, and tumor-checkpoint digest.
Every active slice has a stable ID, modality, slice index, image checksum,
float32 five-class logits, float32 4096-value hidden vector, and validity flag.

The default mode is label-free and omits a patient label. With `--labeled`,
each extraction record also requires a canonical `tumor_type`; optional
`who_grade`, `keyframe_label`, and `report_text` fields follow the schema
in `docs/data-schema.md`.

## Focused training

Training annotations may be JSON objects mapping relative image paths to record
objects, or JSON Lines records with a relative `figure` field. Records require
`who_class` (legacy `WHO_class` is accepted) and `who_grade`. Optional
fields are `age` (legacy `Age`) and `patient_id`. Canonical training values
use angle brackets: `<tumor_1>` through `<tumor_5>` and `<grade_2>`
through `<grade_4>`.

The parser validates path containment, image decoding, finite ages, canonical
labels, allowed tumor-grade pairs, and patient-label consistency. Tumor
training uses all five classes. WHO training filters to tumors 2, 4, and 5 and
mixes the WHO task with the auxiliary tumor task at the historical 80:20
schedule.

Use `configs/brainvlm_tumor.example.yaml` or
`configs/brainvlm_who.example.yaml` as the complete configuration schema.
Required settings cover annotation paths, image root, Llama and BiomedCLIP
assets, caller-owned output directory, optimizer and scheduler values,
iterations, accumulation, data workers, seed, age context, upsampling, AMP, and
an indexed `cuda:N` device. Optional settings support either resume or an
initialization checkpoint and explicit `env://` distributed training.

```bash
midline-train-brainvlm --task tumor --config /path/to/tumor.yaml
midline-train-brainvlm --task who --config /path/to/who.yaml
```

Training writes `checkpoint_last.pth` to the configured output directory. The
checkpoint stores the strict model state plus optimizer, scaler, scheduler,
epoch, global step, configuration, and tensor manifest needed for validated
resume.

## Intentional exclusions

The release does not contain model weights, historical experiment outputs,
private datasets, segmentation tasks, volumetric-to-slice conversion, private
Survival training data or fitted models, private Treatment evidence, generated
BT0156 diagnoses, comparison implementations, or historical debug and
integration code. It also excludes silent malformed-record skipping,
recursive corrupt-image fallback, and default diagnostic labels.
