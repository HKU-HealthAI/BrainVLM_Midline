#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATIONS:?Set ANNOTATIONS to the label-free patient JSON file}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing the referenced images}"
: "${LLAMA_MODEL:?Set LLAMA_MODEL to the Llama 3.1-8B Instruct directory}"
: "${TUMOR_CHECKPOINT:?Set TUMOR_CHECKPOINT to the BrainVLM-Midline checkpoint}"
: "${MIL_CHECKPOINT:?Set MIL_CHECKPOINT to the learnable-logits MIL checkpoint}"
: "${OUTPUT_JSON:?Set OUTPUT_JSON to the destination result JSON}"
: "${DEVICE:?Set DEVICE to an indexed CUDA device, for example cuda:0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"

arguments=(
  --annotations "$ANNOTATIONS"
  --image-root "$IMAGE_ROOT"
  --llama-model "$LLAMA_MODEL"
  --checkpoint "$TUMOR_CHECKPOINT"
  --mil-checkpoint "$MIL_CHECKPOINT"
  --output "$OUTPUT_JSON"
  --device "$DEVICE"
)

if [[ -n "${BIOMEDCLIP_PRETRAINED:-}" ]]; then
  arguments+=(--biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED")
fi

cd "$repository_root"
exec "$PYTHON_BIN" -m scripts.predict_tumor "${arguments[@]}"
