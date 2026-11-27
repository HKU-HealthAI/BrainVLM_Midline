#!/usr/bin/env bash
set -euo pipefail

: "${ANNOTATIONS:?Set ANNOTATIONS to the label-free patient JSON file}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing the referenced images}"
: "${BRAINVLM_SOURCE:?Set BRAINVLM_SOURCE to a BrainVLM checkout}"
: "${LLAMA_MODEL:?Set LLAMA_MODEL to the Llama 3.1-8B Instruct directory}"
: "${WHO_CHECKPOINT:?Set WHO_CHECKPOINT to the WHO Grading checkpoint}"
: "${OUTPUT_JSON:?Set OUTPUT_JSON to the destination result JSON}"
: "${DEVICE:?Set DEVICE to an indexed CUDA device, for example cuda:0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"

arguments=(
  --annotations "$ANNOTATIONS"
  --image-root "$IMAGE_ROOT"
  --brainvlm-source "$BRAINVLM_SOURCE"
  --llama-model "$LLAMA_MODEL"
  --checkpoint "$WHO_CHECKPOINT"
  --output "$OUTPUT_JSON"
  --device "$DEVICE"
)

if [[ -n "${TUMOR_TYPE:-}" ]]; then
  arguments+=(--tumor-type "$TUMOR_TYPE")
else
  TUMOR_RESULT="${TUMOR_RESULT:-tumor_prediction.json}"
  arguments+=(--tumor-result "$TUMOR_RESULT")
fi

if [[ -n "${BIOMEDCLIP_PRETRAINED:-}" ]]; then
  arguments+=(--biomedclip-pretrained "$BIOMEDCLIP_PRETRAINED")
fi

cd "$repository_root"
exec "$PYTHON_BIN" -m scripts.predict_who_grade "${arguments[@]}"
