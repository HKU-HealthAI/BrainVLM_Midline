#!/usr/bin/env bash
set -euo pipefail

: "${BRAINVLM_SOURCE_ROOT:?Set BRAINVLM_SOURCE_ROOT to the external BrainVLM checkout.}"
: "${BRAINVLM_CHECKPOINT:?Set BRAINVLM_CHECKPOINT to the external project checkpoint.}"
: "${BRAINVLM_TRAIN_SCRIPT:?Set BRAINVLM_TRAIN_SCRIPT to the external BrainVLM train.py path.}"
: "${BRAINVLM_TRAIN_CONFIG:?Set BRAINVLM_TRAIN_CONFIG to a user-owned training config.}"

export BRAINVLM_SOURCE_ROOT BRAINVLM_CHECKPOINT

exec python "${BRAINVLM_TRAIN_SCRIPT}" --cfg-path "${BRAINVLM_TRAIN_CONFIG}" "$@"
