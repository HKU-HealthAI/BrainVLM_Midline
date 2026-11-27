#!/usr/bin/env bash
#SBATCH --job-name=midline-who
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the MidlineAgent repository}"
: "${RUN_DIR:?Set RUN_DIR to a writable output directory}"
: "${DEVICE:?Set DEVICE to an indexed CUDA device, for example cuda:0}"

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/tmp"
export TMPDIR="$RUN_DIR/tmp"
export TUMOR_RESULT="${TUMOR_RESULT:-$RUN_DIR/tumor_prediction.json}"
export OUTPUT_JSON="${OUTPUT_JSON:-$RUN_DIR/who_grading.json}"

cd "$PROJECT_ROOT"
exec bash "$PROJECT_ROOT/scripts/run_who_grading.sh"
