#!/usr/bin/env bash
#SBATCH --job-name=midline-tumor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the MidlineAgent repository}"
: "${RUN_DIR:?Set RUN_DIR to a writable output directory}"

mkdir -p "$RUN_DIR"
export OUTPUT_JSON="${OUTPUT_JSON:-$RUN_DIR/tumor_prediction.json}"

cd "$PROJECT_ROOT"
exec bash "$PROJECT_ROOT/scripts/run_tumor_prediction.sh"
