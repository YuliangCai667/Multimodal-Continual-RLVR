#!/bin/bash
set -euo pipefail

: "${BASE_MODEL:?BASE_MODEL must point to Qwen3-VL-4B-Instruct}"
: "${BASE_PATH:?BASE_PATH must point to the unpacked MRCL dataset}"
: "${TRAIN_PYTHON:?TRAIN_PYTHON is required}"
: "${EVAL_PYTHON:?EVAL_PYTHON is required}"

STATE_PYTHON=${STATE_PYTHON:-$TRAIN_PYTHON}
MAIN_LOG=./checkpoints/Qwen3-VL-4B/GRPO-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-4B/GRPO-CL
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting dynamic-node Slurm GRPO continual learning"
echo "Launch time: $(date --iso-8601=seconds)"
echo "Allocation: ${SLURM_JOB_NODELIST}"
echo "======================================"

DATASETS=(
    MedBookVQA
    Navigation
    We-Math2
    Puzzle
    FinMME
)

for TASK_ID in $(seq 1 ${#DATASETS[@]})
do
    "$STATE_PYTHON" scripts/mrcl_run_state.py stage-start \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
    bash scripts/GRPO-CL-4B/grpo_slurm.sh \
        "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    bash scripts/GRPO-CL-4B/eval_slurm.sh \
        "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    "$STATE_PYTHON" scripts/mrcl_run_state.py stage-complete \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
done

echo "All ${#DATASETS[@]} tasks finished."
