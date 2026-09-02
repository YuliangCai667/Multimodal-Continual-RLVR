#!/bin/bash
set -euo pipefail

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

MAIN_LOG=./checkpoints/Qwen3-VL-4B/GRPO-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-4B/GRPO-CL
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting GRPO Continual Learning Training"
echo "Launch time: $(date --iso-8601=seconds)"
echo "======================================"

BASE_MODEL=/home/caiyuliang/models/Qwen3-VL-4B-Instruct
BASE_PATH=/home/caiyuliang/datasets/MRCL

DATASETS=(
    "MedBookVQA"
    "Navigation"
    "We-Math2"
    "Puzzle"
    "FinMME"
)

for TASK_ID in $(seq 1 ${#DATASETS[@]})
do
    "$CONDA_PATH/bin/python" scripts/mrcl_run_state.py stage-start \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
    conda activate trlQwen
    bash scripts/GRPO-CL-4B/grpo.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    conda activate vllmQwen
    bash scripts/GRPO-CL-4B/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    "$CONDA_PATH/bin/python" scripts/mrcl_run_state.py stage-complete \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
done

echo ""
echo "======================================"
echo "All ${#DATASETS[@]} tasks finished!"
echo "======================================"
