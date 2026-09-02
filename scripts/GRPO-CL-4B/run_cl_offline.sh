#!/bin/bash
set -euo pipefail

: "${BASE_MODEL:?BASE_MODEL must point to the unpacked Qwen3-VL-4B-Instruct directory}"
: "${BASE_PATH:?BASE_PATH must point to the unpacked MRCL directory}"
: "${TRAIN_ENV_PATH:?TRAIN_ENV_PATH must point to the unpacked trlQwen environment}"
: "${EVAL_ENV_PATH:?EVAL_ENV_PATH must point to the unpacked vllmQwen environment}"

BASE_SYSTEM_PATH=$PATH
STATE_PYTHON=${STATE_PYTHON:-$TRAIN_ENV_PATH/bin/python}

activate_train_env() {
    export PATH="$TRAIN_ENV_PATH/bin:$BASE_SYSTEM_PATH"
    export CONDA_PREFIX="$TRAIN_ENV_PATH"
    unset CONDA_DEFAULT_ENV || true
}

activate_eval_env() {
    export PATH="$EVAL_ENV_PATH/bin:$BASE_SYSTEM_PATH"
    export CONDA_PREFIX="$EVAL_ENV_PATH"
    unset CONDA_DEFAULT_ENV || true
}

MAIN_LOG=./checkpoints/Qwen3-VL-4B/GRPO-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-4B/GRPO-CL
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting offline GRPO Continual Learning Training"
echo "Launch time: $(date --iso-8601=seconds)"
echo "Training environment: $TRAIN_ENV_PATH"
echo "Evaluation environment: $EVAL_ENV_PATH"
echo "======================================"

DATASETS=(
    "MedBookVQA"
    "Navigation"
    "We-Math2"
    "Puzzle"
    "FinMME"
)

for TASK_ID in $(seq 1 ${#DATASETS[@]})
do
    "$STATE_PYTHON" scripts/mrcl_run_state.py stage-start \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
    activate_train_env
    bash scripts/GRPO-CL-4B/grpo.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    activate_eval_env
    bash scripts/GRPO-CL-4B/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    "$STATE_PYTHON" scripts/mrcl_run_state.py stage-complete \
        --task "${DATASETS[$((TASK_ID - 1))]}" --task-id "$TASK_ID"
done

echo ""
echo "======================================"
echo "All ${#DATASETS[@]} tasks finished!"
echo "======================================"
