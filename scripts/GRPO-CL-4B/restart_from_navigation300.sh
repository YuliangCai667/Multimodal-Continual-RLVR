#!/bin/bash
set -euo pipefail

ROOT=/home/caiyuliang/mrcl_cpo
BASE_PY=/home/caiyuliang/anaconda3/bin/python
CONDA_PATH=/home/caiyuliang/anaconda3
BASE_MODEL=/home/caiyuliang/models/Qwen3-VL-4B-Instruct
BASE_PATH=/home/caiyuliang/datasets/MRCL
NAV_MODEL=./checkpoints/Qwen3-VL-4B/GRPO-CL/training/Navigation/checkpoint-300
RESTART_LOG=./checkpoints/Qwen3-VL-4B/GRPO-CL/restart_from_navigation300.log

cd "$ROOT"

if [ "$(id -u)" -eq 0 ]; then
    echo "Refusing to launch the formal experiment as root." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MRCL_LAUNCHER_PID=$$
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source "$CONDA_PATH/etc/profile.d/conda.sh"
exec > >(tee -a "$RESTART_LOG") 2>&1

finish_status() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        "$BASE_PY" scripts/mrcl_run_state.py run-complete
    else
        "$BASE_PY" scripts/mrcl_run_state.py run-failed --exit-code "$rc" || true
    fi
    exit "$rc"
}
trap finish_status EXIT

echo "======================================"
echo "Restarting from Navigation checkpoint-300"
echo "Later task budget: 300 steps each"
echo "Launch time: $(date --iso-8601=seconds)"
echo "======================================"

"$BASE_PY" scripts/mrcl_run_state.py initialize
"$BASE_PY" scripts/mrcl_run_state.py stage-start --task Navigation --task-id 2

conda activate vllmQwen
for TEST_DATASET in MedBookVQA Navigation; do
    TEST_FILE="$BASE_PATH/$TEST_DATASET/jsons/test/data.json"
    MEDIA_DIR="$BASE_PATH/$TEST_DATASET/images"
    RESULTS_DIR="./results/Qwen3-VL-4B/GRPO-CL/Navigation-checkpoint-300/$TEST_DATASET"
    mkdir -p "$RESULTS_DIR"

    python src/eval/inference.py \
        --base_model "$NAV_MODEL" \
        --test_file "$TEST_FILE" \
        --media_dir "$MEDIA_DIR" \
        --output_dir "$RESULTS_DIR" \
        --prompts_file src/dataset/prompts_2.yaml \
        --max_completion_length 2048 \
        --tensor_parallel_size 4 \
        --batch_size 2048

    python src/eval/eval.py \
        --dataset_name "$TEST_DATASET" \
        --merged_file "$RESULTS_DIR/merge.jsonl" \
        --output_dir "$RESULTS_DIR"
done

"$BASE_PY" scripts/mrcl_run_state.py stage-complete --task Navigation --task-id 2

DATASETS=(MedBookVQA Navigation We-Math2 Puzzle FinMME)
for TASK_ID in 3 4 5; do
    CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
    "$BASE_PY" scripts/mrcl_run_state.py stage-start --task "$CUR_DATASET" --task-id "$TASK_ID"

    conda activate trlQwen
    if [ "$TASK_ID" -eq 3 ]; then
        MODEL_NAME_OVERRIDE="$NAV_MODEL" bash scripts/GRPO-CL-4B/grpo.sh \
            "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    else
        bash scripts/GRPO-CL-4B/grpo.sh \
            "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    fi

    conda activate vllmQwen
    bash scripts/GRPO-CL-4B/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    "$BASE_PY" scripts/mrcl_run_state.py stage-complete --task "$CUR_DATASET" --task-id "$TASK_ID"
done

