#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUNDLE_ROOT=$(cd "$ROOT/.." && pwd)
cd "$ROOT"

if [ "$(id -u)" -eq 0 ]; then
    echo "Refusing to launch the formal experiment as root." >&2
    exit 1
fi

export BASE_MODEL=${BASE_MODEL:-$BUNDLE_ROOT/models/Qwen3-VL-4B-Instruct}
export BASE_PATH=${BASE_PATH:-$BUNDLE_ROOT/datasets/MRCL}
export TRAIN_ENV_PATH=${TRAIN_ENV_PATH:-$BUNDLE_ROOT/envs/trlQwen}
export EVAL_ENV_PATH=${EVAL_ENV_PATH:-$BUNDLE_ROOT/envs/vllmQwen}
export TRAIN_PYTHON=$TRAIN_ENV_PATH/bin/python
export EVAL_PYTHON=$EVAL_ENV_PATH/bin/python
export STATE_PYTHON=$TRAIN_PYTHON

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MRCL_LAUNCHER_PID=$$

for required in \
    "$BASE_MODEL/config.json" \
    "$BASE_PATH/MedBookVQA/jsons/train/data.json" \
    "$TRAIN_PYTHON" \
    "$EVAL_PYTHON"
do
    if [ ! -e "$required" ]; then
        echo "Missing required offline bundle path: $required" >&2
        exit 1
    fi
done

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -ne 8 ]; then
    echo "Expected 8 entries in CUDA_VISIBLE_DEVICES, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

visible_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$visible_gpu_count" -lt 8 ]; then
    echo "Expected at least 8 visible GPUs, found $visible_gpu_count" >&2
    exit 1
fi

mkdir -p logs
"$STATE_PYTHON" scripts/mrcl_run_state.py initialize

finish_status() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        "$STATE_PYTHON" scripts/mrcl_run_state.py run-complete
    else
        "$STATE_PYTHON" scripts/mrcl_run_state.py run-failed --exit-code "$rc" || true
    fi
    exit "$rc"
}
trap finish_status EXIT

bash scripts/GRPO-CL-4B/run_cl_offline.sh
