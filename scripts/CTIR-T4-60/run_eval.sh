#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

EVAL_PYTHON=${EVAL_PYTHON:-/home/caiyuliang/anaconda3/envs/vllmQwen/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export MRCL_GPU_MEMORY_UTILIZATION=${MRCL_GPU_MEMORY_UTILIZATION:-0.6}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LOG_DIR="$REPO_ROOT/experiments/ctir_t4_60/eval"
mkdir -p "$LOG_DIR"
MAIN_LOG=${CTIR_EVAL_LOG:-"$LOG_DIR/eval.log"}
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "CTIR eval start: $(date --iso-8601=seconds)"
echo "EVAL_PYTHON=$EVAL_PYTHON"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MRCL_GPU_MEMORY_UTILIZATION=$MRCL_GPU_MEMORY_UTILIZATION"

"$EVAL_PYTHON" "$REPO_ROOT/scripts/CTIR-T4-60/run_eval.py"
echo "CTIR eval done: $(date --iso-8601=seconds)"
