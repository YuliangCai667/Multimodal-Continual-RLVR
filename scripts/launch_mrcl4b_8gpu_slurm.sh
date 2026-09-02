#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUNDLE_ROOT=$(cd "$ROOT/.." && pwd)
cd "$ROOT"

if [ "$(id -u)" -eq 0 ]; then
    echo "Refusing to launch the formal experiment as root." >&2
    exit 1
fi

: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"

export BASE_MODEL=${BASE_MODEL:-$BUNDLE_ROOT/models/Qwen3-VL-4B-Instruct}
export BASE_PATH=${BASE_PATH:-$BUNDLE_ROOT/datasets/MRCL}
export TRAIN_ENV_PATH=${TRAIN_ENV_PATH:-$BUNDLE_ROOT/envs/trlQwen}
export EVAL_ENV_PATH=${EVAL_ENV_PATH:-$BUNDLE_ROOT/envs/vllmQwen}
export TRAIN_PYTHON=$TRAIN_ENV_PATH/bin/python
export EVAL_PYTHON=$EVAL_ENV_PATH/bin/python
export STATE_PYTHON=$TRAIN_PYTHON
export MRCL_WORLD_SIZE=8
export MRCL_CPUS_PER_TASK=${MRCL_CPUS_PER_TASK:-4}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MRCL_LAUNCHER_PID=$$

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

for required in \
    "$BASE_MODEL/config.json" \
    "$BASE_PATH/MedBookVQA/jsons/train/data.json" \
    "$TRAIN_PYTHON" \
    "$EVAL_PYTHON" \
    scripts/slurm_srun_8gpu.sh \
    scripts/slurm_rank_worker.sh \
    scripts/GRPO-CL-4B/run_cl_slurm.sh \
    scripts/merge_eval_shards.py
do
    if [ ! -e "$required" ]; then
        echo "Missing required path: $required" >&2
        exit 1
    fi
done

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT=${#ALLOCATED_NODES[@]}
if [ "$NODE_COUNT" -lt 1 ] || [ "$NODE_COUNT" -gt 4 ]; then
    echo "Expected Slurm to allocate 1-4 nodes, got $NODE_COUNT" >&2
    exit 1
fi

echo "Dynamic allocation contains $NODE_COUNT node(s): ${ALLOCATED_NODES[*]}"

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

bash scripts/GRPO-CL-4B/run_cl_slurm.sh
