#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?This worker must run inside a Slurm allocation}"
: "${SLURM_PROCID:?SLURM_PROCID is required}"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
: "${SLURM_LOCALID:?SLURM_LOCALID is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 PYTHON [ARGS ...]" >&2
    exit 2
fi

# Slurm exposes exactly one GPU to each task via --gpus-per-task=1. CUDA
# renumbers that task-local device to cuda:0, even when SLURM_LOCALID is > 0.
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=0
export MRCL_NODE_LOCAL_RANK=$SLURM_LOCALID

SAFE_USER=${USER:-mrcl}
LOCAL_CACHE_ROOT=/tmp/mrcl_${SLURM_JOB_ID}_${SAFE_USER}/rank_${RANK}
mkdir -p "$LOCAL_CACHE_ROOT/tmp" "$LOCAL_CACHE_ROOT/torch_extensions" "$LOCAL_CACHE_ROOT/triton"
export TMPDIR=$LOCAL_CACHE_ROOT/tmp
export TORCH_EXTENSIONS_DIR=$LOCAL_CACHE_ROOT/torch_extensions
export TRITON_CACHE_DIR=$LOCAL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${MRCL_OMP_THREADS:-8}

PYTHON_BIN=$1
shift

"$PYTHON_BIN" -c '
import os, socket, torch
rank = os.environ.get("RANK")
world_size = os.environ.get("WORLD_SIZE")
slurm_localid = os.environ.get("SLURM_LOCALID")
visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 1:
    raise RuntimeError(
        f"rank={rank} on {socket.gethostname()} "
        f"expected exactly one task-local GPU, found {count}; "
        f"CUDA_VISIBLE_DEVICES={visible_devices}"
    )
name = torch.cuda.get_device_name(0)
total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
expected = os.environ.get("MRCL_EXPECTED_GPU", "")
if expected and expected not in name:
    raise RuntimeError(f"Expected GPU containing {expected!r}, got {name!r}")
if total_memory_gb < float(os.environ.get("MRCL_MIN_GPU_MEMORY_GB", "75")):
    raise RuntimeError(f"GPU memory is only {total_memory_gb:.2f} GB on {name}")
print(
    f"worker host={socket.gethostname()} rank={rank}/{world_size} "
    f"slurm_localid={slurm_localid} "
    f"gpu={name} total_memory_gb={total_memory_gb:.2f}",
    flush=True,
)
'

# vLLM TP=1 evaluation workers are independent data-parallel replicas. Keep
# the shard identity, but remove torch distributed rendezvous variables so
# vLLM TP=1 evaluation workers are independent four-way data-parallel replicas.
if [ "${MRCL_WORKER_MODE:-train}" = "eval" ]; then
    export MRCL_SHARD_RANK=$RANK
    export MRCL_NUM_SHARDS=$WORLD_SIZE
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
fi

exec "$PYTHON_BIN" "$@"
