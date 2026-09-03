#!/bin/bash
set -euo pipefail

: "${LOCAL_RANK:?torchrun LOCAL_RANK is required}"
: "${RANK:?torchrun RANK is required}"
: "${WORLD_SIZE:?torchrun WORLD_SIZE is required}"
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 PYTHON [ARGS ...]" >&2
    exit 2
fi

TORCHRUN_LOCAL_RANK=$LOCAL_RANK
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#VISIBLE_GPUS[@]}" -ne 4 ]; then
    echo "Expected exactly four visible GPUs, got: ${VISIBLE_GPUS[*]}" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS[$TORCHRUN_LOCAL_RANK]}
export LOCAL_RANK=0
export MRCL_NODE_LOCAL_RANK=$TORCHRUN_LOCAL_RANK

RUN_ID=${MRCL_RUN_ID:-online}
LOCAL_CACHE_ROOT=/tmp/mrcl_${RUN_ID}/rank_${RANK}
mkdir -p "$LOCAL_CACHE_ROOT/tmp" "$LOCAL_CACHE_ROOT/torch_extensions" "$LOCAL_CACHE_ROOT/triton"
export TMPDIR=$LOCAL_CACHE_ROOT/tmp
export TORCH_EXTENSIONS_DIR=$LOCAL_CACHE_ROOT/torch_extensions
export TRITON_CACHE_DIR=$LOCAL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${MRCL_OMP_THREADS:-8}

PYTHON_BIN=$1
shift
"$PYTHON_BIN" -c '
import os, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"rank={os.environ.get('RANK')} expected one task-local GPU; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
name = torch.cuda.get_device_name(0)
memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if "H100" not in name or memory_gb < 75:
    raise RuntimeError(f"Expected H100 with >=75 GB, got {name} ({memory_gb:.2f} GB)")
print(
    f"online worker rank={os.environ.get('RANK')}/{os.environ.get('WORLD_SIZE')} "
    f"physical_local_rank={os.environ.get('MRCL_NODE_LOCAL_RANK')} "
    f"gpu={name} memory_gb={memory_gb:.2f}",
    flush=True,
)
'
exec "$PYTHON_BIN" "$@"
