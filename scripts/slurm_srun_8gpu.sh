#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 PYTHON [ARGS ...]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORLD_SIZE=${MRCL_WORLD_SIZE:-8}
CPUS_PER_TASK=${MRCL_CPUS_PER_TASK:-4}

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT=${#ALLOCATED_NODES[@]}
if [ "$NODE_COUNT" -lt 1 ] || [ "$NODE_COUNT" -gt 4 ]; then
    echo "Expected a 1-4 node allocation, got $NODE_COUNT: ${ALLOCATED_NODES[*]}" >&2
    exit 1
fi
if [ "$WORLD_SIZE" -ne 8 ]; then
    echo "This experiment requires exactly 8 global GPU ranks, got $WORLD_SIZE" >&2
    exit 1
fi

export MASTER_ADDR=${MASTER_ADDR:-${ALLOCATED_NODES[0]}}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

echo "Launching $WORLD_SIZE one-GPU ranks across $NODE_COUNT node(s): ${ALLOCATED_NODES[*]}"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT CPUS_PER_TASK=$CPUS_PER_TASK"

exec srun \
    --label \
    --kill-on-bad-exit=1 \
    --nodes="$NODE_COUNT" \
    --ntasks="$WORLD_SIZE" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    bash "$SCRIPT_DIR/slurm_rank_worker.sh" "$@"
