#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 PYTHON [ARGS ...]" >&2
    exit 2
fi
WORLD_SIZE=${MRCL_WORLD_SIZE:-4}
if [ "$WORLD_SIZE" -ne 4 ]; then
    echo "EXP-CTIR-MULTI-T1-T5-001 requires exactly four GPUs" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=$1
shift
if [ "${MRCL_WORKER_MODE:-train}" = train ]; then
    exec "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nproc-per-node=4 \
        --no-python \
        bash "$SCRIPT_DIR/local_rank_worker_4gpu.sh" "$PYTHON_BIN" "$@"
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#VISIBLE_GPUS[@]}" -ne 4 ]; then
    echo "Expected exactly four visible GPUs, got: ${VISIBLE_GPUS[*]}" >&2
    exit 2
fi
PIDS=()
for RANK_INDEX in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS[$RANK_INDEX]}
        export MRCL_SHARD_RANK=$RANK_INDEX
        export MRCL_NUM_SHARDS=4
        unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
        CACHE_ROOT=/tmp/mrcl_${MRCL_RUN_ID:-online}/eval_rank_${RANK_INDEX}
        mkdir -p "$CACHE_ROOT/tmp" "$CACHE_ROOT/triton"
        export TMPDIR=$CACHE_ROOT/tmp
        export TRITON_CACHE_DIR=$CACHE_ROOT/triton
        exec "$PYTHON_BIN" "$@"
    ) &
    PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        STATUS=1
        for OTHER_PID in "${PIDS[@]}"; do
            kill "$OTHER_PID" 2>/dev/null || true
        done
    fi
done
exit "$STATUS"
