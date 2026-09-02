#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_SCRIPT="$REPO_ROOT/scripts/CTIR-T4-300/run.sh"
LOG_DIR=${CTIR_LOG_DIR:-"$REPO_ROOT/experiments/ctir_t4_300/logs"}
WAIT_LOG=${CTIR_GPU_WAIT_LOG:-"$LOG_DIR/gpu_waiter.log"}
TMUX_SOCKET=${CTIR_GPU_WAIT_SOCKET:-/tmp/ctir_t4_300_wait.sock}
TMUX_SESSION=${CTIR_GPU_WAIT_SESSION:-ctir_t4_300_wait}
IDLE_MEMORY_MIB=${CTIR_IDLE_MEMORY_MIB:-512}
POLL_SECONDS=${CTIR_GPU_POLL_SECONDS:-2}
CONFIRM_SECONDS=${CTIR_GPU_CONFIRM_SECONDS:-1}
REPORT_SECONDS=${CTIR_GPU_REPORT_SECONDS:-300}
TARGET_GPUS=(0 1 2 3)

GPU_STATE_SUMMARY=""
declare -A GPU_UUID=()
declare -A GPU_MEMORY=()
declare -A GPU_PROCESS_COUNT=()

trim() {
    local value=$1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

read_gpu_state() {
    local gpu_csv process_csv index uuid memory process_uuid process_pid gpu
    GPU_UUID=()
    GPU_MEMORY=()
    GPU_PROCESS_COUNT=()

    if ! gpu_csv=$(nvidia-smi \
        --query-gpu=index,uuid,memory.used \
        --format=csv,noheader,nounits 2>/dev/null); then
        GPU_STATE_SUMMARY="GPU query failed"
        return 2
    fi
    if ! process_csv=$(nvidia-smi \
        --query-compute-apps=gpu_uuid,pid \
        --format=csv,noheader,nounits 2>/dev/null); then
        GPU_STATE_SUMMARY="compute-process query failed"
        return 2
    fi

    while IFS=',' read -r index uuid memory; do
        index=$(trim "$index")
        uuid=$(trim "$uuid")
        memory=$(trim "$memory")
        case "$index" in
            0|1|2|3)
                GPU_UUID[$index]=$uuid
                GPU_MEMORY[$index]=$memory
                GPU_PROCESS_COUNT[$index]=0
                ;;
        esac
    done <<< "$gpu_csv"

    for gpu in "${TARGET_GPUS[@]}"; do
        if [[ -z ${GPU_UUID[$gpu]:-} || ! ${GPU_MEMORY[$gpu]:-} =~ ^[0-9]+$ ]]; then
            GPU_STATE_SUMMARY="missing or invalid state for physical GPU $gpu"
            return 2
        fi
    done

    while IFS=',' read -r process_uuid process_pid; do
        process_uuid=$(trim "$process_uuid")
        process_pid=$(trim "$process_pid")
        [[ -z "$process_uuid" || -z "$process_pid" ]] && continue
        for gpu in "${TARGET_GPUS[@]}"; do
            if [[ "$process_uuid" == "${GPU_UUID[$gpu]}" ]]; then
                GPU_PROCESS_COUNT[$gpu]=$((GPU_PROCESS_COUNT[$gpu] + 1))
            fi
        done
    done <<< "$process_csv"

    local ready=0
    local parts=()
    for gpu in "${TARGET_GPUS[@]}"; do
        parts+=("gpu${gpu}=${GPU_MEMORY[$gpu]}MiB/${GPU_PROCESS_COUNT[$gpu]}proc")
        if (( GPU_MEMORY[$gpu] > IDLE_MEMORY_MIB || GPU_PROCESS_COUNT[$gpu] > 0 )); then
            ready=1
        fi
    done
    GPU_STATE_SUMMARY="${parts[*]}"
    return "$ready"
}

status_main() {
    local result
    if read_gpu_state; then
        result=idle
    else
        case $? in
            1) result=busy ;;
            *) result=query_failed ;;
        esac
    fi
    echo "$result: $GPU_STATE_SUMMARY"
}

worker_main() {
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$WAIT_LOG") 2>&1
    trap 'echo "[$(date --iso-8601=seconds)] GPU waiter stopped"; exit 130' INT TERM

    echo "[$(date --iso-8601=seconds)] queued CTIR-T4-300 formal run"
    echo "Trigger: physical GPUs 0,1,2,3 each <=${IDLE_MEMORY_MIB} MiB and zero compute processes"
    echo "Polling every ${POLL_SECONDS}s; confirming once more after ${CONFIRM_SECONDS}s"
    echo "On trigger: exec $RUN_SCRIPT"

    local last_state="" last_report=0 now state result
    while true; do
        if read_gpu_state; then
            result=0
            state=idle
        else
            result=$?
            if (( result == 1 )); then
                state=busy
            else
                state=query_failed
            fi
        fi

        now=$(date +%s)
        if [[ "$state" != "$last_state" ]] || (( now - last_report >= REPORT_SECONDS )); then
            echo "[$(date --iso-8601=seconds)] $state: $GPU_STATE_SUMMARY"
            last_state=$state
            last_report=$now
        fi

        if (( result == 0 )); then
            sleep "$CONFIRM_SECONDS"
            if read_gpu_state; then
                echo "[$(date --iso-8601=seconds)] confirmed idle: $GPU_STATE_SUMMARY"
                echo "[$(date --iso-8601=seconds)] launching formal CTIR-T4-300 now"
                exec bash "$RUN_SCRIPT"
            fi
            echo "[$(date --iso-8601=seconds)] idle condition disappeared during confirmation: $GPU_STATE_SUMMARY"
            last_state=confirmation_lost
        fi
        sleep "$POLL_SECONDS"
    done
}

detach_main() {
    mkdir -p "$LOG_DIR"
    if tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "GPU waiter already active: session=$TMUX_SESSION socket=$TMUX_SOCKET"
        exit 0
    fi
    tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" \
        "bash '$REPO_ROOT/scripts/CTIR-T4-300/launch_when_gpus_free.sh' --worker"
    sleep 1
    if ! tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "GPU waiter failed to stay alive; inspect $WAIT_LOG" >&2
        exit 1
    fi
    echo "Queued in detached tmux: session=$TMUX_SESSION socket=$TMUX_SOCKET"
    echo "Status log: $WAIT_LOG"
}

case "${1:-}" in
    --worker) worker_main ;;
    --status) status_main ;;
    "") detach_main ;;
    *) echo "Usage: $0 [--status|--worker]" >&2; exit 2 ;;
esac
