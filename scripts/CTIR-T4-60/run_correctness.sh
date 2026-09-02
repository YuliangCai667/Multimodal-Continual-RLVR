#!/bin/bash
set -euo pipefail

MODE=${1:?Usage: run_correctness.sh beta0|spectrum}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

case "$MODE" in
    beta0)
        export CTIR_FORCE_BETA=0
        export CTIR_EXACT_SPECTRUM_CHECK=False
        ;;
    spectrum)
        export CTIR_FORCE_BETA=1
        export CTIR_EXACT_SPECTRUM_CHECK=True
        ;;
    *)
        echo "Unknown correctness mode: $MODE" >&2
        exit 2
        ;;
esac

export CTIR_MAX_STEPS=60
export CTIR_SAVE_STEPS=10
export CTIR_STOP_AFTER_STEPS=2
export CTIR_OUTPUT_DIR="$REPO_ROOT/checkpoints/Qwen3-VL-4B/CTIR-T4-60/correctness/$MODE"
export CTIR_LOG_DIR="$REPO_ROOT/experiments/ctir_t4_60/correctness/$MODE"
export CTIR_MAIN_LOG="$CTIR_LOG_DIR/run.log"

exec bash "$REPO_ROOT/scripts/CTIR-T4-60/run.sh"
