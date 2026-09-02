#!/bin/bash
set -euo pipefail

ROOT=/home/caiyuliang/mrcl_cpo
BASE_PY=/home/caiyuliang/anaconda3/bin/python
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

mkdir -p logs
"$BASE_PY" scripts/readiness_mrcl4b.py
"$BASE_PY" scripts/mrcl_run_state.py initialize

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

bash scripts/GRPO-CL-4B/run_cl.sh
