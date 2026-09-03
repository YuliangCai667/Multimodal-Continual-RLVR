#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [ "$(id -u)" -eq 0 ]; then
    echo "Refusing to launch an experiment as root." >&2
    exit 1
fi

: "${BASE_MODEL:?Set BASE_MODEL to the downloaded Qwen3-VL-4B-Instruct directory}"
: "${BASE_PATH:?Set BASE_PATH to the MRCL dataset root}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the training-environment Python}"
: "${EVAL_PYTHON:?Set EVAL_PYTHON to the vLLM-environment Python}"
for interpreter in "$TRAIN_PYTHON" "$EVAL_PYTHON"; do
    if [ ! -x "$interpreter" ]; then
        echo "Python is missing or not executable: $interpreter" >&2
        exit 2
    fi
done

export CTIR_PROBE_ROOT=${CTIR_PROBE_ROOT:-$ROOT/experiments/ctir_multitask_t1_t5/probes}
export MRCL_RUN_ID=${MRCL_RUN_ID:-online-$(date +%Y%m%d-%H%M%S)}
# Existing stage code uses this value only for unique artifact names.
export SLURM_JOB_ID=$MRCL_RUN_ID
export MRCL_WORLD_SIZE=4
export MRCL_OMP_THREADS=${MRCL_OMP_THREADS:-8}
export MRCL_RANK_LAUNCHER=scripts/local_4gpu.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

for required in \
    "$BASE_MODEL/config.json" \
    "$BASE_PATH/MedBookVQA/jsons/train/data.json" \
    "$BASE_PATH/MedBookVQA/jsons/test/data.json" \
    "$BASE_PATH/MedBookVQA/images" \
    "$BASE_PATH/Navigation/jsons/train/data.json" \
    "$BASE_PATH/Navigation/jsons/test/data.json" \
    "$BASE_PATH/Navigation/images" \
    "$BASE_PATH/We-Math2/jsons/train/data.json" \
    "$BASE_PATH/We-Math2/jsons/test/data.json" \
    "$BASE_PATH/We-Math2/images" \
    "$BASE_PATH/Puzzle/jsons/train/data.json" \
    "$BASE_PATH/Puzzle/jsons/test/data.json" \
    "$BASE_PATH/Puzzle/images" \
    "$BASE_PATH/FinMME/jsons/train/data.json" \
    "$BASE_PATH/FinMME/jsons/test/data.json" \
    "$BASE_PATH/FinMME/images" \
    src/dataset/prompts_2.yaml \
    src/train/train_grpo.py \
    src/eval/inference.py \
    src/eval/eval.py \
    scripts/local_4gpu.sh \
    scripts/local_rank_worker_4gpu.sh \
    scripts/zero3_offload_h100_80gb.json; do
    if [ ! -e "$required" ]; then
        echo "Missing required path: $required" >&2
        exit 2
    fi
done

"$TRAIN_PYTHON" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError(f"Expected exactly four visible GPUs, got {torch.cuda.device_count()}")
for index in range(4):
    name = torch.cuda.get_device_name(index)
    memory_gb = torch.cuda.get_device_properties(index).total_memory / 1e9
    if "H100" not in name or memory_gb < 75:
        raise RuntimeError(f"GPU {index}: expected H100 with >=75 GB, got {name} ({memory_gb:.2f} GB)")
    print(f"GPU {index}: {name}, {memory_gb:.2f} GB", flush=True)
'
"$TRAIN_PYTHON" scripts/CTIR-MULTI-T1-T5/prepare_multitask_probes.py --help >/dev/null
"$TRAIN_PYTHON" src/train/train_grpo.py --help >/dev/null
"$EVAL_PYTHON" src/eval/inference.py --help >/dev/null

echo "EXP-CTIR-MULTI-T1-T5-001 online run id: $MRCL_RUN_ID"
exec bash scripts/CTIR-MULTI-T1-T5/run_cl_slurm.sh
