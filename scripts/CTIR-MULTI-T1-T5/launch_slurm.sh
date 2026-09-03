#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUNDLE_ROOT=$(cd "$ROOT/.." && pwd)
cd "$ROOT"
if [ "$(id -u)" -eq 0 ]; then
    echo "Refusing to launch an experiment as root." >&2
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
export CTIR_PROBE_ROOT=${CTIR_PROBE_ROOT:-$ROOT/experiments/ctir_multitask_t1_t5/probes}
export MRCL_WORLD_SIZE=4
export MRCL_CPUS_PER_TASK=${MRCL_CPUS_PER_TASK:-14}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

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
    "$TRAIN_PYTHON" \
    "$EVAL_PYTHON" \
    src/dataset/prompts_2.yaml \
    src/train/train_grpo.py \
    src/eval/inference.py \
    src/eval/eval.py \
    scripts/slurm_srun_4gpu.sh \
    scripts/slurm_rank_worker.sh \
    scripts/zero3_offload_h100_80gb.json \
    scripts/merge_eval_shards.py \
    scripts/CTIR-MULTI-T1-T5/run_cl_slurm.sh \
    scripts/CTIR-MULTI-T1-T5/train_stage_slurm.sh \
    scripts/CTIR-MULTI-T1-T5/eval_stage_slurm.sh \
    scripts/CTIR-MULTI-T1-T5/prepare_multitask_probes.py \
    scripts/CTIR-MULTI-T1-T5/verify_preflight.py; do
    if [ ! -e "$required" ]; then
        echo "Missing required existing cluster path: $required" >&2
        exit 1
    fi
done

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT=${#ALLOCATED_NODES[@]}
if [ "$NODE_COUNT" -lt 1 ] || [ "$NODE_COUNT" -gt 4 ]; then
    echo "Expected 1-4 allocated nodes, got $NODE_COUNT" >&2
    exit 1
fi
echo "EXP-CTIR-MULTI-T1-T5-001 allocation: ${ALLOCATED_NODES[*]}"
exec bash scripts/CTIR-MULTI-T1-T5/run_cl_slurm.sh
