#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TASK_ID" >&2
    exit 2
fi
: "${EVAL_PYTHON:?EVAL_PYTHON must point to the packed evaluation Python}"
: "${BASE_PATH:?BASE_PATH is required}"
RANK_LAUNCHER=${MRCL_RANK_LAUNCHER:-scripts/slurm_srun_4gpu.sh}
TASK_ID=$1
DATASETS=(MedBookVQA Navigation We-Math2 Puzzle FinMME)
CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
MODEL_NAME=./checkpoints/Qwen3-VL-4B/CTIR-MULTI-CL/training/$CUR_DATASET
RESULT_BASE=./results/Qwen3-VL-4B/CTIR-MULTI-CL/$CUR_DATASET
NUM_SHARDS=${MRCL_WORLD_SIZE:-4}

for TEST_TASK_ID in $(seq 1 "$TASK_ID"); do
    TEST_DATASET=${DATASETS[$((TEST_TASK_ID - 1))]}
    RESULTS_DIR=$RESULT_BASE/$TEST_DATASET
    SHARD_DIR=$RESULTS_DIR/shards-${SLURM_JOB_ID}
    mkdir -p "$SHARD_DIR"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export MRCL_WORKER_MODE=eval
    bash "$RANK_LAUNCHER" \
        "$EVAL_PYTHON" src/eval/inference.py \
        --base_model "$MODEL_NAME" \
        --test_file "$BASE_PATH/$TEST_DATASET/jsons/test/data.json" \
        --media_dir "$BASE_PATH/$TEST_DATASET/images" \
        --output_dir "$SHARD_DIR" \
        --prompts_file src/dataset/prompts_2.yaml \
        --max_completion_length 2048 \
        --tensor_parallel_size 1 \
        --batch_size 2048
    "$EVAL_PYTHON" scripts/merge_eval_shards.py \
        --shard-dir "$SHARD_DIR" \
        --output "$RESULTS_DIR/merge.jsonl" \
        --num-shards "$NUM_SHARDS"
    "$EVAL_PYTHON" src/eval/eval.py \
        --dataset_name "$TEST_DATASET" \
        --merged_file "$RESULTS_DIR/merge.jsonl" \
        --output_dir "$RESULTS_DIR"
done
