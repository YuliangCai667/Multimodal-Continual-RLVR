#!/bin/bash
set -euo pipefail

BASE_PATH=$1
TASK_ID=$2
shift 2
DATASETS=("$@")

: "${EVAL_PYTHON:?EVAL_PYTHON must point to the packed evaluation Python}"

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
BASE_MODEL_DIR=./checkpoints/Qwen3-VL-4B/GRPO-CL/training
BASE_RESULTS_DIR=./results/Qwen3-VL-4B/GRPO-CL
BATCH_SIZE=2048
NUM_SHARDS=${MRCL_WORLD_SIZE:-8}
MODEL_NAME=${BASE_MODEL_DIR}/${CUR_DATASET}

echo "======================================"
echo "Eight-way data-parallel evaluation after task $TASK_ID: $CUR_DATASET"
echo "Model: $MODEL_NAME"
echo "======================================"

for TEST_TASK_ID in $(seq 1 "$TASK_ID")
do
    TEST_DATASET=${DATASETS[$((TEST_TASK_ID - 1))]}
    TEST_FILE=${BASE_PATH}/${TEST_DATASET}/jsons/test/data.json
    MEDIA_DIR=${BASE_PATH}/${TEST_DATASET}/images
    RESULTS_DIR=${BASE_RESULTS_DIR}/${CUR_DATASET}/${TEST_DATASET}
    SHARD_DIR=${RESULTS_DIR}/shards-${SLURM_JOB_ID}
    OUTPUT_FILE=${RESULTS_DIR}/merge.jsonl
    mkdir -p "$SHARD_DIR"

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export MRCL_WORKER_MODE=eval
    bash scripts/slurm_srun_8gpu.sh \
        "$EVAL_PYTHON" src/eval/inference.py \
        --base_model "$MODEL_NAME" \
        --test_file "$TEST_FILE" \
        --media_dir "$MEDIA_DIR" \
        --output_dir "$SHARD_DIR" \
        --prompts_file src/dataset/prompts_2.yaml \
        --max_completion_length 2048 \
        --tensor_parallel_size 1 \
        --batch_size "$BATCH_SIZE"

    "$EVAL_PYTHON" scripts/merge_eval_shards.py \
        --shard-dir "$SHARD_DIR" \
        --output "$OUTPUT_FILE" \
        --num-shards "$NUM_SHARDS"

    "$EVAL_PYTHON" src/eval/eval.py \
        --dataset_name "$TEST_DATASET" \
        --merged_file "$OUTPUT_FILE" \
        --output_dir "$RESULTS_DIR"
done

echo "======================================"
echo "Evaluation completed after task $TASK_ID"
echo "======================================"
