#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 formal TASK_ID | preflight 5" >&2
    exit 2
fi
MODE=$1
TASK_ID=$2
if [ "$MODE" != formal ] && [ "$MODE" != preflight ]; then
    echo "MODE must be formal or preflight" >&2
    exit 2
fi
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 5 ]; then
    echo "TASK_ID must be in [1,5]" >&2
    exit 2
fi

: "${TRAIN_PYTHON:?TRAIN_PYTHON must point to the packed training Python}"
: "${BASE_MODEL:?BASE_MODEL is required}"
: "${BASE_PATH:?BASE_PATH is required}"
: "${CTIR_PROBE_ROOT:?CTIR_PROBE_ROOT is required}"
RANK_LAUNCHER=${MRCL_RANK_LAUNCHER:-scripts/slurm_srun_4gpu.sh}

DATASETS=(MedBookVQA Navigation We-Math2 Puzzle FinMME)
CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
EXPERIMENT_ROOT=./checkpoints/Qwen3-VL-4B/CTIR-MULTI-CL
RESULT_ROOT=./results/Qwen3-VL-4B/CTIR-MULTI-CL

if [ "$MODE" = preflight ]; then
    MODEL_NAME=$BASE_MODEL
    OUTPUT_DIR=$EXPERIMENT_ROOT/preflight/job-${SLURM_JOB_ID}
    LOG_DIR=./experiments/ctir_multitask_t1_t5/preflight/job-${SLURM_JOB_ID}
    MAX_STEPS=300
    PROBE_INDEX=$CTIR_PROBE_ROOT/indexes/T5.json
    CONTINUAL_START_STEP=1200
    EXTRA_CTIR_ARGS=(
        --ctir_multitask_force_beta 1.0
        --ctir_multitask_exact_spectrum_check True
        --ctir_multitask_stop_after_steps 2
    )
else
    OUTPUT_DIR=$EXPERIMENT_ROOT/training/$CUR_DATASET
    LOG_DIR=./experiments/ctir_multitask_t1_t5/logs/$CUR_DATASET
    MAX_STEPS=300
    CONTINUAL_START_STEP=$(((TASK_ID - 1) * 300))
    EXTRA_CTIR_ARGS=()
    if [ "$TASK_ID" -eq 1 ]; then
        MODEL_NAME=$BASE_MODEL
        PROBE_INDEX=
    else
        PREV_DATASET=${DATASETS[$((TASK_ID - 2))]}
        MODEL_NAME=$EXPERIMENT_ROOT/training/$PREV_DATASET
        PROBE_INDEX=$CTIR_PROBE_ROOT/indexes/T${TASK_ID}.json
    fi
    if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Refusing to reuse a non-empty formal-stage output directory: $OUTPUT_DIR" >&2
        exit 2
    fi
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RESULT_ROOT"
DATA_PATH=$BASE_PATH/$CUR_DATASET/jsons/train/data.json
MEDIA_DIR=$BASE_PATH/$CUR_DATASET/images

CTIR_ARGS=()
if [ "$TASK_ID" -ge 2 ]; then
    CTIR_ARGS=(
        --ctir_multitask_enable True
        --ctir_multitask_probe_index_path "$PROBE_INDEX"
        --ctir_multitask_probe_count 32
        --ctir_multitask_layer_start 9
        --ctir_multitask_layer_end 26
        --ctir_multitask_tangent_rank 8
        --ctir_multitask_raw_rank 8
        --ctir_multitask_refresh_interval 5
        --ctir_multitask_union_rtol 1e-6
        --ctir_multitask_new_descent_ratio 0.90
        --ctir_multitask_beta_candidates 0,0.25,0.5,0.75,1.0
        --ctir_multitask_continual_start_step "$CONTINUAL_START_STEP"
        --ctir_multitask_log_dir "$LOG_DIR"
        "${EXTRA_CTIR_ARGS[@]}"
    )
fi

echo "mode=$MODE task=$CUR_DATASET task_id=$TASK_ID model=$MODEL_NAME"
echo "world_size=${MRCL_WORLD_SIZE:-4} per_device_batch=8 gradient_accumulation=4 nominal_global_batch=128"
echo "output=$OUTPUT_DIR log=$LOG_DIR"

export MRCL_WORKER_MODE=train
bash "$RANK_LAUNCHER" \
    "$TRAIN_PYTHON" src/train/train_grpo.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload_h100_80gb.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$MEDIA_DIR" \
    --prompt_path src/dataset/prompts_2.yaml \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 2048 \
    --image_min_pixels $((128 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 30 \
    --save_total_limit 20 \
    --dataloader_num_workers 2 \
    --seed 42 \
    --data_seed 42 \
    --full_determinism False \
    --beta 0.0 \
    --temperature 0.8 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --steps_per_generation 16 \
    --epsilon 0.2 \
    --epsilon_high 0.27 \
    "${CTIR_ARGS[@]}"
