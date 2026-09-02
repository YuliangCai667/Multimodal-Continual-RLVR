#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate trlQwen

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:${PYTHONPATH:-}

MODEL_PATH="$REPO_ROOT/checkpoints/Qwen3-VL-4B/GRPO-CL/training/We-Math2"
DATA_ROOT=/home/caiyuliang/datasets/MRCL
OUTPUT_DIR=${CTIR_OUTPUT_DIR:-"$REPO_ROOT/checkpoints/Qwen3-VL-4B/CTIR-T4-60/training/Puzzle"}
LOG_DIR=${CTIR_LOG_DIR:-"$REPO_ROOT/experiments/ctir_t4_60/logs"}
PROBE_PATH="$REPO_ROOT/experiments/ctir_t4_60/probes/navigation_probes.json"
MAX_STEPS=${CTIR_MAX_STEPS:-60}
SAVE_STEPS=${CTIR_SAVE_STEPS:-10}
MAIN_LOG=${CTIR_MAIN_LOG:-"$LOG_DIR/formal_training.log"}
EXTRA_CTIR_ARGS=()
if [ -n "${CTIR_FORCE_BETA:-}" ]; then
    EXTRA_CTIR_ARGS+=(--ctir_force_beta "$CTIR_FORCE_BETA")
fi
if [ "${CTIR_EXACT_SPECTRUM_CHECK:-False}" = "True" ]; then
    EXTRA_CTIR_ARGS+=(--ctir_exact_spectrum_check True)
fi
if [ -n "${CTIR_STOP_AFTER_STEPS:-}" ]; then
    EXTRA_CTIR_ARGS+=(--ctir_stop_after_steps "$CTIR_STOP_AFTER_STEPS")
fi

if [ -e "$OUTPUT_DIR/checkpoint-10" ] || [ -e "$OUTPUT_DIR/trainer_state.json" ]; then
    echo "Refusing to silently resume/overwrite an existing formal CTIR run: $OUTPUT_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "CTIR run start: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "T3 final model: $MODEL_PATH"

deepspeed --num_gpus 4 src/train/train_grpo.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id "$MODEL_PATH" \
    --data_path "$DATA_ROOT/Puzzle/jsons/train/data.json" \
    --image_folder "$DATA_ROOT/Puzzle/images" \
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
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 20 \
    --dataloader_num_workers 4 \
    --beta 0.0 \
    --temperature 0.8 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --steps_per_generation 16 \
    --epsilon 0.2 \
    --epsilon_high 0.27 \
    --ctir_enable True \
    --ctir_probe_path "$PROBE_PATH" \
    --ctir_probe_count 32 \
    --ctir_layer_start 9 \
    --ctir_layer_end 26 \
    --ctir_tangent_rank 8 \
    --ctir_raw_rank 8 \
    --ctir_refresh_interval 5 \
    --ctir_new_descent_ratio 0.90 \
    --ctir_beta_candidates 0,0.25,0.5,0.75,1.0 \
    --ctir_log_dir "$LOG_DIR" \
    "${EXTRA_CTIR_ARGS[@]}"
