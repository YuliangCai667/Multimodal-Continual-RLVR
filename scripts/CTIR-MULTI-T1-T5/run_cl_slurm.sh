#!/bin/bash
set -euo pipefail

: "${TRAIN_PYTHON:?TRAIN_PYTHON is required}"
: "${EVAL_PYTHON:?EVAL_PYTHON is required}"
: "${BASE_PATH:?BASE_PATH is required}"
: "${CTIR_PROBE_ROOT:?CTIR_PROBE_ROOT is required}"

echo "Preparing deterministic code-local probe manifests from existing cluster data"
"$TRAIN_PYTHON" scripts/CTIR-MULTI-T1-T5/prepare_multitask_probes.py \
    --data-root "$BASE_PATH" \
    --output-root "$CTIR_PROBE_ROOT"

echo "Running mandatory worst-case T5 geometry preflight in this same allocation"
bash scripts/CTIR-MULTI-T1-T5/train_stage_slurm.sh preflight 5
"$TRAIN_PYTHON" scripts/CTIR-MULTI-T1-T5/verify_preflight.py \
    "./experiments/ctir_multitask_t1_t5/preflight/job-${SLURM_JOB_ID}"

for TASK_ID in 1 2 3 4 5; do
    echo "Starting formal T${TASK_ID}/5"
    bash scripts/CTIR-MULTI-T1-T5/train_stage_slurm.sh formal "$TASK_ID"
    bash scripts/CTIR-MULTI-T1-T5/eval_stage_slurm.sh "$TASK_ID"
done
echo "EXP-CTIR-MULTI-T1-T5-001 complete"
