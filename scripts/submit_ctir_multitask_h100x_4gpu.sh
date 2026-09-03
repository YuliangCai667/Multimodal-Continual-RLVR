#!/bin/bash
# This file is the yhbatch job body.  Slurm copies it to /tmp/slurmd/job*/,
# so it must never locate companion scripts relative to BASH_SOURCE[0].
set -eo pipefail

BUNDLE_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle
HOME_ROOT=/XYAIFS00/HOME/sysu_shenli/sysu_shenli_2
HDD_USER_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl

source /etc/profile >/dev/null 2>&1 || true
if ! module load CUDA/12.4; then
    echo "CUDA/12.4 is not available in the module catalog." >&2
    exit 1
fi
set -u

: "${SLURM_JOB_ID:?Submit this file with yhbatch}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
for command in srun scontrol; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required Slurm command is unavailable: $command" >&2
        exit 1
    fi
done
export CUDA_HOME=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
export HF_HOME=$HDD_USER_ROOT/cache/huggingface
export XDG_CACHE_HOME=$HDD_USER_ROOT/cache
export MRCL_WORLD_SIZE=4
export MRCL_EXPECTED_GPU=H100
export MRCL_MIN_GPU_MEMORY_GB=75
export TORCH_CUDA_ARCH_LIST=9.0
export MRCL_CPUS_PER_TASK=14
export MRCL_OMP_THREADS=8
mkdir -p "$HOME_ROOT/mrcl_job_logs" "$HF_HOME" "$XDG_CACHE_HOME"

for required in \
    "$BUNDLE_ROOT/CPO/scripts/CTIR-MULTI-T1-T5/launch_slurm.sh" \
    "$BUNDLE_ROOT/models/Qwen3-VL-4B-Instruct/config.json" \
    "$BUNDLE_ROOT/datasets/MRCL/MedBookVQA/jsons/train/data.json" \
    "$BUNDLE_ROOT/datasets/MRCL/Navigation/jsons/train/data.json" \
    "$BUNDLE_ROOT/datasets/MRCL/We-Math2/jsons/train/data.json" \
    "$BUNDLE_ROOT/datasets/MRCL/Puzzle/jsons/train/data.json" \
    "$BUNDLE_ROOT/datasets/MRCL/FinMME/jsons/train/data.json" \
    "$BUNDLE_ROOT/envs/trlQwen/bin/python" \
    "$BUNDLE_ROOT/envs/vllmQwen/bin/python"; do
    if [ ! -e "$required" ]; then
        echo "Missing required bundle path: $required" >&2
        exit 1
    fi
done

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "CUDA_HOME: $CUDA_HOME"
cd "$BUNDLE_ROOT/CPO"
exec bash "$BUNDLE_ROOT/CPO/scripts/CTIR-MULTI-T1-T5/launch_slurm.sh"
