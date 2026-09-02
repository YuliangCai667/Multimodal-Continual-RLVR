#!/bin/bash
# Load the cluster profile before enabling nounset: TianheXY-AI's profile reads
# variables such as LC_IDENTIFICATION that may legitimately be unset.
set -eo pipefail

HOME_ROOT=/XYAIFS00/HOME/sysu_shenli/sysu_shenli_2
HDD_USER_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl
BUNDLE_ROOT=$HDD_USER_ROOT/mrcl_bundle

source /etc/profile >/dev/null 2>&1 || true
if module avail CUDA/12.4 2>&1 | grep -q 'CUDA/12.4'; then
    module load CUDA/12.4
else
    echo "CUDA/12.4 is not available in the module catalog." >&2
    exit 1
fi
set -u

: "${SLURM_JOB_ID:?This script must run under yhbatch/sbatch}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"

for command in srun sinfo scontrol; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required Slurm command is unavailable: $command" >&2
        exit 1
    fi
done

for required in \
    "$BUNDLE_ROOT/CPO/scripts/launch_mrcl4b_8gpu_slurm.sh" \
    "$BUNDLE_ROOT/models/Qwen3-VL-4B-Instruct/config.json" \
    "$BUNDLE_ROOT/datasets/MRCL/MedBookVQA/jsons/train/data.json" \
    "$BUNDLE_ROOT/envs/trlQwen/bin/python" \
    "$BUNDLE_ROOT/envs/vllmQwen/bin/python"
do
    if [ ! -e "$required" ]; then
        echo "Missing required bundle path: $required" >&2
        exit 1
    fi
done

export CUDA_HOME=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
export HF_HOME=$HDD_USER_ROOT/cache/huggingface
export XDG_CACHE_HOME=$HDD_USER_ROOT/cache
export MRCL_WORLD_SIZE=8
export MRCL_CPUS_PER_TASK=${MRCL_CPUS_PER_TASK:-4}
mkdir -p "$HOME_ROOT/mrcl_job_logs" "$HF_HOME" "$XDG_CACHE_HOME"

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node expression: $SLURM_JOB_NODELIST"
echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "CUDA_HOME: $CUDA_HOME"

"$BUNDLE_ROOT/envs/trlQwen/bin/python" -c \
    'import torch; print("train torch", torch.__version__, "CUDA", torch.version.cuda)'
"$BUNDLE_ROOT/envs/vllmQwen/bin/python" -c \
    'import torch, vllm; print("eval torch", torch.__version__, "CUDA", torch.version.cuda, "vLLM", vllm.__version__)'

cd "$BUNDLE_ROOT/CPO"
exec bash scripts/launch_mrcl4b_8gpu_slurm.sh
