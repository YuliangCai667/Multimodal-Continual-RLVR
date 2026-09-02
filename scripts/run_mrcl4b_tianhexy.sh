#!/bin/bash
# TianheXY-AI's /etc/profile reads variables that may be unset, so nounset
# must only be enabled after the cluster profile and module environment load.
set -eo pipefail

HOME_ROOT=/XYAIFS00/HOME/sysu_shenli/sysu_shenli_2
HDD_USER_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl
BUNDLE_ROOT=$HDD_USER_ROOT/mrcl_bundle
JOB_CACHE_ROOT=$HDD_USER_ROOT/job_cache/${SLURM_JOB_ID:-manual}

# Do not inspect or create anything in the permission-denied HDD_POOL parents.
for required in \
    "$BUNDLE_ROOT/CPO/scripts/launch_mrcl4b_8gpu_offline.sh" \
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

mkdir -p "$HOME_ROOT/mrcl_job_logs" "$JOB_CACHE_ROOT" \
    "$HDD_USER_ROOT/cache/huggingface" \
    "$HDD_USER_ROOT/cache/torch_extensions" \
    "$HDD_USER_ROOT/cache/triton"

# Batch nodes provide only a minimal shell. The cluster's confirmed toolkit is
# CUDA 12.4; the packed environments carry their CUDA 12.8 runtime libraries.
source /etc/profile >/dev/null 2>&1 || true
if module avail CUDA/12.4 2>&1 | grep -q 'CUDA/12.4'; then
    module load CUDA/12.4
else
    echo "CUDA/12.4 is not available in the module catalog." >&2
    exit 1
fi
set -u

export CUDA_HOME=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME=$HDD_USER_ROOT/cache/huggingface
export TORCH_EXTENSIONS_DIR=$HDD_USER_ROOT/cache/torch_extensions
export TRITON_CACHE_DIR=$HDD_USER_ROOT/cache/triton
export XDG_CACHE_HOME=$HDD_USER_ROOT/cache
export TMPDIR=$JOB_CACHE_ROOT

echo "Job ID: ${SLURM_JOB_ID:-unavailable}"
echo "Node: $(hostname)"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ -n "${MRCL_EXPECTED_GPU:-}" ] && \
   ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "$MRCL_EXPECTED_GPU"; then
    echo "Allocated GPU does not match expected type: $MRCL_EXPECTED_GPU" >&2
    exit 1
fi

echo "Checking packed training environment against the allocated GPUs..."
"$BUNDLE_ROOT/envs/trlQwen/bin/python" -c \
    'import torch; print("train torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
echo "Checking packed vLLM environment against the allocated GPUs..."
"$BUNDLE_ROOT/envs/vllmQwen/bin/python" -c \
    'import torch, vllm; print("eval torch", torch.__version__, "CUDA", torch.version.cuda, "vLLM", vllm.__version__); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'

cd "$BUNDLE_ROOT/CPO"
exec bash scripts/launch_mrcl4b_8gpu_offline.sh
