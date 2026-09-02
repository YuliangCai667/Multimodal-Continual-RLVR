#!/bin/bash
set -euo pipefail

# Queue one job in both eligible partitions; Slurm starts it in whichever can
# satisfy the flexible eight-GPU allocation first:
# yhbatch -G 8 -N 1-4 -p h100x,a100x --time=<LIMIT> \
#   /XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle/CPO/scripts/submit_mrcl4b_any_slurm.sh

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *H100*)
        export MRCL_EXPECTED_GPU=H100
        export TORCH_CUDA_ARCH_LIST=9.0
        export MRCL_CPUS_PER_TASK=14
        ;;
    *A100*)
        export MRCL_EXPECTED_GPU=A100
        export TORCH_CUDA_ARCH_LIST=8.0
        export MRCL_CPUS_PER_TASK=12
        ;;
    *)
        echo "Unsupported allocated GPU type: $GPU_NAME" >&2
        exit 1
        ;;
esac
export MRCL_OMP_THREADS=8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mrcl4b_tianhexy_slurm.sh"
