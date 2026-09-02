#!/bin/bash
set -euo pipefail

# Submit with a flexible 1-4 node allocation:
# yhbatch -G 8 -N 1-4 -p h100x --time=<LIMIT> \
#   /XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle/CPO/scripts/submit_mrcl4b_h100x_slurm.sh

export MRCL_EXPECTED_GPU=H100
export TORCH_CUDA_ARCH_LIST=9.0
export MRCL_CPUS_PER_TASK=14
export MRCL_OMP_THREADS=8
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mrcl4b_tianhexy_slurm.sh"
