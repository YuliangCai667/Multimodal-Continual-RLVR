#!/bin/bash
set -euo pipefail

# Submit from the login node with:
# yhbatch -G 8 -N 1 -p a100x /XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle/CPO/scripts/submit_mrcl4b_a100x.sh

export MRCL_EXPECTED_GPU=A100
export TORCH_CUDA_ARCH_LIST=8.0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mrcl4b_tianhexy.sh"
