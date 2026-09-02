#!/bin/bash
set -euo pipefail

# Submit from the login node with:
# yhbatch -G 8 -N 1 -p h100x /XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle/CPO/scripts/submit_mrcl4b_h100x.sh

export MRCL_EXPECTED_GPU=H100
export TORCH_CUDA_ARCH_LIST=9.0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mrcl4b_tianhexy.sh"
