#!/bin/bash
# Compile AddCustom AscendC kernel for Ascend 910B using ascendc_library().
#
# This produces libadd_custom_kernels.so with the aclrtlaunch_add_custom()
# host-side launch symbol, bypassing the ACLNN framework entirely.
#
# Prerequisites:
#   - CANN 8.3.RC2 installed at /usr/local/Ascend/ascend-toolkit/latest
#   - cmake >= 3.16
#
# Usage:
#   cd examples/mindtorch_v2/ascendc/custom_add && bash build.sh
#
# Output:
#   ./kernel_launch/build/lib/libadd_custom_kernels.so
#
# The compiled library provides:
#   aclrtlaunch_add_custom(blockDim, stream, x, y, z, workspace, tiling)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

# Detect SOC version from npu-smi
SOC_VERSION="${SOC_VERSION:-Ascend910B}"

# Ensure ASCEND environment is sourced
if [ -f "${ASCEND_HOME}/bin/setenv.bash" ]; then
    source "${ASCEND_HOME}/bin/setenv.bash" || true
elif [ -f "${ASCEND_HOME}/../set_env.sh" ]; then
    source "${ASCEND_HOME}/../set_env.sh" || true
fi

BUILD_DIR="${SCRIPT_DIR}/kernel_launch/build"

echo "=== Step 1: Configure cmake (SOC=${SOC_VERSION}) ==="
rm -rf "${BUILD_DIR}"
cmake -B "${BUILD_DIR}" \
    -S "${SCRIPT_DIR}/kernel_launch" \
    -DRUN_MODE=npu \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME}" \
    -DCMAKE_BUILD_TYPE=Release

echo "=== Step 2: Build ==="
cmake --build "${BUILD_DIR}" -j

echo "=== Step 3: Verify output ==="
KERNEL_SO="${BUILD_DIR}/lib/libadd_custom_kernels.so"
if [ -f "${KERNEL_SO}" ]; then
    echo "SUCCESS: ${KERNEL_SO}"
    nm -D "${KERNEL_SO}" | grep -q "aclrtlaunch_add_custom" && \
        echo "  Symbol: aclrtlaunch_add_custom — OK"
else
    echo "ERROR: libadd_custom_kernels.so not found"
    find "${BUILD_DIR}" -name "*.so" -type f
    exit 1
fi

echo "=== Build complete ==="
