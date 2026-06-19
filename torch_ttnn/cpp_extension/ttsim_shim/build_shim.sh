#!/bin/bash
#
# TTSim Shim Library Build Script
#
# UGLY WORKAROUND for TTSim/UMD version incompatibility.
# See: https://github.com/tenstorrent/ttsim/issues/4
#
# Usage:
#   ./build_shim.sh [output_dir]
#
# Arguments:
#   output_dir - Directory to place libttsim_shim.so (default: ./build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/build}"

echo "=============================================="
echo "TTSim Shim Library Builder"
echo "=============================================="
echo "UGLY WORKAROUND - Remove when TTSim is fixed!"
echo "Issue: https://github.com/tenstorrent/ttsim/issues/4"
echo "=============================================="
echo ""

# Create build directory
BUILD_DIR="${SCRIPT_DIR}/_build"
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Building shim library..."
echo "  Source: ${SCRIPT_DIR}/ttsim_shim.c"
echo "  Output: ${OUTPUT_DIR}/libttsim_shim.so"
echo ""

# Build using CMake
cd "${BUILD_DIR}"
cmake "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# Copy to output directory (handle both .so and .dylib)
if [[ -f "${BUILD_DIR}/libttsim_shim.so" ]]; then
    cp "${BUILD_DIR}/libttsim_shim.so" "${OUTPUT_DIR}/"
elif [[ -f "${BUILD_DIR}/libttsim_shim.dylib" ]]; then
    # macOS builds .dylib, rename to .so for consistency
    cp "${BUILD_DIR}/libttsim_shim.dylib" "${OUTPUT_DIR}/libttsim_shim.so"
else
    echo "ERROR: Could not find built library"
    ls -la "${BUILD_DIR}/"
    exit 1
fi

echo ""
echo "=============================================="
echo "Build successful!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}/libttsim_shim.so"
echo ""
echo "To use the shim:"
echo "  1. Rename real TTSim library:"
echo "     mv /tmp/ttsim/libttsim.so /tmp/ttsim/libttsim_real.so"
echo ""
echo "  2. Copy shim as libttsim.so:"
echo "     cp ${OUTPUT_DIR}/libttsim_shim.so /tmp/ttsim/libttsim.so"
echo ""
echo "  3. (Optional) Set custom real library path:"
echo "     export TTSIM_SHIM_REAL_LIB=/path/to/libttsim_real.so"
echo ""
echo "=============================================="
