#!/bin/bash

# parse build type
BUILD_TYPE=${1:-Release}
echo "> Build type: $BUILD_TYPE"

# Current directory
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "> Current directory: $CUR_DIR"

# ABI flags used when compiling torch
TORCH_ABI_FLAGS=$(python3 $CUR_DIR/utils/get_torch_abi_flags.py)
echo "> TORCH_ABI_FLAGS: $TORCH_ABI_FLAGS"

# Ensure TT_METAL_HOME is set (default to submodule)
if [ -z "$TT_METAL_HOME" ]; then
    export TT_METAL_HOME=$CUR_DIR/third-party/tt-metal
fi
echo "> TT_METAL_HOME: $TT_METAL_HOME"

# # Build tt-metal using its own script to ensure identical toolchain/config
# echo "> Building tt-metal via build_metal.sh"
# pushd "$TT_METAL_HOME" >/dev/null
# ./build_metal.sh --build-type "$BUILD_TYPE"

# rm -rf python_env
# ./create_venv.sh
# source ./python_env/bin/activate

# popd >/dev/null

# # Ensure tt-metal python package is available in the active environment
# pip3 install -e "$TT_METAL_HOME/" --no-build-isolation

# Select the same toolchain file as tt-metal's build_metal.sh would use
FLAVOR=`grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
TOOLCHAIN_PATH="cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake"
if [[ "$FLAVOR" == "ubuntu" && "$VERSION" == "20.04" ]]; then
    TOOLCHAIN_PATH="cmake/x86_64-linux-clang-17-libcpp-toolchain.cmake"
fi
echo "> Using toolchain file: $TT_METAL_HOME/$TOOLCHAIN_PATH"

# Ensure CMake can locate Torch package config from current Python
TORCH_CMAKE_PREFIX=$(python3 - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)
export CMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# Build cpp extension using pip editable install with the same toolchain
echo "> Building cpp extension"
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE};-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache;-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_HOME/$TOOLCHAIN_PATH;-DWITH_PYTHON_BINDINGS=ON" \
    pip install -e "$CUR_DIR" --use-pep517 --no-build-isolation
