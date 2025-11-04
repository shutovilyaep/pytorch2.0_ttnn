#!/bin/bash

# Current directory
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate tt-metal venv if it exists
TT_METAL_VENV="${CUR_DIR}/third-party/tt-metal/python_env"
if [ -f "${TT_METAL_VENV}/bin/activate" ]; then
    echo "> Activating tt-metal venv: ${TT_METAL_VENV}"
    source "${TT_METAL_VENV}/bin/activate"
    echo "> Python: $(which python3)"
else
    echo "> Warning: tt-metal venv not found at ${TT_METAL_VENV}"
    echo "> Continuing with system Python: $(which python3)"
fi

# parse build type
BUILD_TYPE=${1:-Release}
echo "> Build type: $BUILD_TYPE"

echo "> Current directory: $CUR_DIR"

# Ensure TT_METAL_HOME is set (default to submodule)
if [ -z "$TT_METAL_HOME" ]; then
    export TT_METAL_HOME=$CUR_DIR/third-party/tt-metal
fi
echo "> TT_METAL_HOME: $TT_METAL_HOME"

# Determine tt-metal version for CMake
# Try to get version from git tag, fallback to TT_METAL_REF env var, or default
TT_METAL_VERSION=""
if [ -d "$TT_METAL_HOME/.git" ]; then
    # Try to get version from git describe
    TT_METAL_VERSION=$(cd "$TT_METAL_HOME" && git describe --abbrev=0 --tags 2>/dev/null | sed 's/^v//' || echo "")
fi
if [ -z "$TT_METAL_VERSION" ] && [ -n "${TT_METAL_REF:-}" ]; then
    # Use TT_METAL_REF if provided (e.g., v0.60.1 -> 0.60.1)
    TT_METAL_VERSION="${TT_METAL_REF#v}"
fi
if [ -z "$TT_METAL_VERSION" ]; then
    # Fallback to default version
    TT_METAL_VERSION="0.60.1"
    echo "> Warning: Could not determine tt-metal version, using fallback: $TT_METAL_VERSION"
else
    echo "> Detected tt-metal version: $TT_METAL_VERSION"
fi

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
# Pass VERSION_NUMERIC to tt-metal CMake to avoid fallback warnings
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE};-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache;-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_HOME/$TOOLCHAIN_PATH;-DWITH_PYTHON_BINDINGS=ON;-DVERSION_NUMERIC=$TT_METAL_VERSION" \
    pip install -e "$CUR_DIR" --use-pep517 --no-build-isolation
