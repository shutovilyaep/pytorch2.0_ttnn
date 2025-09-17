#!/usr/bin/env bash
set -euo pipefail

TT_METAL_VERSION=v0.58.1
TT_METAL_HOME=/home/epam/shutov/tt-metal/$TT_METAL_VERSION
export TT_METAL_HOME=$TT_METAL_HOME

# Prefer external tt-metal venv if present; otherwise fall back to local submodule and create venv if needed
if [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
    source "$TT_METAL_HOME/python_env/bin/activate"
    echo "Using venv from version $TT_METAL_VERSION"
else
    echo "WEIRD"
    # LOCAL_TT_METAL="/home/epam/shutov/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal"
    # if [ ! -f "$LOCAL_TT_METAL/python_env/bin/activate" ]; then
    #     echo "Local tt-metal venv not found. Creating one at $LOCAL_TT_METAL/python_env ..."
    #     pushd "$LOCAL_TT_METAL" >/dev/null
    #     ./create_venv.sh
    #     popd >/dev/null
    # fi
    # source "$LOCAL_TT_METAL/python_env/bin/activate"
    # export TT_METAL_HOME="$LOCAL_TT_METAL"
fi


CACHE_DIR=/home/epam/shutov/.cache/cpp-extension-cache

# # Added pre-build steps with venv recreation for clean build
# pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
# ./build_metal.sh
# rm -rf python_env
# ./create_venv.sh
# source ./python_env/bin/activate
# popd >/dev/null


# This script reproduces the steps from .github/actions/build_cpp_extension_artifacts/action.yaml
# Run from the repository root. Expects environment variable CACHE_DIR to be set
# (in CI it's set to /root/.cache/cpp-extension-cache).

# echo "[1/6] Docker cleanup"
# docker system prune -a -f
# df -h

# echo "[2/6] Install system dependencies"
# apt upgrade -y && apt update -y
# apt install -y cmake python3 python3-venv python3-pip git-lfs ccache gcc-12 g++-12

# git config --global --add safe.directory /home/ubuntu/actions-runner/_work/pytorch2.0_ttnn/pytorch2.0_ttnn || true
# git config --global --add safe.directory /__w/pytorch2.0_ttnn/pytorch2.0_ttnn || true

# echo "[3/6] Prepare tt-metal dependencies (skip hugepages)"
# sed -i '/^configure_hugepages() {/,/^}/c\configure_hugepages() {\n    echo "Skip hugepages installation"\n}' ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh
# ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh

# DEBUG: DONE
echo "[4/6] Install Python dependencies"
# python3 -m ensurepip --upgrade || true
# python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
# python3 -m pip install -r requirements-dev.txt
# python3 -m pip install pytest-github-report
# # Match CI behavior: keep pip at 21.2.4 and avoid upgrading numpy to 2.x
# # python3 -m pip install --force-reinstall 'pip==21.2.4'
# # python3 -m pip install 'numpy>=1.24.4,<2' setuptools wheel
# python3 -m pip install -e .

# # echo "[5/6] Sync submodules and build C++ extensions"
# # git submodule sync
# # git submodule update --init --recursive
# # git submodule foreach 'git lfs fetch --all && git lfs pull'

pushd torch_ttnn/cpp_extension >/dev/null
# ./build_cpp_extension.sh
# + ./build_cpp_extension.sh content
#!/bin/bash

# parse build type
BUILD_TYPE=${1:-Release}
echo "> Build type: $BUILD_TYPE"

# Current directory
# CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# echo "> Current directory: $CUR_DIR"

# ABI flags used when compiling torch
# TORCH_ABI_FLAGS=$(python3 $TT_METAL_HOME/utils/get_torch_abi_flags.py)
TORCH_ABI_FLAGS=$(python3 utils/get_torch_abi_flags.py)
# TORCH_ABI_FLAGS=$(python3 $CUR_DIR/utils/get_torch_abi_flags.py)
echo "> TORCH_ABI_FLAGS: $TORCH_ABI_FLAGS"

# Configure ttnn
# TODO: check if c++17 is enough
echo "> Configuring ttnn"
cmake -B $TT_METAL_HOME/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX=$TT_METAL_HOME/build \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
    -DENABLE_CCACHE=TRUE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DTT_UNITY_BUILDS=ON \
    -DTT_ENABLE_LIGHT_METAL_TRACE=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_TOOLCHAIN_FILE=$TT_METAL_HOME/cmake/x86_64-linux-torch-toolchain.cmake \
    -DCMAKE_CXX_FLAGS="${TORCH_ABI_FLAGS} -std=c++20" \
    -S $TT_METAL_HOME

# cmake -B $CUR_DIR/third-party/tt-metal/build \
#     -G Ninja \
#     -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
#     -DCMAKE_INSTALL_PREFIX=$CUR_DIR/third-party/tt-metal/build \
#     -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
#     -DENABLE_CCACHE=TRUE \
#     -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
#     -DTT_UNITY_BUILDS=ON \
#     -DTT_ENABLE_LIGHT_METAL_TRACE=ON \
#     -DWITH_PYTHON_BINDINGS=ON \
#     -DCMAKE_TOOLCHAIN_FILE=$CUR_DIR/cmake/x86_64-linux-torch-toolchain.cmake \
#     -DCMAKE_CXX_FLAGS="${TORCH_ABI_FLAGS} -std=c++20" \
#     -S $CUR_DIR/third-party/tt-metal


# Build ttnn
echo "> Building ttnn"
ninja -C $TT_METAL_HOME/build install

pip3 install -e $TT_METAL_HOME/

# export TT_METAL_HOME=$CUR_DIR/third-party/tt-metal
echo "> TT_METAL_HOME: $TT_METAL_HOME"
echo "> Building cpp extension"
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE};-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER=g++-12;-DCMAKE_C_COMPILER=gcc-12" python3 setup.py develop

# - ./build_cpp_extension.sh content
popd >/dev/null

echo "[6/6] Collect build artifacts into CACHE_DIR=$CACHE_DIR"
pushd torch_ttnn/cpp_extension >/dev/null
PYTHON_LIB_SUFFIX=$(python3 -c "import importlib.machinery; print(importlib.machinery.EXTENSION_SUFFIXES[0])")
mkdir -p "$CACHE_DIR"
CCACHE_DIR=$(ccache --get-config cache_dir)

# cpp-extension cache
cp -r build "$CACHE_DIR"
cp -r "ttnn_device_extension${PYTHON_LIB_SUFFIX}" "$CACHE_DIR"

# tt-metal cache
mkdir -p "$CACHE_DIR/tt-metal"
cp -r $TT_METAL_HOME/build "$CACHE_DIR/tt-metal/"
cp -r $TT_METAL_HOME/.cpmcache "$CACHE_DIR/tt-metal/"
# mkdir -p "$CACHE_DIR/tt-metal"
# cp -r third-party/tt-metal/build "$CACHE_DIR/tt-metal/"
# cp -r third-party/tt-metal/.cpmcache "$CACHE_DIR/tt-metal/"

# ccache
cp -r "$CCACHE_DIR" "$CACHE_DIR"
popd >/dev/null

echo "Done."


