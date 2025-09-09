#!/bin/bash

# Prefer prebuilt ttnn/tt-metal from the active venv; fall back to submodule if not found
# export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$TT_METAL_HOME/build/lib64:$LD_LIBRARY_PATH"

# Use venv headers and libs to ensure ABI compatibility with installed ttnn
export TT_METAL_HOME=/home/epam/shutov/.venv/lib/python3.10/site-packages/ttnn
export TTNN_LIB_DIR=${TT_METAL_HOME}/build/lib

echo "> TT_METAL_HOME: $TT_METAL_HOME"
echo "> TTNN_LIB_DIR: $TTNN_LIB_DIR"

source /home/epam/shutov/.venv/bin/activate

# parse build type
BUILD_TYPE=${1:-Release}
echo "> Build type: $BUILD_TYPE"

# Current directory
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "> Current directory: $CUR_DIR"

# # ABI flags used when compiling torch
# TORCH_ABI_FLAGS=$(python3 $CUR_DIR/utils/get_torch_abi_flags.py)
# echo "> TORCH_ABI_FLAGS: $TORCH_ABI_FLAGS"

# # Configure ttnn
# # TODO: check if c++17 is enough
# echo "> Configuring ttnn"
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


# # Build ttnn
# echo "> Building ttnn"
# ninja -C $CUR_DIR/third-party/tt-metal/build install

# pip3 install -e $CUR_DIR/third-party/tt-metal/

pip3 install -r /home/epam/shutov/pytorch2.0_ttnn/requirements-dev.txt

pushd /home/epam/shutov/pytorch2.0_ttnn/torch_ttnn/cpp_extension

echo "> Building cpp extension"
# Force using submodule headers and venv libs for linking
# Use clang-17 to avoid GCC11 ICEs and minimize ABI/linker issues
export CC=clang-17
export CXX=clang++-17
export CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE};-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache;-DCMAKE_C_COMPILER=clang-17;-DCMAKE_CXX_COMPILER=clang++-17;-DENABLE_LOCAL_TT_METAL_BUILD=ON;-DTTNN_LIB_DIR=${TTNN_LIB_DIR};${CMAKE_FLAGS}"
python3 setup.py develop

popd
