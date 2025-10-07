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

# Configure environment
export TT_METAL_HOME=${TT_METAL_HOME:-$CUR_DIR/third-party/tt-metal}
echo "> TT_METAL_HOME: $TT_METAL_HOME"

# Verify tt-metal build artifacts exist (built by scripts/venv_recreate.sh or upstream), do not build here
if [ ! -f "$TT_METAL_HOME/build/lib/libtt_metal.so" ] || [ ! -f "$TT_METAL_HOME/build/lib/_ttnn.so" ] || [ ! -f "$TT_METAL_HOME/build/lib/_ttnncpp.so" ]; then
  echo "[ERROR] tt-metal build artifacts not found in $TT_METAL_HOME/build/lib"
  echo "        Expected: libtt_metal.so, _ttnn.so, _ttnncpp.so"
  echo "        Please run: ./scripts/venv_recreate.sh (or tt-metal/build_metal.sh) before building the extension."
  exit 1
fi

# Ensure Python package for tt-metal is present (editable install), but don't rebuild C++
if ! python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ttnn') else 1)"; then
  echo "> Installing python package: ttnn (editable)"
  pip3 install -e $CUR_DIR/third-party/tt-metal/
fi

echo "> Building cpp extension"
# Если правим подмодуль tt-metal и хотим отстроить его внутри CMake расширения:
# export CMAKE_FLAGS="${CMAKE_FLAGS:-};-DENABLE_SUBMODULE_TT_METAL_BUILD=ON"
# Включаем AVX2 через x86-64-v3 для всех целей (и C, и C++)
CMAKE_FLAGS="-DENABLE_SUBMODULE_TT_METAL_BUILD=ON;-DCMAKE_BUILD_TYPE=${BUILD_TYPE};-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER=g++-12;-DCMAKE_C_COMPILER=gcc-12;-DCMAKE_CXX_FLAGS=-march=x86-64-v3;-DCMAKE_C_FLAGS=-march=x86-64-v3;${CMAKE_FLAGS}" python3 setup.py develop
