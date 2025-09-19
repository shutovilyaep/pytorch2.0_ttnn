#!/usr/bin/env bash

export TT_METAL_HOME=$(realpath torch_ttnn/cpp_extension/third-party/tt-metal)
echo "> TT_METAL_HOME: $TT_METAL_HOME"

rm -rf torch_ttnn/cpp_extension/build
rm -rf torch_ttnn/cpp_extension/third-party/tt-metal/build

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
./build_metal.sh
# ./build_metal.sh --enable-ccache
rm -rf python_env
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null
