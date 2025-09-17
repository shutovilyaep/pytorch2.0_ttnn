#!/usr/bin/env bash

export TT_METAL_HOME=$(realpath torch_ttnn/cpp_extension/third-party/tt-metal)
echo "> TT_METAL_HOME: $TT_METAL_HOME"
source $TT_METAL_HOME/python_env/bin/activate

pushd torch_ttnn/cpp_extension/
./build_cpp_extension.sh
popd

python scripts/test_script.py
