#!/usr/bin/env bash

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
./build_metal.sh
rm -rf python_env
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null
