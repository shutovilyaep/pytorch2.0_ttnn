#!/usr/bin/env bash

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
unset Boost_DIR BOOST_ROOT
rm -rf .cpmcache build build_Release build_Debug ~/.cache/tt-metal-cache /tmp/tt-metal-cache
./build_metal.sh
rm -rf python_env
export PYTHON_ENV_DIR="$(pwd)/python_env"
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null
