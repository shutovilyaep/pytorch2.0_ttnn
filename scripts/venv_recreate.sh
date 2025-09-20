#!/usr/bin/env bash

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
unset Boost_DIR BOOST_ROOT
export CARGO_HOME="${HOME}/.cargo"
export RUSTUP_HOME="${HOME}/.rustup"
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
rm -rf .cpmcache build build_Release build_Debug ~/.cache/tt-metal-cache /tmp/tt-metal-cache
./build_metal.sh
rm -rf python_env
export PYTHON_ENV_DIR="$(pwd)/python_env"
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null
