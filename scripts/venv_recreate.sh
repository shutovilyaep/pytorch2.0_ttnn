#!/usr/bin/env bash

# Ensure Rust toolchain is available
export CARGO_HOME="${HOME}/.cargo"
export RUSTUP_HOME="${HOME}/.rustup"
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
export PATH="${HOME}/.cargo/bin:${PATH}"
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89.0
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi
rustup default 1.89.0
rustc --version && cargo --version

# Harden pip to avoid cache/corrupted downloads
export PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
export PIP_RETRIES=${PIP_RETRIES:-5}
python3 -m pip cache purge || true

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
unset Boost_DIR BOOST_ROOT
# Ensure ccache is available (build uses it)
if ! command -v ccache >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ccache
fi
rm -rf .cpmcache build build_Release build_Debug ~/.cache/tt-metal-cache /tmp/tt-metal-cache
./build_metal.sh --without-distributed --clean --build-type RelWithDebInfo --build-tests
# ./build_metal.sh
rm -rf python_env
export PYTHON_ENV_DIR="$(pwd)/python_env"
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null
