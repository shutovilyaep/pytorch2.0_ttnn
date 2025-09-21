#!/usr/bin/env bash

# Added pre-build steps with venv recreation for clean build
pushd /workspace/pytorch2.0_ttnn/ >/dev/null
source /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/python_env/bin/activate
export TT_METAL_HOME="/workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal"
export TT_METAL_KERNEL_PATH="${TT_METAL_HOME}"
export CARGO_HOME="${HOME}/.cargo"; export RUSTUP_HOME="${HOME}/.rustup"; export PATH="${HOME}/.cargo/bin:${PATH}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
pip install -e . --use-pep517 --no-cache-dir --no-build-isolation

popd >/dev/null

pushd /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/ >/dev/null
# Ensure ccache exists (build system invokes it)
if ! command -v ccache >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ccache
fi
# ./build_cpp_extension.sh
./build_cpp_extension.sh RelWithDebInfo

# Export compile_commands.json to workspace root for editor tooling
if [ -f /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json ]; then
  cp /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json /workspace/compile_commands.json
fi

popd >/dev/null

python /workspace/pytorch2.0_ttnn/scripts/test_script.py

# pushd /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/build/bin >/dev/null
# ./tt-metal-trace
# popd >/dev/null
