#!/usr/bin/env bash

set -euo pipefail

# Workspace directory
WORKSPACE_DIR="/home/ilia_shutov/pytorch2.0_ttnn.dev"
# Previous container path for reference:
# WORKSPACE_DIR="/workspace/pytorch2.0_ttnn"

# Added pre-build steps with venv recreation for clean build
pushd "${WORKSPACE_DIR}/" >/dev/null
source "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/third-party/tt-metal/python_env/bin/activate"
export TT_METAL_HOME="${WORKSPACE_DIR}/torch_ttnn/cpp_extension/third-party/tt-metal"
export TT_METAL_KERNEL_PATH="${TT_METAL_HOME}"
export CARGO_HOME="${HOME}/.cargo"; export RUSTUP_HOME="${HOME}/.rustup"; export PATH="${HOME}/.cargo/bin:${PATH}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
# Clean previous extension build artifacts to avoid stale CMake cache
rm -rf "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/build" || true

pip install -e . --use-pep517 --no-cache-dir --no-build-isolation

popd >/dev/null

pushd "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/" >/dev/null
# Ensure ccache exists (build system invokes it)
if ! command -v ccache >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ccache
fi
# ./build_cpp_extension.sh
./build_cpp_extension.sh RelWithDebInfo

# # Hardcoded host workspace mapping (used to rewrite compile_commands for host IDE)
# HOST_WORKSPACE="/home/kilka/Projects/ML/TT-NN/dev.docker/workspace"

# # Export compile_commands.json to workspace root for editor tooling
# if [ -f "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json" ]; then
#   cp "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json" "${WORKSPACE_DIR}/compile_commands.json"
#   # Optionally emit host-adjusted compile_commands.json if HOST_WORKSPACE is provided
#   if [ -n "${HOST_WORKSPACE:-}" ]; then
#     python3 "${WORKSPACE_DIR}/scripts/rewrite_compile_commands.py" \
#       "${WORKSPACE_DIR}/compile_commands.json" \
#       "${WORKSPACE_DIR}/compile_commands.host.json" \
#       /workspace \
#       ${HOST_WORKSPACE}
#   fi
# fi

popd >/dev/null

python "${WORKSPACE_DIR}/scripts/test_script.py"

# pushd "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/third-party/tt-metal/build/bin" >/dev/null
# ./tt-metal-trace
# popd >/dev/null
