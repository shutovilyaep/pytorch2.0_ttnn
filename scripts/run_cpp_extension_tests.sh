#!/usr/bin/env bash

set -euo pipefail

# Configurable vars
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
TT_METAL_HOME_DEFAULT="$REPO_ROOT/torch_ttnn/cpp_extension/third-party/tt-metal"
USE_EXTERNAL_TT_METAL=0
if [[ "${1:-}" == "--external" ]]; then
  USE_EXTERNAL_TT_METAL=1
  shift || true
fi

echo "> Repo root: $REPO_ROOT"

# If a different venv is active, deactivate it to avoid ABI mismatches
if type -t deactivate >/dev/null 2>&1; then
  deactivate || true
fi

# 1) Point to the embedded tt-metal unless explicitly told to use external
if [[ "$USE_EXTERNAL_TT_METAL" -eq 1 ]]; then
  if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "[ERROR] --external set but TT_METAL_HOME is empty" >&2
    exit 1
  fi
else
  export TT_METAL_HOME="$TT_METAL_HOME_DEFAULT"
fi
echo "> TT_METAL_HOME: $TT_METAL_HOME"

# 2) Activate tt-metal's venv
if [[ ! -f "$TT_METAL_HOME/python_env/bin/activate" ]]; then
    echo "[ERROR] tt-metal virtualenv not found at $TT_METAL_HOME/python_env. Did you run create_venv.sh?" >&2
    exit 1
fi
source "$TT_METAL_HOME/python_env/bin/activate"
echo "> Activated venv: $(python -c 'import sys; print(sys.prefix)')"

# 2.5) Unset cluster envs that can force distributed YAML parsing and break single-device opens
unset TT_METAL_CLUSTER_DESC TT_METAL_CLUSTER_CONFIG TT_METAL_DEVICES TT_METAL_HOST_RANKS TT_METAL_HOSTS || true
# Also remove LD_LIBRARY_PATH to avoid mixing libtorch/libstdc++ from other envs
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "> Clearing LD_LIBRARY_PATH to avoid ABI conflicts"
  unset LD_LIBRARY_PATH
fi

# 3) Ensure Python can import ttnn (from tt-metal) and torch_ttnn (from this repo)
export PYTHONPATH="$TT_METAL_HOME:$REPO_ROOT:${PYTHONPATH:-}"
echo "> PYTHONPATH: $PYTHONPATH"

python - <<'PY'
import sys
print('sys.path[0:6]=', sys.path[:6])
import ttnn, torch_ttnn, torch
print('ttnn at:', ttnn.__file__)
print('torch_ttnn at:', torch_ttnn.__file__)
print('torch at:', torch.__file__)
print('torch version:', torch.__version__)
import pathlib
print('torch lib dir:', pathlib.Path(torch.__file__).parent / 'lib')
PY

# Ensure dynamic loader finds the active venv's torch libraries first
TORCH_LIB_DIR=$(python - <<'PY'
import torch, pathlib
print(pathlib.Path(torch.__file__).parent / 'lib')
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "> LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 4) Rebuild the C++ extension to match the active torch (always, to avoid ABI drift)
pushd "$REPO_ROOT/torch_ttnn/cpp_extension" >/dev/null
  BUILD_TYPE=${1:-RelWithDebInfo}
  ./build_cpp_extension.sh "$BUILD_TYPE"
popd >/dev/null

# Diagnose linkage without importing (in case import crashes)
EXT_SO_PATH=$(ls -1 "$REPO_ROOT/torch_ttnn/cpp_extension"/ttnn_device_extension*.so 2>/dev/null | head -n1 || true)
if [[ -n "$EXT_SO_PATH" ]]; then
  echo "> ttnn_device_extension path: $EXT_SO_PATH"
  echo "> ldd of extension:"
  ldd "$EXT_SO_PATH" | sort || true
fi

# Try import now
python - <<'PY' || true
try:
    import ttnn_device_extension as _ext
    print('ttnn_device_extension OK:', getattr(_ext, '__file__', '<builtin>'))
except Exception as e:
    print('Import failed:', e)
PY

# 5) Run the cpp_extension tests
pushd "$REPO_ROOT" >/dev/null
  echo "> Running tests/cpp_extension ..."
  pytest -q tests/cpp_extension -s
popd >/dev/null

echo "> Done."


