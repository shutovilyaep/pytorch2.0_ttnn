#!/usr/bin/env bash

set -euo pipefail
trap 'echo "ERROR: команда завершилась с ошибкой: $BASH_COMMAND"; exit 1' ERR

# Полная локальная репликация job `cpp-extension-tests` из
# .github/workflows/run-cpp-native-tests.yaml (1-в-1 команды).
# Запускать из любого места — скрипт сам перейдет в корень репозитория.

# Определяем корень репозитория (рядом со scripts/direct.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Activate tt-metal venv if it exists and set Python executable
TT_METAL_VENV="${REPO_ROOT}/torch_ttnn/cpp_extension/third-party/tt-metal/python_env"
if [ -f "${TT_METAL_VENV}/bin/activate" ]; then
    echo "> Activating tt-metal venv: ${TT_METAL_VENV}"
    source "${TT_METAL_VENV}/bin/activate"
    PYTHON="${TT_METAL_VENV}/bin/python3"
    echo "> Python: ${PYTHON}"
    # Upgrade pip to support modern pyproject.toml editable installs
    echo "> Upgrading pip in venv for editable install support"
    "${PYTHON}" -m pip install --upgrade pip setuptools wheel --quiet || true
else
    echo "> Warning: tt-metal venv not found at ${TT_METAL_VENV}"
    PYTHON="$(which python3)"
    echo "> Using system Python: ${PYTHON}"
fi

# Ensure PYTHON is set
PYTHON="${PYTHON:-python3}"

# Эмуляция переменных окружения GitHub Actions
export GITHUB_WORKSPACE="${REPO_ROOT}"
export GITHUB_ENV="${REPO_ROOT}/.github_env"
: > "${GITHUB_ENV}"

###############################################################################
# Step: Install git-lfs for checkout
###############################################################################
# apt-get update -y
# apt-get install -y git-lfs
# git lfs install

# Step: actions/checkout@v4 (эквивалентно: LFS + полная история + сабмодули)
# Прямого аналога нет, используем наиболее близкие команды

# TODO: NOW: Temp disable sync not to overwrite build_metal.sh
# git lfs fetch --all || true
# git lfs pull || true
# if git rev-parse --is-shallow-repository >/dev/null 2>&1; then
#   git fetch --unshallow || true
# fi
# git submodule sync --recursive
# git submodule update --init --recursive

###############################################################################
# Step: Update system
###############################################################################
# apt update -y && apt upgrade -y
# apt install -y curl jq

###############################################################################
# Step: Update .gitsubmodules
###############################################################################
(
  # Set pinned tt-metal ref here. Leave empty to auto-detect latest prerelease.
  # Should be set manually with each PR required to make it work with new tt-metal version
  TT_METAL_REF="v0.63.0"

  if [ -z "${TT_METAL_REF}" ]; then
    # Auto-detect latest prerelease tag (e.g., v0.64.0-rc8)
    latest_pre_release=$(curl -s https://api.github.com/repos/tenstorrent/tt-metal/releases | jq -r '[.[] | select(.prerelease == true)][0].tag_name')
    RESOLVED_TT_METAL_REF="$latest_pre_release"
    echo "Auto-detected latest prerelease tag: $RESOLVED_TT_METAL_REF"
  else
    RESOLVED_TT_METAL_REF="$TT_METAL_REF"
    echo "Using pinned tt-metal ref: $RESOLVED_TT_METAL_REF"
  fi

  # Export for later steps (эмулируем GHA через файл окружения)
  echo "TT_METAL_REF=$RESOLVED_TT_METAL_REF" >> "$GITHUB_ENV"
)
# Применяем значения из GITHUB_ENV
set -a; source "$GITHUB_ENV"; set +a

###############################################################################
# Step: Setup submodules
###############################################################################
# Allow git operations in runner workspace paths
# git config --global --add safe.directory /__w/pytorch2.0_ttnn/pytorch2.0_ttnn || true
# git config --global --add safe.directory "${GITHUB_WORKSPACE}" || true
# git config --global --add safe.directory "$(pwd)" || true

# Retry helper with exponential backoff
retry() {
  local max_attempts=${1:-5}; shift
  local attempt=1
  until "$@"; do
    if (( attempt >= max_attempts )); then
      echo "Command failed after ${attempt} attempts: $*"
      return 1
    fi
    echo "Command failed (attempt ${attempt}/${max_attempts}): $*"
    sleep $(( 2 ** attempt ))
    ((attempt++))
  done
}

# TODO: NOW: Temp disable sync not to overwrite build_metal.sh
# # Be tolerant to network hiccups for submodule operations
# retry 5 git submodule sync --recursive
# retry 5 git submodule update --init --recursive

# # Check out the requested tt-metal ref inside the submodule without editing .gitmodules
# pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
# # Avoid tag clobber issues if upstream moved tags; clean local tags and fetch with force
# git config fetch.prune true || true
# git config fetch.pruneTags true || true
# git tag -l | xargs -r -n 1 git tag -d || true
# retry 5 git -c protocol.version=2 fetch --all --tags --force --prune --prune-tags
# if echo "$TT_METAL_REF" | grep -q '^releases/'; then
#   # Treat as branch on origin
#   if git rev-parse --verify "$TT_METAL_REF" >/dev/null 2>&1; then
#     retry 5 git checkout -f "$TT_METAL_REF"
#   else
#     retry 5 git checkout -B "$TT_METAL_REF" "origin/$TT_METAL_REF" || retry 5 git checkout -f "origin/$TT_METAL_REF"
#   fi
# else
#   # Treat as tag or plain ref
#   retry 5 git -c advice.detachedHead=false checkout -f "tags/$TT_METAL_REF" || retry 5 git checkout -f "$TT_METAL_REF"
# fi
# # Update nested submodules to the versions recorded by the checked-out ref
# retry 5 git submodule sync --recursive
# retry 5 git submodule update --init --recursive
# # LFS for all nested submodules
# git submodule foreach --recursive 'git lfs fetch --all || true'
# git submodule foreach --recursive 'git lfs pull || true'
# popd >/dev/null

###############################################################################
# Step: Install dependencies
###############################################################################
# apt upgrade -y && apt update -y
# apt install -y cmake python3 python3-venv python3-pip git-lfs ccache
# git config --global --add safe.directory /home/ubuntu/actions-runner/_work/pytorch2.0_ttnn/pytorch2.0_ttnn
# git config --global --add safe.directory /__w/pytorch2.0_ttnn/pytorch2.0_ttnn

# Remove hugepages setup from install_dependencies.sh
# sed -i '/^configure_hugepages() {/,/^}/c\\configure_hugepages() {\n    echo "Skip hugepages installation"\n}' ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh
# ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh

###############################################################################
# Step: Install python dependencies
###############################################################################
# python3 -m pip install --upgrade pip
"${PYTHON}" -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
"${PYTHON}" -m pip install pytest-github-report

###############################################################################
# Step: Install root package
# NOTE: tt-metal will be built and installed via pip install in the next step
# (similar to CI workflow - no separate build step needed)
###############################################################################
# Upgrade pip to support editable installs with pyproject.toml
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -e .[dev]

###############################################################################
# Step: Build C++ Extension
# NOTE: For local testing, use PyPI ttnn package (pre-built) to avoid build issues
# In CI, tt-metal is built from source via pip install (see YAML workflow)
###############################################################################
export TT_METAL_HOME=$(realpath ./torch_ttnn/cpp_extension/third-party/tt-metal)

# Determine tt-metal version for PyPI package
TT_METAL_VERSION=""
if [ -d "${TT_METAL_HOME}/.git" ]; then
  TT_METAL_VERSION=$(cd "${TT_METAL_HOME}" && git describe --abbrev=0 --tags 2>/dev/null | sed 's/^v//' || echo "")
fi
if [ -z "${TT_METAL_VERSION}" ] && [ -n "${TT_METAL_REF:-}" ]; then
  TT_METAL_VERSION="${TT_METAL_REF#v}"
fi
if [ -z "${TT_METAL_VERSION}" ]; then
  TT_METAL_VERSION="0.63.0"
  echo "Warning: Could not determine tt-metal version, using fallback: ${TT_METAL_VERSION}"
else
  echo "Detected tt-metal version: ${TT_METAL_VERSION}"
fi

# Ensure PEP517 backend & native build tools are available for pyproject builds
"${PYTHON}" -m pip install --upgrade scikit-build-core cmake ninja

# Install clang-format system package (required by tt-metal codegen during CMake build)
# Even though we use PyPI ttnn, CMake still builds tt-metal from submodule
# Note: Python clang-format package conflicts with system clang-format, so we need system version
if ! command -v clang-format >/dev/null 2>&1; then
  echo "Warning: clang-format not found in PATH. Attempting to install..."
  # Try to install via apt if available (requires sudo, may fail)
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y clang-format || {
      echo "Warning: Failed to install clang-format via apt-get"
    }
  else
    echo "Warning: apt-get not available, cannot install clang-format automatically"
  fi
fi
# Ensure clang-format is in PATH (venv may have wrong clang-format script)
if command -v clang-format >/dev/null 2>&1; then
  CLANG_FORMAT_PATH=$(which clang-format)
  # Check if it's a Python script (wrong one) or binary (correct one)
  if head -1 "${CLANG_FORMAT_PATH}" | grep -q "python"; then
    echo "Warning: Found Python clang-format script instead of binary, may cause issues"
    # Try to find system clang-format
    if [ -f "/usr/bin/clang-format" ]; then
      export PATH="/usr/bin:${PATH}"
    elif [ -f "/usr/local/bin/clang-format" ]; then
      export PATH="/usr/local/bin:${PATH}"
    fi
  fi
fi

# Install ttnn from PyPI pinned to TT_METAL_REF (e.g., v0.63.0 -> 0.63.0)
# This avoids build issues with tt-metal v0.63.0 (CMake export errors, example linking issues)
# CI uses source build, but for local testing PyPI is more reliable
# NOTE: CMakeLists.txt still builds tt-metal from submodule, so we need clang-format above
"${PYTHON}" -m pip uninstall -y ttnn || true
if [[ -n "${TT_METAL_REF:-}" ]]; then
  TTNN_PYPI_VER="${TT_METAL_REF#v}"
  "${PYTHON}" -m pip install --upgrade --no-build-isolation "ttnn==${TTNN_PYPI_VER}" || {
    echo "Не удалось установить ttnn==${TTNN_PYPI_VER}, пробую без пина";
    "${PYTHON}" -m pip install --upgrade --no-build-isolation ttnn;
  }
else
  "${PYTHON}" -m pip install --upgrade --no-build-isolation ttnn==0.63.0
fi

# Ensure CMake can locate Torch package config from current Python
set +e
_CMAKE_PREFIX_PATH=$("${PYTHON}" - <<'PY'
try:
    import torch
    print(torch.utils.cmake_prefix_path)
except Exception:
    print("")
PY
)
set -e
export CMAKE_PREFIX_PATH="${_CMAKE_PREFIX_PATH}"

# Minimal CMake args for our extension (no tt-metal toolchain needed when using PyPI)
if command -v clang-17 >/dev/null 2>&1; then
  export CC=clang-17
  export CXX=clang++-17
fi
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# Make sure runtime can locate ttnn shared libs (manylinux wheels may already embed rpaths)
set +e
export TTNN_LIB_DIR=$("${PYTHON}" - <<'PY'
import pathlib, sys
try:
    import ttnn
    p = pathlib.Path(ttnn.__file__).parent
    # Try common locations; ignore if missing
    cands = [p.joinpath('build','lib'), p]
    for c in cands:
        if c.exists():
            print(str(c))
            break
except Exception:
    print("")
PY
)
set -e
[ -n "${TTNN_LIB_DIR:-}" ] && export LD_LIBRARY_PATH="${TTNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Early sanity check: ensure 'ttnn' and its pybind module resolve
# Add MPI library path temporarily for this check (required by ttnn package's _ttnncpp.so)
if [ -d "/opt/openmpi-v5.0.7-ulfm/lib" ]; then
  export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:${LD_LIBRARY_PATH:-}"
fi
"${PYTHON}" -c "import importlib, os; print('TT_METAL_HOME=', os.environ.get('TT_METAL_HOME')); import ttnn; importlib.import_module('ttnn._ttnn'); print('import ttnn OK')" || echo "Warning: ttnn import check failed, continuing anyway"

cd torch_ttnn/cpp_extension
# Ensure pip is up to date for editable installs
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
# Builds and installs our project C++ extension package 'torch_ttnn_cpp_extension'
"${PYTHON}" -m pip install -e .
cd "${GITHUB_WORKSPACE}"

###############################################################################
# Step: Run C++ Extension Tests
###############################################################################
# Test phase: imports 'ttnn' (from PyPI) and our 'torch_ttnn_cpp_extension'.
# Ensure runtime can locate shared libraries (libtt_metal.so, libtt_stl.so, etc.)
export TT_METAL_HOME="${GITHUB_WORKSPACE}/torch_ttnn/cpp_extension/third-party/tt-metal"
export TTNN_LIB_PATHS=$("${PYTHON}" - <<'PY'
import pathlib
try:
    import ttnn
    p = pathlib.Path(ttnn.__file__).parent
    candidates = []
    for d in [p / 'build' / 'lib', p / '.libs', p]:
        if d.exists():
            candidates.append(str(d))
    print(':'.join(candidates))
except Exception:
    print('')
PY
)
if [ -n "${TTNN_LIB_PATHS}" ]; then
  export LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${TT_METAL_HOME}/build/lib:${TTNN_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${TT_METAL_HOME}/build/lib:${TT_METAL_HOME}/build_Release/tt_stl:${TT_METAL_HOME}/build/tt_stl:${LD_LIBRARY_PATH:-}"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Ensure libtt_stl.so and libtt_metal.so are discoverable: scan ttnn & tt-metal and append their dirs
STL_LIB_DIR=$("${PYTHON}" - <<'PY'
import pathlib, site
def find_in_ttnn():
    try:
        import ttnn
        p = pathlib.Path(ttnn.__file__).parent
        for d in [p/'build'/'lib', p/'.libs', p]:
            f = d/'libtt_stl.so'
            if f.exists():
                print(str(d)); return
    except Exception:
        pass
    # site-packages fallback
    for sp in set(site.getsitepackages()+[site.getusersitepackages()]):
        d = pathlib.Path(sp)/'ttnn'/'build'/'lib'
        if (d/'libtt_stl.so').exists():
            print(str(d)); return
        d2 = pathlib.Path(sp)/'ttnn'
        if (d2/'libtt_stl.so').exists():
            print(str(d2)); return
find_in_ttnn()
PY
)
if [ -n "${STL_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${STL_LIB_DIR}:${LD_LIBRARY_PATH}"
else
  # search in tt-metal tree as a last resort
  STL_FROM_TTM=$(find "${TT_METAL_HOME}" -maxdepth 8 -type f -name 'libtt_stl.so' 2>/dev/null | head -n1 || true)
  if [ -n "${STL_FROM_TTM}" ]; then
    export LD_LIBRARY_PATH="$(dirname "${STL_FROM_TTM}"):${LD_LIBRARY_PATH}"
  fi
fi

echo "Running C++ extension tests"
"${PYTHON}" -m pytest tests/cpp_extension/test_cpp_extension_functionality.py -v

echo "Running BERT C++ extension tests"
"${PYTHON}" -m pytest tests/cpp_extension/test_bert_cpp_extension.py -v

# echo "Running models C++ extension tests"
# pytest tests/models/ --native_integration -v

echo "Running tests passed"


