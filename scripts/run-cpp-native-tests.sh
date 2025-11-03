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

# Activate tt-metal venv if it exists
TT_METAL_VENV="${REPO_ROOT}/torch_ttnn/cpp_extension/third-party/tt-metal/python_env"
if [ -f "${TT_METAL_VENV}/bin/activate" ]; then
    echo "> Activating tt-metal venv: ${TT_METAL_VENV}"
    source "${TT_METAL_VENV}/bin/activate"
    echo "> Python: $(which python3)"
    # Upgrade pip to support modern pyproject.toml editable installs
    echo "> Upgrading pip in venv for editable install support"
    python3 -m pip install --upgrade pip setuptools wheel --quiet || true
else
    echo "> Warning: tt-metal venv not found at ${TT_METAL_VENV}"
    echo "> Continuing with system Python: $(which python3)"
fi

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
git lfs fetch --all || true
git lfs pull || true
if git rev-parse --is-shallow-repository >/dev/null 2>&1; then
  git fetch --unshallow || true
fi
git submodule sync --recursive
git submodule update --init --recursive

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
  TT_METAL_REF="v0.60.1"

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

# Be tolerant to network hiccups for submodule operations
retry 5 git submodule sync --recursive
retry 5 git submodule update --init --recursive

# Check out the requested tt-metal ref inside the submodule without editing .gitmodules
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
# Avoid tag clobber issues if upstream moved tags; clean local tags and fetch with force
git config fetch.prune true || true
git config fetch.pruneTags true || true
git tag -l | xargs -r -n 1 git tag -d || true
retry 5 git -c protocol.version=2 fetch --all --tags --force --prune --prune-tags
if echo "$TT_METAL_REF" | grep -q '^releases/'; then
  # Treat as branch on origin
  if git rev-parse --verify "$TT_METAL_REF" >/dev/null 2>&1; then
    retry 5 git checkout -f "$TT_METAL_REF"
  else
    retry 5 git checkout -B "$TT_METAL_REF" "origin/$TT_METAL_REF" || retry 5 git checkout -f "origin/$TT_METAL_REF"
  fi
else
  # Treat as tag or plain ref
  retry 5 git -c advice.detachedHead=false checkout -f "tags/$TT_METAL_REF" || retry 5 git checkout -f "$TT_METAL_REF"
fi
# Update nested submodules to the versions recorded by the checked-out ref
retry 5 git submodule sync --recursive
retry 5 git submodule update --init --recursive
# LFS for all nested submodules
git submodule foreach --recursive 'git lfs fetch --all || true'
git submodule foreach --recursive 'git lfs pull || true'
popd >/dev/null

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
python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python3 -m pip install pytest-github-report

###############################################################################
# Step: Build tt-metal native libs (pre-req for libtt_metal.so at runtime)
# NOTE: Если это требуется в CI, перенесите этот блок в YAML.
###############################################################################
export TT_METAL_HOME=$(realpath ./torch_ttnn/cpp_extension/third-party/tt-metal)
if [ -d "${TT_METAL_HOME}" ]; then
  echo "Building tt-metal native libraries at ${TT_METAL_HOME}"
  pushd "${TT_METAL_HOME}" >/dev/null
  # Используем clang-17 если доступен
  if command -v clang-17 >/dev/null 2>&1; then
    export CC=clang-17
    export CXX=clang++-17
  fi
  # Сборка релизной конфигурации и включение shared sub-libs для TTNN (нужно libtt_stl.so)
  ./build_metal.sh --build-type Release --ttnn-shared-sub-libs --enable-ccache | tee build-metal.log
  popd >/dev/null
  # Экспортируем пути для рантайма
  export LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${TT_METAL_HOME}/build/lib:${TT_METAL_HOME}/build_Release/tt_stl:${TT_METAL_HOME}/build/tt_stl:${LD_LIBRARY_PATH:-}"

  # Если libtt_stl.so отсутствует, попытаться дособрать целевую библиотеку и добавить её директорию
  if ! find "${TT_METAL_HOME}/build_Release" "${TT_METAL_HOME}/build" -type f -name 'libtt_stl.so' -print -quit >/dev/null 2>&1; then
    if [ -d "${TT_METAL_HOME}/build_Release" ]; then
      pushd "${TT_METAL_HOME}/build_Release" >/dev/null || true
      cmake --build . --target tt_stl -j"$(nproc)" || true
      popd >/dev/null || true
    elif [ -d "${TT_METAL_HOME}/build" ]; then
      pushd "${TT_METAL_HOME}/build" >/dev/null || true
      cmake --build . --target tt_stl -j"$(nproc)" || true
      popd >/dev/null || true
    fi
  fi

  STL_FROM_TTM=$(find "${TT_METAL_HOME}/build_Release" "${TT_METAL_HOME}/build" -type f -name 'libtt_stl.so' 2>/dev/null | head -n1 || true)
  if [ -n "${STL_FROM_TTM}" ]; then
    export LD_LIBRARY_PATH="$(dirname "${STL_FROM_TTM}"):${LD_LIBRARY_PATH}"
  fi
else
  echo "WARN: TT_METAL_HOME not found at ${TT_METAL_HOME}; libtt_metal.so может быть недоступен"
fi

###############################################################################
# Step: Install root package
###############################################################################
# Upgrade pip to support editable installs with pyproject.toml
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .[dev]

###############################################################################
# Step: Build C++ Extension (use PyPI ttnn, do NOT build tt-metal from source)
###############################################################################
# Ensure PEP517 backend & native build tools are available for pyproject builds
python3 -m pip install --upgrade scikit-build-core cmake ninja

# Install ttnn from PyPI pinned to TT_METAL_REF (e.g., v0.60.1 -> 0.60.1)
if [[ -n "${TT_METAL_REF:-}" ]]; then
  TTNN_PYPI_VER="${TT_METAL_REF#v}"
  python3 -m pip install --upgrade --no-build-isolation "ttnn==${TTNN_PYPI_VER}" || {
    echo "Не удалось установить ttnn==${TTNN_PYPI_VER}, пробую без пина";
    python3 -m pip install --upgrade --no-build-isolation ttnn;
  }
else
  python3 -m pip install --upgrade --no-build-isolation ttnn==0.60.1
fi

# Ensure CMake can locate Torch package config from current Python
set +e
_CMAKE_PREFIX_PATH=$(python3 - <<'PY'
try:
    import torch
    print(torch.utils.cmake_prefix_path)
except Exception:
    print("")
PY
)
set -e
export CMAKE_PREFIX_PATH="${_CMAKE_PREFIX_PATH}"

# Prefer clang-17 if available; otherwise fall back to system compilers
if command -v clang-17 >/dev/null 2>&1; then
  export CC=clang-17
  export CXX=clang++-17
fi

# Minimal CMake args for our extension
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# Make sure runtime can locate ttnn shared libs (manylinux wheels may already embed rpaths)
set +e
export TTNN_LIB_DIR=$(python3 - <<'PY'
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

cd torch_ttnn/cpp_extension
# Ensure pip is up to date for editable installs
python3 -m pip install --upgrade pip setuptools wheel
# Builds and installs our project C++ extension package 'torch_ttnn_cpp_extension'
python3 -m pip install -e .
cd "${GITHUB_WORKSPACE}"

###############################################################################
# Step: Run C++ Extension Tests
###############################################################################
# Test phase: imports 'ttnn' (from PyPI) and our 'torch_ttnn_cpp_extension'.
# Ensure runtime can locate shared libraries (libtt_metal.so, libtt_stl.so, etc.)
export TT_METAL_HOME="${GITHUB_WORKSPACE}/torch_ttnn/cpp_extension/third-party/tt-metal"
export TTNN_LIB_PATHS=$(python3 - <<'PY'
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
STL_LIB_DIR=$(python3 - <<'PY'
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
pytest tests/cpp_extension/test_cpp_extension_functionality.py -v

echo "Running BERT C++ extension tests"
pytest tests/cpp_extension/test_bert_cpp_extension.py -v

# echo "Running models C++ extension tests"
# pytest tests/models/ --native_integration -v

echo "Running tests passed"


