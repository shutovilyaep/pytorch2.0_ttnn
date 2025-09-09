#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR=/home/epam/shutov/.cache/cpp-extension-cache

# Added pre-build steps with venv recreation for clean build
pushd torch_ttnn/cpp_extension/third-party/tt-metal >/dev/null
./build_metal.sh
rm -rf python_env
./create_venv.sh
source ./python_env/bin/activate
popd >/dev/null


# This script reproduces the steps from .github/actions/build_cpp_extension_artifacts/action.yaml
# Run from the repository root. Expects environment variable CACHE_DIR to be set
# (in CI it's set to /root/.cache/cpp-extension-cache).

# echo "[1/6] Docker cleanup"
# docker system prune -a -f
# df -h

# echo "[2/6] Install system dependencies"
# apt upgrade -y && apt update -y
# apt install -y cmake python3 python3-venv python3-pip git-lfs ccache gcc-12 g++-12

# git config --global --add safe.directory /home/ubuntu/actions-runner/_work/pytorch2.0_ttnn/pytorch2.0_ttnn || true
# git config --global --add safe.directory /__w/pytorch2.0_ttnn/pytorch2.0_ttnn || true

# echo "[3/6] Prepare tt-metal dependencies (skip hugepages)"
# sed -i '/^configure_hugepages() {/,/^}/c\configure_hugepages() {\n    echo "Skip hugepages installation"\n}' ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh
# ./torch_ttnn/cpp_extension/third-party/tt-metal/install_dependencies.sh

echo "[4/6] Install Python dependencies"
python3 -m ensurepip --upgrade || true
python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-dev.txt
python3 -m pip install pytest-github-report
# Match CI behavior: keep pip at 21.2.4 and avoid upgrading numpy to 2.x
# python3 -m pip install --force-reinstall 'pip==21.2.4'
# python3 -m pip install 'numpy>=1.24.4,<2' setuptools wheel
python3 -m pip install -e .

# echo "[5/6] Sync submodules and build C++ extensions"
# git submodule sync
# git submodule update --init --recursive
# git submodule foreach 'git lfs fetch --all && git lfs pull'

pushd torch_ttnn/cpp_extension >/dev/null
./build_cpp_extension.sh
popd >/dev/null

echo "[6/6] Collect build artifacts into CACHE_DIR=$CACHE_DIR"
pushd torch_ttnn/cpp_extension >/dev/null
PYTHON_LIB_SUFFIX=$(python3 -c "import importlib.machinery; print(importlib.machinery.EXTENSION_SUFFIXES[0])")
mkdir -p "$CACHE_DIR"
CCACHE_DIR=$(ccache --get-config cache_dir)

# cpp-extension cache
cp -r build "$CACHE_DIR"
cp -r "ttnn_device_extension${PYTHON_LIB_SUFFIX}" "$CACHE_DIR"

# tt-metal cache
mkdir -p "$CACHE_DIR/tt-metal"
cp -r third-party/tt-metal/build "$CACHE_DIR/tt-metal/"
cp -r third-party/tt-metal/.cpmcache "$CACHE_DIR/tt-metal/"

# ccache
cp -r "$CCACHE_DIR" "$CACHE_DIR"
popd >/dev/null

echo "Done."


