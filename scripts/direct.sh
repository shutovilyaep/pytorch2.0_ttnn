#!/usr/bin/env bash

set -euo pipefail

# Workspace directory
# WORKSPACE_DIR="/home/ilia_shutov/pytorch2.0_ttnn"
# Previous container path for reference:
WORKSPACE_DIR="/workspace/pytorch2.0_ttnn"

# Added pre-build steps with venv recreation for clean build
pushd "${WORKSPACE_DIR}/" >/dev/null

# Activate tt-metal venv if it exists and set Python executable
TT_METAL_VENV="${WORKSPACE_DIR}/torch_ttnn/cpp_extension/third-party/tt-metal/python_env"
if [ -f "${TT_METAL_VENV}/bin/activate" ]; then
    echo "> Activating tt-metal venv: ${TT_METAL_VENV}"
    source "${TT_METAL_VENV}/bin/activate"
    PYTHON="${TT_METAL_VENV}/bin/python3"
    echo "> Python: ${PYTHON}"
else
    echo "> Warning: tt-metal venv not found at ${TT_METAL_VENV}"
    PYTHON="$(which python3)"
    echo "> Using system Python: ${PYTHON}"
fi

# Ensure PYTHON is set
PYTHON="${PYTHON:-python3}"

export TT_METAL_HOME="${WORKSPACE_DIR}/torch_ttnn/cpp_extension/third-party/tt-metal"
export TT_METAL_KERNEL_PATH="${TT_METAL_HOME}"
export CARGO_HOME="${HOME}/.cargo"
export RUSTUP_HOME="${HOME}/.rustup"
export PATH="${HOME}/.cargo/bin:${PATH}"
export PYTHONFAULTHANDLER=1
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"

# Настройка LD_LIBRARY_PATH для сборки (логика из run-cpp-extension-tests-only.sh)
declare -a EXTRA_LIB_DIRS

# 1) tt-metal типовые директории сборки
for d in \
  "${TT_METAL_HOME}/build_Release/lib" \
  "${TT_METAL_HOME}/build/lib" \
  "${TT_METAL_HOME}/build_Release/tt_stl" \
  "${TT_METAL_HOME}/build/tt_stl"
do
  [[ -d "$d" ]] && EXTRA_LIB_DIRS+=("$d")
done

# 2) директории ttnn (site-packages) + ttnn.libs
TTNN_LIB_PATHS=$("${PYTHON}" - <<'PY'
import pathlib, site
try:
    import ttnn
    p = pathlib.Path(ttnn.__file__).parent
    libs_dir = p.parent / 'ttnn.libs'
    cands = []
    for d in [p / 'build' / 'lib', p / '.libs', p, libs_dir]:
        if d.exists():
            cands.append(str(d))
    print(':'.join(cands))
except Exception:
    print('')
PY
)
if [[ -n "${TTNN_LIB_PATHS}" ]]; then
  IFS=':' read -r -a arr <<<"${TTNN_LIB_PATHS}"
  for d in "${arr[@]}"; do [[ -d "$d" ]] && EXTRA_LIB_DIRS+=("$d"); done
fi

# 3) torch/lib директории (критически важно для libc10.so, libtorch.so и т.д.)
TORCH_LIB_DIRS=$("${PYTHON}" - <<'PY'
import pathlib, site
out=[]
try:
    import torch
    torch_root = pathlib.Path(torch.__file__).parent
    lib = torch_root / 'lib'
    if lib.is_dir(): out.append(str(lib))
except Exception:
    pass
for sp in set(site.getsitepackages()+[site.getusersitepackages()]):
    p = pathlib.Path(sp) / 'torch' / 'lib'
    if p.is_dir(): out.append(str(p))
print(':'.join(out))
PY
)
# Добавляем torch/lib в НАЧАЛО списка, так как он критически важен
TORCH_PRIORITY_DIRS=()
if [[ -n "${TORCH_LIB_DIRS}" ]]; then
  IFS=':' read -r -a arr <<<"${TORCH_LIB_DIRS}"
  for d in "${arr[@]}"; do 
    if [[ -d "$d" ]]; then
      TORCH_PRIORITY_DIRS+=("$d")
      # Также добавляем в общий список для полноты
      EXTRA_LIB_DIRS+=("$d")
    fi
  done
fi

# 4) если найдём libtt_stl.so внутри дерева tt-metal — добавим его директорию
STL_FROM_TTM=$(find "${TT_METAL_HOME}" -maxdepth 8 -type f -name 'libtt_stl.so' 2>/dev/null | head -n1 || true)
if [[ -n "${STL_FROM_TTM}" ]]; then
  EXTRA_LIB_DIRS+=("$(dirname "${STL_FROM_TTM}")")
fi

# Уникализируем пути
declare -A SEEN
UNIQ_DIRS=()
for p in "${EXTRA_LIB_DIRS[@]}"; do
  if [[ -n "$p" && -d "$p" && -z "${SEEN[$p]:-}" ]]; then
    SEEN[$p]=1; UNIQ_DIRS+=("$p")
  fi
done

FINAL_LD="$(IFS=:; echo "${UNIQ_DIRS[*]}")"
# Убеждаемся, что torch/lib в начале LD_LIBRARY_PATH (критично для libc10.so)
TORCH_LD=""
if [[ ${#TORCH_PRIORITY_DIRS[@]} -gt 0 ]]; then
  TORCH_LD="$(IFS=:; echo "${TORCH_PRIORITY_DIRS[*]}")"
fi
# Собираем финальный путь: torch/lib ПЕРВЫМ, затем остальные
if [[ -n "${TORCH_LD}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LD}:${FINAL_LD}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${FINAL_LD}:${LD_LIBRARY_PATH:-}"
fi

echo "[paths] TT_METAL_HOME=${TT_METAL_HOME}"
echo "[paths] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Upgrade pip to support editable installs with pyproject.toml
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

# Clean previous extension build artifacts to avoid stale CMake cache
# rm -rf ${WORKSPACE_DIR}/torch_ttnn/cpp_extension/build || true

"${PYTHON}" -m pip install -e . --use-pep517 --no-cache-dir --no-build-isolation

popd >/dev/null

pushd "${WORKSPACE_DIR}/torch_ttnn/cpp_extension/" >/dev/null
# Ensure PEP517 backend & native build tools are available for pyproject builds
"${PYTHON}" -m pip install --upgrade scikit-build-core cmake ninja

# Clean runtime/sfpi directory to avoid FetchContent conflicts
# CMake FetchContent tries to rename ex-sfpi*/sfpi to runtime/sfpi, but fails if directory exists
if [[ -d "${TT_METAL_HOME}/runtime/sfpi" ]]; then
  echo "> Cleaning runtime/sfpi directory to avoid FetchContent conflicts"
  rm -rf "${TT_METAL_HOME}/runtime/sfpi"
fi
# Also clean any temporary ex-sfpi* directories
if [[ -d "${TT_METAL_HOME}/runtime" ]]; then
  find "${TT_METAL_HOME}/runtime" -maxdepth 1 -type d -name "ex-sfpi*" -exec rm -rf {} + 2>/dev/null || true
fi

# Ensure ccache exists (build system invokes it)
if ! command -v ccache >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ccache
fi
# ./build_cpp_extension.sh
./build_cpp_extension.sh RelWithDebInfo

# # Hardcoded host workspace mapping (used to rewrite compile_commands for host IDE)
# HOST_WORKSPACE="/home/kilka/Projects/ML/TT-NN/dev.docker/workspace"

# # Export compile_commands.json to workspace root for editor tooling
# if [ -f /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json ]; then
#   cp /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/build/temp.linux-x86_64-3.10/ttnn_device_extension/compile_commands.json /workspace/pytorch2.0_ttnn/compile_commands.json
#   # Optionally emit host-adjusted compile_commands.json if HOST_WORKSPACE is provided
#   if [ -n "${HOST_WORKSPACE:-}" ]; then
#     python3 /workspace/pytorch2.0_ttnn/scripts/rewrite_compile_commands.py \
#       /workspace/pytorch2.0_ttnn/compile_commands.json \
#       /workspace/pytorch2.0_ttnn/compile_commands.host.json \
#       /workspace \
#       ${HOST_WORKSPACE}
#   fi
# fi

popd >/dev/null

"${PYTHON}" "${WORKSPACE_DIR}/scripts/test_script.py"

# pushd /workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/build/bin >/dev/null
# ./tt-metal-trace
# popd >/dev/null
