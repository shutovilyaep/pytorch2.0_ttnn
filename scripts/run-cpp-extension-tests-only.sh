#!/usr/bin/env bash

set -euo pipefail

# Быстрый запуск только тестов C++ расширения с диагностикой LD_LIBRARY_PATH
# Ничего не билдит и не устанавливает, лишь подбирает пути к .so и запускает pytest

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
else
    echo "> Warning: tt-metal venv not found at ${TT_METAL_VENV}"
    PYTHON="$(which python3)"
    echo "> Using system Python: ${PYTHON}"
fi

# Ensure PYTHON is set
PYTHON="${PYTHON:-python3}"

usage() {
  echo "Usage: $0 [--ld-debug] [--only NAME]"
  echo "  --ld-debug   Enable LD_DEBUG=libs for import tracing"
  echo "  --only NAME  Run only one test file: functionality | bert"
}

LD_DEBUG_LIBS=0
ONLY_WHAT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ld-debug) LD_DEBUG_LIBS=1; shift;;
    --only) ONLY_WHAT="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

export TT_METAL_HOME="${REPO_ROOT}/torch_ttnn/cpp_extension/third-party/tt-metal"

# Соберём кандидаты путей для LD_LIBRARY_PATH
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

# 5) ЖЁСТКО прописанные пути к библиотекам (хардкод для быстрого запуска)
# ВАЖНО: torch/lib должен быть ПЕРВЫМ для правильной загрузки libc10.so
HARDCODED_PATHS=(
  "/opt/venv/lib/python3.10/site-packages/torch/lib"
  "${TT_METAL_HOME}/build_Release/lib"
  "${TT_METAL_HOME}/build_Release/ttnn"
  "${TT_METAL_HOME}/build/lib"
  "${TT_METAL_HOME}/build/ttnn"
  "/opt/venv/lib/python3.10/site-packages/ttnn/build/lib"
  "/opt/venv/lib/python3.10/site-packages/ttnn"
  "/opt/venv/lib/python3.10/site-packages/ttnn.libs"
  "/workspace/pytorch2.0_ttnn/torch_ttnn/cpp_extension"
)
for hp in "${HARDCODED_PATHS[@]}"; do
  [[ -d "$hp" ]] && EXTRA_LIB_DIRS+=("$hp")
done

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

# Проверка критических библиотек PyTorch и добавление их путей
# Используем отдельные файловые дескрипторы для stdout и stderr
TORCH_LIB_FOUND=$("${PYTHON}" 2> >(tee /tmp/torch_lib_diag.$$ >&2) - <<'PY'
import site, pathlib
import sys

torch_lib_dirs = []
for sp in set(site.getsitepackages()+[site.getusersitepackages()]):
    torch_lib = pathlib.Path(sp) / "torch" / "lib"
    if torch_lib.exists() and torch_lib.is_dir():
        torch_lib_dirs.append(str(torch_lib))
        # Проверяем наличие критических библиотек
        for lib in ["libc10.so", "libtorch.so", "libtorch_python.so"]:
            lib_file = torch_lib / lib
            if lib_file.exists():
                print(f"[ok] {lib} найден в {torch_lib}", file=sys.stderr)
            else:
                print(f"[warn] {lib} не найден в {torch_lib}", file=sys.stderr)

# Выводим только пути в stdout
if torch_lib_dirs:
    print(":".join(torch_lib_dirs))
else:
    print("")
PY
)

# Выводим диагностику из stderr
if [[ -f /tmp/torch_lib_diag.$$ ]]; then
  cat /tmp/torch_lib_diag.$$
  rm -f /tmp/torch_lib_diag.$$
fi

# Добавляем найденные пути к библиотекам PyTorch в начало LD_LIBRARY_PATH
if [[ -n "${TORCH_LIB_FOUND}" ]]; then
  # Убеждаемся, что torch/lib в начале LD_LIBRARY_PATH
  if [[ "${LD_LIBRARY_PATH}" != *"${TORCH_LIB_FOUND}"* ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_FOUND}:${LD_LIBRARY_PATH:-}"
    echo "[paths] Added PyTorch lib paths to LD_LIBRARY_PATH: ${TORCH_LIB_FOUND}"
  fi
fi

# Если SONAME libtt_stl.so не доступен напрямую, создадим локальный symlink на найденный libtt_stl.so*
ART_LIB_DIR="${REPO_ROOT}/.ttnn_runtime_artifacts/lib"
mkdir -p "${ART_LIB_DIR}"

ensure_soname() {
  local lib_basename="$1"   # например libtt_stl.so
  shift
  local found=""
  # Ищем в переданных директориях первым делом точное имя
  for base in "$@"; do
    if [[ -f "${base}/${lib_basename}" ]]; then
      found="${base}/${lib_basename}"; break
    fi
  done
  # Если точного нет — ищем lib_basename.*
  if [[ -z "${found}" ]]; then
    for base in "$@"; do
      cand=$(ls -1 "${base}/${lib_basename}"* 2>/dev/null | head -n1 || true)
      if [[ -n "${cand}" && -f "${cand}" ]]; then
        found="${cand}"; break
      fi
    done
  fi
  # Fallback: ищем по префиксу libtt_stl*.so (чтобы поймать хешированные файлы auditwheel)
  if [[ -z "${found}" ]]; then
    local prefix="${lib_basename%.so}" # libtt_stl
    for base in "$@"; do
      cand=$(ls -1 "${base}/${prefix}"*.so 2>/dev/null | head -n1 || true)
      if [[ -n "${cand}" && -f "${cand}" ]]; then
        found="${cand}"; break
      fi
    done
  fi
  if [[ -n "${found}" ]]; then
    ln -sf "${found}" "${ART_LIB_DIR}/${lib_basename}" || true
    export LD_LIBRARY_PATH="${ART_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "[soname] ${lib_basename} -> ${found} (via ${ART_LIB_DIR})"
  fi
}

# Кандидаты для поиска libtt_stl.so и libtt_metal.so
SEARCH_DIRS=(
  "${TT_METAL_HOME}/build_Release/lib"
  "${TT_METAL_HOME}/build/lib"
  "${TT_METAL_HOME}/build_Release/tt_stl"
  "${TT_METAL_HOME}/build/tt_stl"
)

if [[ -n "${TTNN_LIB_PATHS}" ]]; then
  IFS=':' read -r -a arr <<<"${TTNN_LIB_PATHS}"
  for d in "${arr[@]}"; do [[ -d "$d" ]] && SEARCH_DIRS+=("$d"); done
fi

if [[ -n "${TORCH_LIB_DIRS}" ]]; then
  IFS=':' read -r -a arr <<<"${TORCH_LIB_DIRS}"
  for d in "${arr[@]}"; do [[ -d "$d" ]] && SEARCH_DIRS+=("$d"); done
fi

ensure_soname "libtt_stl.so" "${SEARCH_DIRS[@]}"
ensure_soname "libtt_metal.so" "${SEARCH_DIRS[@]}"

# Если libtt_stl.so всё ещё не найден, попробуем использовать libttnn_core.so как замену
# (так как tt_stl обычно header-only, но некоторые сборки требуют .so, и он может быть в составе core)
if [[ ! -f "${ART_LIB_DIR}/libtt_stl.so" ]]; then
  # Ищем libttnn_core.so как возможную замену
  for alt_lib in "${TT_METAL_HOME}/build_Release/lib/libttnn_core.so" \
                 "${TT_METAL_HOME}/build_Release/ttnn/libttnn_core.so" \
                 "${TT_METAL_HOME}/build/lib/libttnn_core.so" \
                 "/opt/venv/lib/python3.10/site-packages/ttnn/build/lib/libttnn_core.so"; do
    if [[ -f "${alt_lib}" ]]; then
      echo "[warn] libtt_stl.so не найден, использую libttnn_core.so как замену: ${alt_lib}"
      # Создаём symlink на libttnn_core.so с именем libtt_stl.so
      ln -sf "${alt_lib}" "${ART_LIB_DIR}/libtt_stl.so" 2>/dev/null || \
      cp "${alt_lib}" "${ART_LIB_DIR}/libtt_stl.so" 2>/dev/null || true
      export LD_LIBRARY_PATH="${ART_LIB_DIR}:$(dirname "${alt_lib}"):${LD_LIBRARY_PATH}"
      break
    fi
  done
  # Если и libttnn_core.so не найден, попробуем libtt_metal.so
  if [[ ! -f "${ART_LIB_DIR}/libtt_stl.so" ]]; then
    for alt_lib in "${TT_METAL_HOME}/build_Release/lib/libtt_metal.so" \
                   "/opt/venv/lib/python3.10/site-packages/ttnn/build/lib/_ttnncpp.so"; do
      if [[ -f "${alt_lib}" ]]; then
        echo "[warn] libtt_stl.so не найден, создаю заглушку из $(basename ${alt_lib})"
        ln -sf "${alt_lib}" "${ART_LIB_DIR}/libtt_stl.so" 2>/dev/null || \
        cp "${alt_lib}" "${ART_LIB_DIR}/libtt_stl.so" 2>/dev/null || true
        export LD_LIBRARY_PATH="${ART_LIB_DIR}:${LD_LIBRARY_PATH}"
        break
      fi
    done
  fi
fi

# Диагностика: где ttnn и пробный поиск расширения
echo "[debug] Searching for ttnn_device_extension module..."
EXT_MODULE_INFO=$("${PYTHON}" - <<'PY'
import site, sys, pathlib
import os

# Поиск ttnn_device_extension (с расширением .so или без)
roots = set(site.getsitepackages() + [site.getusersitepackages()])
found_files = []
found_dirs = []

for r in roots:
    p = pathlib.Path(r)
    # Ищем в torch_ttnn_cpp_extension директории
    ext_dir = p / "torch_ttnn_cpp_extension"
    if ext_dir.exists():
        # Ищем файл с расширением .so
        for so_file in ext_dir.glob("ttnn_device_extension*.so"):
            found_files.append(str(so_file))
            found_dirs.append(str(so_file.parent))
            print(f"[found] {so_file}")
            break
        # Также ищем файл без расширения (может быть установлен без .so)
        ext_file = ext_dir / "ttnn_device_extension"
        if ext_file.exists() and ext_file.is_file():
            found_files.append(str(ext_file))
            found_dirs.append(str(ext_file.parent))
            print(f"[found] {ext_file} (no extension)")
            break
    # Также ищем рекурсивно
    for so_file in p.rglob("ttnn_device_extension*.so"):
        found_files.append(str(so_file))
        found_dirs.append(str(so_file.parent))
        print(f"[found] {so_file}")
        break

# Выводим информацию: путь к файлу и директория (разделенные |)
if found_files:
    print(f"{found_files[0]}|{found_dirs[0]}")
else:
    print("|")
PY
)

# Парсим результат
EXT_MODULE_FILE="${EXT_MODULE_INFO%%|*}"
EXT_MODULE_PATH="${EXT_MODULE_INFO##*|}"

# Добавляем найденный путь в PYTHONPATH для импорта модуля
if [[ -n "${EXT_MODULE_PATH}" && -d "${EXT_MODULE_PATH}" ]]; then
  echo "[debug] Found extension module directory: ${EXT_MODULE_PATH}"
  if [[ -n "${EXT_MODULE_FILE}" && -f "${EXT_MODULE_FILE}" ]]; then
    echo "[debug] Extension file: ${EXT_MODULE_FILE}"
  fi
  export PYTHONPATH="${EXT_MODULE_PATH}:${PYTHONPATH:-}"
  echo "[debug] Added to PYTHONPATH: ${EXT_MODULE_PATH}"
else
  echo "[warn] ttnn_device_extension not found in site-packages"
  echo "[warn] Will try to import from build directory or via ttnn_device_mode.py"
fi

# Проверяем наличие модуля (диагностика)
echo "[debug] Checking extension module location..."
"${PYTHON}" - <<'PY' || true
import sys
import site
from pathlib import Path

print('[python] sys.executable =', sys.executable)
print('[python] PYTHONPATH =', sys.path[:5])

# Показываем все возможные пути
print('[debug] Searching in site-packages:')
for sp in site.getsitepackages():
    ext_dir = Path(sp) / "torch_ttnn_cpp_extension"
    if ext_dir.exists():
        print(f'  {ext_dir} exists')
        for f in ext_dir.iterdir():
            print(f'    {f.name}')
    else:
        print(f'  {ext_dir} does not exist')
PY

# Если нашли ttnn_device_extension — покажем ldd
EXT_SO=$("${PYTHON}" - <<'PY'
import site, pathlib
for r in set(site.getsitepackages()+[site.getusersitepackages()]):
    ext_dir = pathlib.Path(r) / "torch_ttnn_cpp_extension"
    if ext_dir.exists():
        # Сначала ищем с расширением .so
        for so_file in ext_dir.glob("ttnn_device_extension*.so"):
            print(str(so_file))
            raise SystemExit(0)
        # Если не нашли, ищем без расширения
        ext_file = ext_dir / "ttnn_device_extension"
        if ext_file.exists() and ext_file.is_file():
            print(str(ext_file))
            raise SystemExit(0)
print('')
PY
)
if [[ -n "${EXT_SO}" && -f "${EXT_SO}" ]]; then
  echo "[ldd] ${EXT_SO}"
  ldd "${EXT_SO}" || true
fi

if [[ ${LD_DEBUG_LIBS} -eq 1 ]]; then
  export LD_DEBUG=libs
fi

# LD_PRELOAD для принудительной загрузки библиотек перед расширением
# Это помогает решить проблемы с undefined symbols (особенно weak symbols)
PRELOAD_LIBS=()

# 1. tt-metal _ttnncpp.so (для разрешения weak symbols типа BinaryOperation::invoke)
for lib_path in "${TT_METAL_HOME}/build/lib/_ttnncpp.so" \
                "${TT_METAL_HOME}/build_Release/lib/_ttnncpp.so"; do
  if [[ -f "${lib_path}" ]]; then
    PRELOAD_LIBS+=("${lib_path}")
    echo "[preload] Adding tt-metal library: ${lib_path}"
    break
  fi
done

# 2. PyTorch libraries (для совместимости версий)
for lib in "libc10.so" "libtorch.so" "libtorch_cpu.so" "libtorch_python.so"; do
  lib_path="/opt/venv/lib/python3.10/site-packages/torch/lib/${lib}"
  if [[ -f "${lib_path}" ]]; then
    PRELOAD_LIBS+=("${lib_path}")
  fi
done

# Также проверяем tt-metal venv
if [[ -n "${TT_METAL_VENV}" && -d "${TT_METAL_VENV}" ]]; then
  for lib in "libc10.so" "libtorch.so" "libtorch_cpu.so" "libtorch_python.so"; do
    lib_path="${TT_METAL_VENV}/lib/python3.10/site-packages/torch/lib/${lib}"
    if [[ -f "${lib_path}" ]]; then
      PRELOAD_LIBS+=("${lib_path}")
    fi
  done
fi

if [[ ${#PRELOAD_LIBS[@]} -gt 0 ]]; then
  export LD_PRELOAD="$(IFS=:; echo "${PRELOAD_LIBS[*]}")"
  echo "[preload] LD_PRELOAD=${LD_PRELOAD}"
fi

# Проверка версии PyTorch для диагностики
PYTORCH_VERSION=$("${PYTHON}" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "[info] PyTorch version: ${PYTORCH_VERSION}"
echo "[warn] Если видите 'undefined symbol', возможно расширение собрано с другой версией PyTorch"
echo "[warn] Попробуйте пересобрать: cd torch_ttnn/cpp_extension && pip install -e . --force-reinstall --no-cache-dir"

# Добавляем путь к исходникам для импорта torch_ttnn.cpp_extension
export PYTHONPATH="${REPO_ROOT}/torch_ttnn/cpp_extension:${REPO_ROOT}:${PYTHONPATH:-}"

# Финальная проверка импорта перед запуском тестов
echo "[debug] Final import check before running tests..."
echo "[debug] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[debug] LD_PRELOAD=${LD_PRELOAD}"
echo "[debug] PYTHONPATH=${PYTHONPATH}"

# Выполняем импорт с явным выводом ошибок
# Используем timeout если доступен, иначе просто python3
# Временно отключаем LD_PRELOAD для системных команд (mktemp, grep, head и т.д.)
SAVED_LD_PRELOAD="${LD_PRELOAD:-}"

# Создаем временный файл для вывода (без LD_PRELOAD)
unset LD_PRELOAD
IMPORT_TMPFILE=$(mktemp 2>/dev/null || echo "/tmp/ttnn_import_$$.log")
trap "rm -f ${IMPORT_TMPFILE}" EXIT
export LD_PRELOAD="${SAVED_LD_PRELOAD}"

# Важно: LD_PRELOAD должен быть установлен для Python процесса импорта
# но не для системных команд (timeout, mktemp и т.д.)
if command -v timeout >/dev/null 2>&1; then
  # timeout запускается без LD_PRELOAD, но Python внутри получит LD_PRELOAD из окружения
  # Однако нужно явно передать LD_PRELOAD в Python процесс
  unset LD_PRELOAD
  timeout 30 env LD_PRELOAD="${SAVED_LD_PRELOAD}" "${PYTHON}" -c "
import sys
import warnings
warnings.filterwarnings('ignore')

# Фильтруем предупреждения о .METAL_VERSION
class FilterStderr:
    def __init__(self):
        self.original = sys.stderr
    def write(self, s):
        if 'METAL_VERSION' not in s and 'library_tweaks' not in s:
            self.original.write(s)
    def flush(self):
        self.original.flush()

old_stderr = sys.stderr
sys.stderr = FilterStderr()

try:
    # Пытаемся импортировать напрямую через ttnn_device_mode
    sys.path.insert(0, '${REPO_ROOT}/torch_ttnn/cpp_extension')
    from torch_ttnn.cpp_extension.ttnn_device_mode import ttnn_module
    sys.stderr = old_stderr
    print('SUCCESS')
except ImportError:
    # Если прямой импорт не работает, пробуем через __init__
    try:
        from torch_ttnn.cpp_extension import ttnn_module
        sys.stderr = old_stderr
        print('SUCCESS')
    except Exception as e2:
        sys.stderr = old_stderr
        print(f'ERROR: {type(e2).__name__}: {e2}', file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
except Exception as e:
    sys.stderr = old_stderr
    print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" > "${IMPORT_TMPFILE}" 2>&1
  IMPORT_EXIT_CODE=$?
  # Восстанавливаем LD_PRELOAD
  export LD_PRELOAD="${SAVED_LD_PRELOAD}"
  
  # Фильтруем предупреждения о .METAL_VERSION
  unset LD_PRELOAD
  if [[ -f "${IMPORT_TMPFILE}" ]]; then
    IMPORT_OUTPUT=$(cat "${IMPORT_TMPFILE}" | grep -v "METAL_VERSION" | grep -v "library_tweaks" || cat "${IMPORT_TMPFILE}")
  else
    IMPORT_OUTPUT=""
  fi
  export LD_PRELOAD="${SAVED_LD_PRELOAD}"
else
  # Если timeout недоступен, используем Python напрямую
  unset LD_PRELOAD
  env LD_PRELOAD="${SAVED_LD_PRELOAD}" "${PYTHON}" -c "
import sys
import warnings
warnings.filterwarnings('ignore')

class FilterStderr:
    def __init__(self):
        self.original = sys.stderr
    def write(self, s):
        if 'METAL_VERSION' not in s and 'library_tweaks' not in s:
            self.original.write(s)
    def flush(self):
        self.original.flush()

old_stderr = sys.stderr
sys.stderr = FilterStderr()

try:
    sys.path.insert(0, '${REPO_ROOT}/torch_ttnn/cpp_extension')
    from torch_ttnn.cpp_extension.ttnn_device_mode import ttnn_module
    sys.stderr = old_stderr
    print('SUCCESS')
except ImportError:
    try:
        from torch_ttnn.cpp_extension import ttnn_module
        sys.stderr = old_stderr
        print('SUCCESS')
    except Exception as e2:
        sys.stderr = old_stderr
        print(f'ERROR: {type(e2).__name__}: {e2}', file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
except Exception as e:
    sys.stderr = old_stderr
    print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" > "${IMPORT_TMPFILE}" 2>&1
  IMPORT_EXIT_CODE=$?
  export LD_PRELOAD="${SAVED_LD_PRELOAD}"
  
  # Фильтруем предупреждения
  unset LD_PRELOAD
  if [[ -f "${IMPORT_TMPFILE}" ]]; then
    IMPORT_OUTPUT=$(cat "${IMPORT_TMPFILE}" | grep -v "METAL_VERSION" | grep -v "library_tweaks" || cat "${IMPORT_TMPFILE}")
  else
    IMPORT_OUTPUT=""
  fi
  export LD_PRELOAD="${SAVED_LD_PRELOAD}"
fi

echo "[debug] Import command completed with exit code: ${IMPORT_EXIT_CODE}"
echo "[debug] Import output:"
if [[ -n "${IMPORT_OUTPUT:-}" ]]; then
  # Используем Python для обрезки вместо head (чтобы избежать проблем с LD_PRELOAD)
  echo "${IMPORT_OUTPUT}" | head -c 2000 || echo "${IMPORT_OUTPUT:0:2000}"
else
  echo "[warn] Import output is empty"
  if [[ -f "${IMPORT_TMPFILE:-}" ]]; then
    echo "[debug] Reading from temp file:"
    cat "${IMPORT_TMPFILE}" | head -c 2000 || head -c 2000 "${IMPORT_TMPFILE}" 2>/dev/null || echo "Could not read temp file"
  fi
fi
echo ""

if [[ ${IMPORT_EXIT_CODE} -ne 0 ]]; then
  echo "[error] Failed to import ttnn_module (exit code: ${IMPORT_EXIT_CODE})"
  echo "[error] Full output:"
  echo "${IMPORT_OUTPUT}"
  
  # Проверяем, является ли это ошибкой undefined symbol (требуется пересборка)
  if echo "${IMPORT_OUTPUT}" | grep -q "undefined symbol"; then
    echo ""
    echo "[error] =========================================="
    echo "[error] UNDEFINED SYMBOL ERROR DETECTED"
    echo "[error] =========================================="
    
    # Извлекаем имя символа из ошибки
    UNDEF_SYMBOL=$(echo "${IMPORT_OUTPUT}" | grep -o "undefined symbol: [^ ]*" | cut -d' ' -f3 || echo "unknown")
    echo "[error] Missing symbol: ${UNDEF_SYMBOL}"
    
    # Деманглируем символ для читаемости
    if command -v c++filt &> /dev/null && [[ -n "${UNDEF_SYMBOL}" && "${UNDEF_SYMBOL}" != "unknown" ]]; then
      DEMANGLED=$(c++filt "${UNDEF_SYMBOL}" 2>/dev/null || echo "${UNDEF_SYMBOL}")
      echo "[error] Demangled: ${DEMANGLED}"
    fi
    
    # Проверяем наличие символа в библиотеках
    echo "[debug] Checking for symbol in tt-metal libraries..."
    if [[ -n "${UNDEF_SYMBOL}" && "${UNDEF_SYMBOL}" != "unknown" ]]; then
      for lib in "${TT_METAL_HOME}/build/lib/_ttnncpp.so" \
                 "${TT_METAL_HOME}/build_Release/lib/_ttnncpp.so" \
                 "/opt/venv/lib/python3.10/site-packages/ttnn/build/lib/_ttnncpp.so"; do
        if [[ -f "${lib}" ]]; then
          SYMBOL_FOUND=$(nm -D "${lib}" 2>/dev/null | grep -c "${UNDEF_SYMBOL}" || echo "0")
          if [[ "${SYMBOL_FOUND}" != "0" ]]; then
            echo "[info] Symbol found in ${lib} (${SYMBOL_FOUND} occurrences)"
            nm -D "${lib}" 2>/dev/null | grep "${UNDEF_SYMBOL}" | head -3
          fi
        fi
      done
    fi
    
    echo "[error]"
    echo "[error] POSSIBLE CAUSES:"
    echo "[error] 1. Extension was built against incompatible tt-metal version"
    echo "[error] 2. Missing library dependencies in LD_LIBRARY_PATH"
    echo "[error] 3. Symbol visibility issues (weak symbols not resolved)"
    echo "[error]"
    echo "[error] SOLUTION: Rebuild the extension:"
    echo "[error]   cd torch_ttnn/cpp_extension"
    echo "[error]   pip install -e . --force-reinstall --no-cache-dir"
    echo "[error] =========================================="
    echo ""
  fi
  
  echo "[error] Checking installation..."
  "${PYTHON}" - <<'PY'
import sys
import site
from pathlib import Path

print("=== Installation Check ===")
print(f"Python: {sys.executable}")
print(f"PYTHONPATH: {sys.path[:5]}")

# Проверяем установку пакета
try:
    import torch_ttnn_cpp_extension
    print(f"torch_ttnn_cpp_extension found at: {Path(torch_ttnn_cpp_extension.__file__).parent}")
except ImportError as e:
    print(f"torch_ttnn_cpp_extension not found: {e}")

# Проверяем site-packages
print("\n=== Site-packages Check ===")
for sp in site.getsitepackages():
    ext_dir = Path(sp) / "torch_ttnn_cpp_extension"
    print(f"{ext_dir}: {'exists' if ext_dir.exists() else 'missing'}")
    if ext_dir.exists():
        for f in ext_dir.iterdir():
            print(f"  {f.name} ({'file' if f.is_file() else 'dir'})")
PY
  exit 1
fi

# Проверяем, что импорт действительно успешен
if echo "${IMPORT_OUTPUT}" | grep -q "SUCCESS"; then
  echo "[success] Module imported successfully"
else
  echo "[warn] Import completed but no SUCCESS message found"
  echo "[debug] Import output: ${IMPORT_OUTPUT}"
fi

echo "[run] pytest selection: ${ONLY_WHAT:-all}"
set -x
case "${ONLY_WHAT}" in
  functionality) "${PYTHON}" -m pytest tests/cpp_extension/test_cpp_extension_functionality.py -v;;
  bert)          "${PYTHON}" -m pytest tests/cpp_extension/test_bert_cpp_extension.py -v;;
  "")           "${PYTHON}" -m pytest tests/cpp_extension/test_cpp_extension_functionality.py -v && \
                 "${PYTHON}" -m pytest tests/cpp_extension/test_bert_cpp_extension.py -v;;
  *)             echo "Unknown value for --only: ${ONLY_WHAT}"; exit 2;;
esac

