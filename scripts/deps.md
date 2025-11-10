# PyTorch TTNN Dependency & Artifact Reference

## Overview

- The repository exposes the editable `torch-ttnn` Python package together with the native `torch_ttnn_cpp_extension`.
- All native builds are driven by the tt-metal submodule at `torch_ttnn/cpp_extension/third-party/tt-metal`, aligned with **TT_METAL_REF = v0.63.0** in CI (`.github/workflows/run-cpp-native-tests.yaml`).
- A successful build produces reusable tt-metal libraries, installs the `ttnn` Python package inside the tt-metal virtual environment, and drops `ttnn_device_extension.so` (plus its dependent shared libraries) into the active Python environment.

## Component inventory

| Component | Source | Version / ref | Purpose |
| --- | --- | --- | --- |
| tt-metal | `torch_ttnn/cpp_extension/third-party/tt-metal` | `v0.63.0` (default CI tag) | Provides the Metal runtime, TTNN C++ sources, toolchains and helper scripts such as `create_venv.sh`. |
| ttnn (Python) | Installed from the tt-metal checkout via `create_venv.sh` | matches tt-metal tag | Supplies Python APIs and ships `_ttnn.so`, `_ttnncpp.so`, and MPI-enabled runtime libraries. |
| torch-ttnn | Root `pyproject.toml` | depends on `torch==2.7.1+cpu` (`x86_64`), `torchvision==0.22.1+cpu`, `ttnn==0.63.0`, etc. | Front-end compiler and Python integration layer. |
| torch_ttnn_cpp_extension | `torch_ttnn/cpp_extension` | built with `scikit-build-core` | Loads Tenstorrent devices inside PyTorch; links to tt-metal and the PyTorch C++ ABI. |

## Build workflow

```mermaid
flowchart TD
    subgraph Phase1 ["Phase 1 – tt-metal sources"]
        A[tt-metal submodule] --> B[./build_metal.sh --release --enable-ccache]
        B --> C[(build/lib*, build_Release/lib*, headers)]
    end
    subgraph Phase2 ["Phase 2 – Python environment"]
        C --> D[./create_venv.sh]
        D --> E[(python_env site-packages/ttnn)]
    end
    subgraph Phase3 ["Phase 3 – torch-ttnn"]
        E --> F[pip install -e .[dev]]
        F --> G[(ttnn_device_extension.so + copied libs)]
    end
    G --> H[pytest suites]
```

## Phase 1 – Build tt-metal

1. Set `TT_METAL_HOME=/path/to/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal`.
2. From `${TT_METAL_HOME}` execute:

   ```bash
   ./install_dependencies.sh
   ./build_metal.sh --release --enable-ccache
   ```

Key facts:

- `torch_ttnn/cpp_extension/third-party/CMakeLists.txt` sets `BUILD_SHARED_LIBS ON`, so the default build emits shared libraries alongside static archives.
- `./build_metal.sh` symlinks `${TT_METAL_HOME}/build` to the active configuration (for `--release`, that resolves to `${TT_METAL_HOME}/build_Release`).
- Headers are available in `${TT_METAL_HOME}/build*/include`; toolchains remain in `cmake/`.

### Phase 1 artifacts

| Artifact | Location | Notes |
| --- | --- | --- |
| `libtt_metal.so`, `libtt_metal.a` | `${TT_METAL_HOME}/build/lib` (→ `build_Release/lib`) | Shared libs are consumed by the extension; static archives remain for custom scenarios. |
| `libtt_stl.so`, `libtt_stl.a` | `${TT_METAL_HOME}/build/tt_stl` (→ `build_Release/tt_stl`) | STL runtime used by tt-metal and TTNN. |
| `_ttnn.so`, `_ttnncpp.so` | `${TT_METAL_HOME}/build/lib` | Python bindings required by the `ttnn` package. |
| Headers | `${TT_METAL_HOME}/build/include` (→ `build_Release/include`) | Exposed to the extension via `target_include_directories`. |

## Phase 2 – Provision the Python environment

Run `${TT_METAL_HOME}/create_venv.sh`. The script:

- Creates `${TT_METAL_HOME}/python_env` (override with `PYTHON_ENV_DIR`).
- Pins pip/setuptools on Ubuntu 22.04 and configures `https://download.pytorch.org/whl/cpu` as an extra wheel index.
- Installs tt-metal development requirements and runs `pip install -e .`, registering the `ttnn` package built from source.
- Activates the virtual environment. Subsequent steps must reuse that shell or `source python_env/bin/activate`.

The installed `ttnn` wheel exposes `_ttnn.so`, `_ttnncpp.so`, and the tt-metal shared libraries under `${PYTHON_ENV_DIR}/lib/python*/site-packages/ttnn/`.

## Phase 3 – Build torch-ttnn and the C++ extension

With the tt-metal virtualenv active and `TT_METAL_HOME` exported:

```bash
cd /path/to/pytorch2.0_ttnn
python -m pip install --upgrade pip
python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .[dev]
```

Important details:

- `scikit-build-core` drives the CMake project in `torch_ttnn/cpp_extension`, inheriting toolchain settings from `${TT_METAL_HOME}/cmake` and Torch via `torch.utils.cmake_prefix_path`.
- The extension links against `TT::Metalium`, `TTNN::CPP`, `Torch::Torch`, and `Torch::Python`.
- Install rules copy any discovered `libtt_metal.so` and `libtt_stl.so` into `${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension`, ensuring they sit next to `ttnn_device_extension.so`.
- Editable installs keep the generated build tree under `torch_ttnn/cpp_extension/build/`.

### Extension artifacts

| Location | Produced by | Contents |
| --- | --- | --- |
| `torch_ttnn/cpp_extension/build/lib.*/torch_ttnn_cpp_extension/` | CMake build tree | Transient extension binaries during editable builds. |
| `${VIRTUAL_ENV}/lib/python*/site-packages/torch_ttnn_cpp_extension/` | `pip install -e .[dev]` | `ttnn_device_extension.so`, copied tt-metal shared libs, package metadata. |
| `${VIRTUAL_ENV}/lib/python*/site-packages/torch_ttnn/` | `pip install -e .[dev]` | Python sources and runtime helpers. |
| `${VIRTUAL_ENV}/lib/python*/site-packages/ttnn/` | `create_venv.sh` | TTNN Python package with `_ttnn*.so` and supporting libraries. |

## Runtime linking and environment setup

- `ttnn_device_extension.so` is built with `BUILD_RPATH="$ORIGIN"`, so bundled `libtt_metal.so`/`libtt_stl.so` are found automatically once installed.
- For developer workflows we still extend `LD_LIBRARY_PATH` with the tt-metal build outputs so scripts that run directly from the build tree can locate their dependencies. The helper below prefers the canonical `build` symlink and falls back to `build_Release` if the symlink was not created (for example after manual edits):

  ```bash
  add_tt_path() {
      local dir="$1"
      [ -d "$dir" ] || return
      case ":${LD_LIBRARY_PATH:-}:" in
          *":$dir:"*) ;;
          *) LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}";;
      esac
  }

  TT_METAL_LIB_DIR="${TT_METAL_HOME}/build/lib"
  [ -d "${TT_METAL_LIB_DIR}" ] || TT_METAL_LIB_DIR="${TT_METAL_HOME}/build_Release/lib"
  TT_METAL_STL_DIR="${TT_METAL_HOME}/build/tt_stl"
  [ -d "${TT_METAL_STL_DIR}" ] || TT_METAL_STL_DIR="${TT_METAL_HOME}/build_Release/tt_stl"

  add_tt_path "${TT_METAL_LIB_DIR}"
  add_tt_path "${TT_METAL_STL_DIR}"
  export LD_LIBRARY_PATH
  ```

- TTNN’s `_ttnncpp.so` links against MPI; append your MPI installation (CI uses `/opt/openmpi-v5.0.7-ulfm/lib`) to `LD_LIBRARY_PATH` when running distributed workloads.
- Prefer importing through `from torch_ttnn.cpp_extension.ttnn_device_mode import ttnn_module`, which relies on `ExtensionFileLoader` to find the compiled module.

## Configuration knobs

| Setting | How to change | Effect |
| --- | --- | --- |
| `TT_METAL_HOME` | Export before invoking CMake/pip | Points CMake to the tt-metal checkout. |
| `TT_METAL_VERSION` | Export before building | Overrides the version string embedded into the build. |
| `PYTHON_CMD`, `PYTHON_ENV_DIR` | Environment variables for `create_venv.sh` | Select the interpreter or destination virtualenv. |
| `CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF"` | Prefix the `python -m pip install ...` call | Forces a static tt-metal build (requires a clean build directory). |
| `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"` | Prefix the install command | Generates a Debug build of the extension. |

## Common issues and resolutions

1. **Missing `libtt_metal.so` at runtime**

   - *Cause*: libraries were not copied because `TT_METAL_HOME` referenced the wrong checkout or the build tree was outdated.
   - *Fix*: rebuild tt-metal, rerun `pip install -e .[dev]`, and confirm the files exist in `${TT_METAL_HOME}/build*/lib` and `${VIRTUAL_ENV}/lib/python*/site-packages/torch_ttnn_cpp_extension/`.

2. **`MPIX_Comm_revoke` unresolved**

   - *Cause*: MPI libraries are absent from `LD_LIBRARY_PATH` when importing `ttnn`.
   - *Fix*: export the MPI library directory, for example `export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:${LD_LIBRARY_PATH}"`.

3. **Multiple TTNN installations conflict**

   - *Cause*: mixing the PyPI `ttnn` wheel with the source build inside the same environment.
   - *Fix*: `pip uninstall ttnn` before running `create_venv.sh`, or recreate the virtual environment from scratch.

4. **Wrong Python interpreter**

   - *Cause*: running the host `python3` instead of `${TT_METAL_HOME}/python_env/bin/python`.
   - *Fix*: activate the virtual environment prior to building or testing (`source ${TT_METAL_HOME}/python_env/bin/activate`).

## Quick validation checklist

```bash
# Verify versions
python -m pip show torch ttnn torch-ttnn

# Inspect extension dependencies
python -c "import torch_ttnn_cpp_extension.ttnn_device_extension as mod; print(mod.__file__)"
ldd $(python -c "import torch_ttnn_cpp_extension.ttnn_device_extension as mod; print(mod.__file__)")

# Confirm tt-metal tag
pushd "${TT_METAL_HOME}" && git describe --tags && popd
```

This reference matches the CI workflow defined in `.github/workflows/run-cpp-native-tests.yaml` and should be kept in sync with future toolchain bumps.
