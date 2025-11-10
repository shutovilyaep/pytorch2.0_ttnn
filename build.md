# Build Guide

This document describes the supported workflow for building the PyTorch TTNN
native extension and running the C++ regression tests. The flow mirrors the
continuous-integration job and keeps the project aligned with the
**tt-metal&nbsp;0.63.0** toolchain.

## 1. Prerequisites

### System packages

Install the host dependencies (Ubuntu 22.04 or later):

```bash
sudo apt update
sudo apt install -y git-lfs cmake ninja-build python3 python3-venv python3-pip \
    build-essential clang-17 llvm-17-dev ccache
```

The `clang-17` toolchain matches the compilers used by tt-metal. `git-lfs` is
required because the tt-metal submodule tracks large binaries via LFS.

### Repository checkout

Clone the repository and sync the tt-metal submodule:

```bash
git clone https://github.com/tenstorrent/pytorch2.0_ttnn.git
cd pytorch2.0_ttnn
git submodule update --init --recursive
```

The tt-metal submodule tracks the `v0.63.0` release (see the `TT_METAL_REF`
setting in `.github/workflows/run-cpp-native-tests.yaml`). If the checkout falls
back to a different version, explicitly reset it:

```bash
pushd torch_ttnn/cpp_extension/third-party/tt-metal
git fetch --tags
git checkout v0.63.0
git submodule update --init --recursive
popd
```

## 2. Build tt-metal

All native builds rely on the libraries produced by tt-metal. Build them once
and keep the environment around for subsequent iterations:

```bash
export TT_METAL_HOME="$(pwd)/torch_ttnn/cpp_extension/third-party/tt-metal"
cd "$TT_METAL_HOME"
./install_dependencies.sh
./build_metal.sh --release --enable-ccache
./create_venv.sh
source ./python_env/bin/activate
```

The commands above produce:

| Artifact | Location |
| --- | --- |
| Compiled tt-metal libraries | `TT_METAL_HOME/build` (symlink to the active build directory, e.g. `build_Release`) |
| Python virtual environment | `TT_METAL_HOME/python_env` |
| Shared libraries exposed to dependants | `TT_METAL_HOME/build/lib` and `TT_METAL_HOME/build/tt_stl` (fall back to `build_Release/*` if the symlink is missing) |

The `create_venv.sh` script activates the virtual environment and installs the
Python-facing `ttnn` package that ships with tt-metal. Keep the environment
active for the remaining steps.

## 3. Install pytorch2.0_ttnn

From the repository root (with the virtual environment still activated):

```bash
cd /path/to/pytorch2.0_ttnn
python -m pip install --upgrade pip
python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .[dev]
```

The `pip install -e .[dev]` invocation builds the C++ extension via
`scikit-build-core` using the in-tree CMake project. The configuration phase
picks up the following automatically:

- `TT_METAL_HOME` – provides headers, shared libraries and the default toolchain
- `clang-17` – supplied through the tt-metal toolchain files
- PyTorch’s CMake configuration – discovered via the active Python interpreter

During the first build CMake generates the native build tree under
`torch_ttnn/cpp_extension/build/*`. Subsequent installs reuse the existing build
artifacts unless the directory is removed manually.

## 4. Running the C++ tests

Still inside the tt-metal virtual environment:

Run the regression suites that mirror the CI workflow:

```bash
python -m pytest tests/cpp_extension/test_cpp_extension_functionality.py -v
python -m pytest tests/cpp_extension/test_bert_cpp_extension.py -v
python -m pytest tests/models --native_integration -v
```

The tests link against the libraries installed in the virtual environment. If
the runtime loader cannot locate a dependency, append the tt-metal library
folders to `LD_LIBRARY_PATH`:

```bash
tt_lib_dir="${TT_METAL_HOME}/build/lib"
if [ ! -d "${tt_lib_dir}" ]; then
    tt_lib_dir="${TT_METAL_HOME}/build_Release/lib"
fi
tt_stl_dir="${TT_METAL_HOME}/build/tt_stl"
if [ ! -d "${tt_stl_dir}" ]; then
    tt_stl_dir="${TT_METAL_HOME}/build_Release/tt_stl"
fi
export LD_LIBRARY_PATH="${tt_lib_dir}:${tt_stl_dir}:${LD_LIBRARY_PATH:-}"
```

## 5. Rebuilding

Whenever you change C++ sources, re-run `pip install -e .[dev]`. The editable
install rebuilds the extension in place. To force a fully fresh build, delete
`torch_ttnn/cpp_extension/build` before reinstalling.

## 6. Troubleshooting

- Ensure the virtual environment created by `create_venv.sh` is active while
  building pytorch2.0_ttnn. Mixing Python interpreters can result in ABI
  mismatches.
- Verify that `TT_METAL_HOME` points to the freshly built tt-metal checkout.
- If `find_package(Torch)` fails, confirm that PyTorch is installed inside the
  virtual environment (`python -c "import torch; print(torch.__version__)"`).
- Use `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" python -m pip install -e .` to
  override the default Release configuration during iterative debugging.

For additional context see the CI configuration in
[.github/workflows/run-cpp-native-tests.yaml](.github/workflows/run-cpp-native-tests.yaml),
which implements the exact same sequence.
