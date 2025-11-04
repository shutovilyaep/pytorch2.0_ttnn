# PyTorch TTNN Build Dependencies and Process

## Overview

This project builds a PyTorch extension that integrates with Tenstorrent's TTNN (Tensor Neural Network) library. The build process involves multiple layers of dependencies and produces various artifacts that must be correctly linked and located at runtime.

## Key Components

### 1. Main Python Package (`torch-ttnn`)
- **Location**: Root `pyproject.toml`
- **Dependencies**:
  - `torch==2.2.1+cpu` (PyTorch)
  - `ttnn==0.60.1` (Tenstorrent TTNN from PyPI)
  - Various Python packages for ML/compilation
- **Artifacts**: Pure Python package with TTNN compiler

### 2. C++ Extension (`torch_ttnn_cpp_extension`)
- **Location**: `torch_ttnn/cpp_extension/`
- **Build System**: scikit-build-core + CMake
- **Purpose**: Provides C++ backend for PyTorch operations on TTNN devices

## Build Process Flow

### Phase 1: tt-metal Build (Dependency)
```
Input: tt-metal git submodule (third-party/tt-metal/)
Build: ./build_metal.sh --build-type Release --ttnn-shared-sub-libs --enable-ccache
Output: Static/shared libraries, headers, Python bindings
```

**Key Build Flags:**
- `--ttnn-shared-sub-libs`: Enables `ENABLE_TTNN_SHARED_SUBLIBS=ON` in CMake
- This builds tt_metal and tt_stl as shared libraries instead of static

**Artifacts Produced:**
- `libtt_metal.a` / `libtt_metal.so` (static/shared metal library)
- `libtt_stl.a` / `libtt_stl.so` (static/shared STL library)
- `_ttnncpp.so` (TTNN C++ Python extension)
- `_ttnn.so` (TTNN Python extension)
- Various other dependencies (fmt, spdlog, etc.)

### Phase 2: TTNN Installation
```
Input: PyPI package ttnn==0.60.1 (or built from source)
Output: Installed TTNN Python package with shared libraries
```

**When installed from PyPI:**
- Libraries placed in site-packages/ttnn/
- May include: `_ttnn.so`, `_ttnncpp.so`, `libtt_metal.so`, `libtt_stl.so`

**When built from source:**
- Libraries in tt-metal build directories
- Need to be discoverable via LD_LIBRARY_PATH

### Phase 3: C++ Extension Build
```
Input: C++ source code + tt-metal static libraries
Build: pip install -e . (scikit-build-core)
Output: torch_ttnn_cpp_extension Python package
```

**CMake Configuration:**
- `BUILD_SHARED_LIBS=OFF` - Forces static linking of tt-metal libraries
- `WITH_PYTHON_BINDINGS=ON` - Enables TTNN Python bindings
- Extension builds tt-metal as static libraries internally

**Linkage:**
- Extension statically links tt-metal libraries (libtt_metal.a, etc.)
- Only external dependencies: Torch, Python, system libs
- Produces self-contained `ttnn_device_extension.so`

## Runtime Library Dependencies

### With Full Static Linking (Current Implementation)
The extension embeds all tt-metal and ttnn code statically. Only these external dependencies are needed:

1. **PyPI TTNN libraries**: `_ttnn.so`, `_ttnncpp.so` (Python API)
2. **Torch libraries**: PyTorch native extensions
3. **Python runtime**: Standard Python shared libraries
4. **System libraries**: libc, libstdc++, libm, etc.

**No tt-metal shared libraries needed at runtime!**

### LD_LIBRARY_PATH Requirements
- **Before**: Complex setup with tt-metal + ttnn shared libs
- **After**: Minimal, only for PyPI TTNN package location

### Library Location Hierarchy

Libraries are searched in this order at runtime:

1. **rpath** (`$ORIGIN` in extension) - Libraries bundled with extension
2. **LD_LIBRARY_PATH** - Environment variable paths
3. **System paths** - `/usr/lib`, `/usr/local/lib`
4. **Package directories** - site-packages locations

## Common Build Issues

### Issue 1: Missing libtt_metal.so at Runtime
**Symptom:** `ModuleNotFoundError: No module named 'ttnn_device_extension'`
**Cause:** `libtt_metal.so` not in LD_LIBRARY_PATH during testing
**Solution:** Ensure tt-metal build used `--ttnn-shared-sub-libs` flag

### Issue 2: Static vs Shared Library Mismatch
**Symptom:** Link errors or missing symbols
**Cause:** tt-metal built with static libs, extension expects shared
**Solution:** Use `--ttnn-shared-sub-libs` when building tt-metal

### Issue 3: Multiple TTNN Installations
**Symptom:** ABI mismatches, crashes
**Cause:** PyPI TTNN + source-built TTNN both present
**Solution:** `pip uninstall ttnn` before building from source

## Build Configuration Analysis

### Current Implementation (Default: Static)
**Status**: ✅ **Static tt_metal + Static ttnn** (fully self-contained extension)

- **tt-metal build**: `BUILD_SHARED_LIBS=OFF` (static libraries)
- **TTNN source**: PyPI package provides runtime libraries
- **Extension linkage**: Static tt_metal + Static ttnn (embedded in .so)
- **Wheel contents**: Only `ttnn_device_extension.so` (self-contained)
- **Runtime needs**: Only PyPI TTNN libs + system libs
- **Advantages**: No shared library conflicts, minimal distribution

### Alternative: Shared Libraries (with Wheel Packaging)
**Status**: 🔄 **Shared tt_metal + Shared ttnn** (with proper wheel packaging)

- **Build**: `TORCH_TTNN_BUILD_SHARED_LIBS=ON` environment variable
- **Extension linkage**: Links against shared tt_metal.so + ttnn.so
- **Wheel contents**: Extension + bundled tt-metal shared libs
- **Install rules**: CMake automatically packages dependencies in wheel
- **Runtime**: Self-contained wheel, no external tt-metal libs needed
- **Advantages**: Can share libraries between components, easier debugging

### Previous Shared Approach (Broken)
**Status**: ❌ **Not recommended** (causes runtime conflicts)

- **tt-metal build**: `--ttnn-shared-sub-libs` (shared sublibraries)
- **Extension linkage**: Links against shared tt-metal/ttnn libs
- **Wheel contents**: Extension only (missing shared libs!)
- **Runtime needs**: All tt-metal shared libs must be in LD_LIBRARY_PATH
- **Problems**: Shared libs not included in wheel, runtime loading failures

### Implementation Details

**Static Linking (Current)**:
```cmake
# In extension/third-party/CMakeLists.txt
set(BUILD_SHARED_LIBS OFF)  # Forces static tt-metal build

# In tt-metal/ttnn/CMakeLists.txt (patched at build time)
if(BUILD_SHARED_LIBS)
    add_library(ttnncpp SHARED)  # For shared builds
else()
    add_library(ttnncpp STATIC)  # For static builds (patched)
endif()
```

**Patching Strategy:**
- tt-metal is a third-party submodule, so direct modifications are lost on CI checkout
- CI patches `ttnn/CMakeLists.txt` at build time to enable conditional static/shared builds
- Local development script also applies the same patch

## Build Configuration Matrix

| Scenario | Environment Variable | tt-metal build | TTNN source | Extension links to | Wheel contents | Runtime needs |
|----------|----------------------|----------------|-------------|-------------------|----------------|---------------|
| **CI/CD (Default)** | `TORCH_TTNN_BUILD_SHARED_LIBS` unset | Static libs | PyPI (0.60.1) | tt-metal static + PyPI shared | Extension only | PyPI libs only |
| **Shared Libs (Testing)** | `TORCH_TTNN_BUILD_SHARED_LIBS=ON` | Shared libs | PyPI | tt-metal shared + PyPI shared | Extension + tt-metal .so | Self-contained |
| **Previous (Broken)** | N/A | `--ttnn-shared-sub-libs` | PyPI/Source | tt-metal shared | Extension only | tt-metal + PyPI libs |

**Wheel Building**:
- scikit-build-core automatically includes the extension .so
- For static builds: no additional shared libs needed in wheel
- For shared builds: CMake install rules bundle tt-metal shared libs in wheel
- PyPI TTNN provides runtime dependencies separately

**Runtime Resolution**:
- Extension loads with embedded tt-metal/ttnn code
- PyPI TTNN provides `_ttnn.so`, `_ttnncpp.so` for Python API
- No LD_LIBRARY_PATH manipulation needed

## Artifact Locations

### tt-metal Build Outputs
```
build_Release/
├── lib/
│   ├── libtt_metal.a         # Static metal lib
│   ├── libtt_metal.so        # Shared metal lib (with --ttnn-shared-sub-libs)
│   ├── libtt_stl.so          # Shared STL lib (with --ttnn-shared-sub-libs)
│   ├── _ttnncpp.so           # TTNN C++ extension
│   └── _ttnn.so              # TTNN Python extension
└── include/                  # Headers for extension building
```

### C++ Extension Build Outputs
```
torch_ttnn/cpp_extension/build/lib.linux-x86_64-3.10/
├── ttnn_device_extension.cpython-310-x86_64-linux-gnu.so  # Main extension
├── libtt_metal.so          # Copied/linked from tt-metal
├── _ttnncpp.so            # Copied/linked from tt-metal
└── _ttnn.so               # Copied/linked from tt-metal
```

### Installed Package Structure
```
site-packages/
├── torch_ttnn/                    # Main Python package
├── torch_ttnn_cpp_extension/      # C++ extension package
│   └── ttnn_device_extension.so   # The extension module
└── ttnn/                          # TTNN package (from PyPI or source)
    ├── _ttnn.so
    ├── _ttnncpp.so
    ├── libtt_metal.so
    └── libtt_stl.so
```

## Runtime Library Resolution

The extension uses several strategies to locate libraries:

1. **Build-time rpath**: `$ORIGIN` points to extension directory
2. **Environment LD_LIBRARY_PATH**: Set in CI/test scripts
3. **Dynamic search**: Python code searches site-packages and tt-metal dirs
4. **Fallback search**: tt-metal build directories

### LD_LIBRARY_PATH Construction (from CI)
```bash
# Base tt-metal dirs
LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${TT_METAL_HOME}/build/lib"

# TTNN package libs
LD_LIBRARY_PATH="${TTNN_LIB_DIR}:${LD_LIBRARY_PATH}"

# MPI libs (for distributed)
LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:${LD_LIBRARY_PATH}"

# Found tt_stl/tt_metal dirs
LD_LIBRARY_PATH="${STL_LIB_DIR}:${METAL_LIB_DIR}:${LD_LIBRARY_PATH}"
```

## Debugging Library Issues

### Check Extension Dependencies
```bash
# Find extension .so file
find . -name "*ttnn_device_extension*.so"

# Check dependencies
ldd /path/to/ttnn_device_extension.so

# See what Python finds
python3 -c "import ttnn_device_extension; print(ttnn_device_extension.__file__)"
```

### Check TTNN Package
```bash
# Find TTNN installation
python3 -c "import ttnn; import pathlib; print(pathlib.Path(ttnn.__file__).parent)"

# List TTNN libraries
python3 -c "import ttnn, pathlib, os; p=pathlib.Path(ttnn.__file__).parent; [print(f) for f in os.listdir(p) if '.so' in f]"
```

### Check LD_LIBRARY_PATH
```bash
echo $LD_LIBRARY_PATH
# Should include paths to all required .so files
```

## Testing Different Build Configurations

### Static Build (Default)
```bash
# Default behavior - static linking
pip install -e torch_ttnn/cpp_extension
```

### Shared Build (Testing)
```bash
# Force shared libraries build
TORCH_TTNN_BUILD_SHARED_LIBS=ON pip install -e torch_ttnn/cpp_extension

# Check that shared libs are included in wheel
find torch_ttnn/cpp_extension/build -name "*.so"
```

### Local Script Testing
```bash
# Test static build (default)
./scripts/run-cpp-native-tests.sh

# Test shared build
TORCH_TTNN_BUILD_SHARED_LIBS=ON ./scripts/run-cpp-native-tests.sh
```

## Recommendations

1. **✅ Default to static linking**: Most reliable, no runtime issues
2. **Test shared libs occasionally**: Ensure wheel packaging works correctly
3. **Clean builds**: Remove build artifacts when switching linkage types
4. **PyPI TTNN**: Use official package for runtime dependencies
5. **Verify with ldd**: Check extension dependencies after build
