# Build and Test Issues Documentation

## Overview

This document describes build and runtime issues encountered when building and testing the PyTorch TTNN C++ extension.

## Issue 1: Undefined Symbol - BinaryOperation::invoke

### Symptoms

When importing the C++ extension module, the following error occurs:

```
undefined symbol: _ZN4ttnn10operations6binary15BinaryOperationILNS1_12BinaryOpTypeE0EE6invokeEN4ttsl10StrongTypeIhNS_10QueueIdTagEEERKN2tt8tt_metal6TensorESD_RKSt8optionalIKNSA_8DataTypeEERKSE_INSA_12MemoryConfigEERKSE_ISB_ESt4spanIKNS0_5unary14UnaryWithParamELm18446744073709551615EESV_SV_RKSE_IbE
```

Demangled symbol:
```
ttnn::operations::binary::BinaryOperation<(ttnn::operations::binary::BinaryOpType)0>::invoke(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::DataType const> const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<tt::tt_metal::Tensor> const&, std::span<ttnn::operations::unary::UnaryWithParam const, 18446744073709551615ul>, std::span<ttnn::operations::unary::UnaryWithParam const, 18446744073709551615ul>, std::span<ttnn::operations::unary::UnaryWithParam const, 18446744073709551615ul>, std::optional<bool> const&)
```

### Root Cause

The symbol exists in `_ttnncpp.so` but is marked as a **weak symbol** (`W` in `nm` output). Weak symbols can be overridden and may not be properly resolved during dynamic linking if:

1. The symbol is not properly exported from the library
2. Library loading order is incorrect
3. There are version mismatches between the extension and tt-metal libraries

### Investigation

1. **Symbol exists but is weak**: `nm -D _ttnncpp.so | grep BinaryOperation` shows the symbol exists but is marked as `W` (weak)
2. **Used in code**: The extension uses `ttnn::add()` which internally calls `BinaryOperation<0>::invoke()`
3. **Library dependencies**: The extension depends on `_ttnncpp.so` which should contain this symbol

### Solution

The issue is likely due to:
- **Template instantiation**: `BinaryOperation` is a template class, and the specific instantiation may not be properly exported
- **Library linking order**: The extension may need to explicitly link against tt-metal binary operation libraries

### Workaround

1. Ensure `LD_LIBRARY_PATH` includes the tt-metal build directory:
   ```bash
   export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH}"
   ```

2. Use `LD_PRELOAD` to force-load tt-metal libraries:
   ```bash
   export LD_PRELOAD="${TT_METAL_HOME}/build/lib/_ttnncpp.so"
   ```

3. Rebuild the extension to ensure it's linked against the correct tt-metal version:
   ```bash
   cd torch_ttnn/cpp_extension
   pip install -e . --force-reinstall --no-cache-dir
   ```

## Issue 2: Undefined Symbol - Tensor::to_device (RESOLVED)

### Symptoms

Previous error with `Tensor::to_device` method not being found.

### Root Cause

The `Tensor::to_device` method is not exported from tt-metal libraries. The extension was calling `tensor.to_device(device)` which uses a non-exported method.

### Solution

Changed code to use the exported function `ttnn::operations::core::to_device()`:

```cpp
// Before (doesn't work):
ttnn::Tensor src_dev = src_cpu.to_device(ttnn_device);

// After (works):
ttnn::Tensor src_dev = ttnn::operations::core::to_device(src_cpu, ttnn_device, std::nullopt);
```

**Files modified:**
- `torch_ttnn/cpp_extension/ttnn_cpp_extension/src/core/copy.cpp`

## Issue 3: MeshDevice API Migration

### Background

TT-Metal migrated from `Device` API to `MeshDevice` API. The extension code was updated to use `MeshDevice*` instead of `Device*`.

### Changes Required

1. **Device type**: Changed from `ttnn::Device*` to `ttnn::MeshDevice*`
2. **to_device calls**: Use `ttnn::operations::core::to_device()` instead of `tensor.to_device()`
3. **Device opening**: Use `MeshDevice::create_unit_meshes()` or similar MeshDevice APIs

## Testing

### Isolated Test Execution

Use `scripts/run-cpp-extension-tests-only.sh` to run tests without rebuilding:

```bash
./scripts/run-cpp-extension-tests-only.sh
```

This script:
- Sets up `LD_LIBRARY_PATH` correctly
- Finds and loads the extension module
- Provides detailed diagnostics for import failures
- Detects undefined symbol errors and suggests solutions

### Debugging Import Issues

The script provides:
1. **Symbol checking**: Checks if undefined symbols exist in tt-metal libraries
2. **Library path diagnostics**: Shows all library paths being searched
3. **Import diagnostics**: Detailed error messages with demangled symbols

## Recommendations

1. **Always rebuild after tt-metal updates**: The extension must be rebuilt when tt-metal is updated
2. **Use exported APIs**: Prefer using exported functions from `ttnn::operations::core::` namespace
3. **Check symbol visibility**: Use `nm -D` to check if symbols are exported (marked with `T` or `W`)
4. **Library loading order**: Ensure tt-metal libraries are loaded before the extension

## Future Work

1. **Explicit template instantiation**: Consider explicitly instantiating `BinaryOperation<0>` in the extension to ensure the symbol is available
2. **Static linking option**: Investigate if static linking can resolve weak symbol issues
3. **Symbol export verification**: Add build-time checks to verify all required symbols are available

## Recommended Solution for PR

### Immediate Fix

For the PR, add `LD_PRELOAD` to ensure `_ttnncpp.so` is loaded before the extension. This resolves weak symbol issues:

```bash
# In run-cpp-extension-tests-only.sh, add before import check:
if [[ -f "${TT_METAL_HOME}/build/lib/_ttnncpp.so" ]]; then
  export LD_PRELOAD="${TT_METAL_HOME}/build/lib/_ttnncpp.so:${LD_PRELOAD:-}"
elif [[ -f "${TT_METAL_HOME}/build_Release/lib/_ttnncpp.so" ]]; then
  export LD_PRELOAD="${TT_METAL_HOME}/build_Release/lib/_ttnncpp.so:${LD_PRELOAD:-}"
fi
```

This ensures weak symbols from `_ttnncpp.so` are available when the extension loads.

### Long-term Solution

1. **Investigate template instantiation**: Work with tt-metal team to ensure `BinaryOperation<ADD>` is properly exported
2. **Consider explicit instantiation**: Add explicit template instantiation in extension if needed
3. **Documentation**: Update build.md with final solution once root cause is identified

