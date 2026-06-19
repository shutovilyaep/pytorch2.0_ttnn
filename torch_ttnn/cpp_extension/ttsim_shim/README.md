# TTSim Shim Library

## ⚠️ UGLY WORKAROUND ⚠️

**This is a temporary workaround for TTSim/UMD version incompatibility.**

This shim library should be removed once Tenstorrent releases a fixed TTSim version.

**Tracking Issue:** https://github.com/tenstorrent/ttsim/issues/4

## Problem

The tt-metal UMD (User-Mode Driver) v0.62.0+ requires two symbols that TTSim v1.3.6 doesn't export:

- `libttsim_tensix_reset_deassert(int x, int y)`
- `libttsim_tensix_reset_assert(int x, int y)`

These symbols were added to UMD in commit `b230cec5` (Sep 4, 2025) as part of PR [tt-umd#1266](https://github.com/tenstorrent/tt-umd/pull/1266).

However, TTSim v1.3.6 (released Feb 20, 2026) still doesn't export them, causing runtime errors:

```
RuntimeError: TT_THROW @ simulation_device.cpp:93: tt::exception
info: Failed to find '%s' symbol: libttsim_tensix_reset_deassert
```

## Solution

This shim library:

1. **Provides no-op implementations** for the missing symbols (`libttsim_tensix_reset_*`)
2. **Forwards all other calls** to the real TTSim library via `dlsym`

The Tensix reset functions are used to control core reset state in the simulator. Since TTSim doesn't implement them, providing no-op stubs allows the UMD to initialize without crashing.

## Usage

### Building

```bash
./build_shim.sh
```

This produces `libttsim_shim.so`.

### CI Integration

1. Download real TTSim library to `/tmp/ttsim/libttsim_real.so`
2. Build the shim: `./build_shim.sh`
3. Copy shim to `/tmp/ttsim/libttsim.so`
4. Set environment: `export TT_METAL_SIMULATOR=/tmp/ttsim/libttsim.so`

The shim will automatically load the real library from `TTSIM_SHIM_REAL_LIB` env var (defaults to `/tmp/ttsim/libttsim_real.so`).

## When to Remove

Remove this workaround when:

1. Tenstorrent releases a TTSim version that exports `libttsim_tensix_reset_deassert` and `libttsim_tensix_reset_assert`
2. Update CI to use the new TTSim version directly
3. Delete the `ttsim_shim/` directory

## Files

- `ttsim_shim.c` - Shim implementation
- `CMakeLists.txt` - Build configuration
- `build_shim.sh` - Build script for CI
- `README.md` - This file
