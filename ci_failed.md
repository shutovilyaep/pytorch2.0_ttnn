# CI Build Failure Analysis

## Error Summary

**Date**: 2025-11-04  
**Workflow**: `run-cpp-native-tests.yaml`  
**Step**: Build C++ Extension → Installing `ttnn` package  
**Exit Code**: 1

## Error Details

### Primary Error
```
CMake Error at CMakeLists.txt:260 (target_link_libraries):
  Target "metal_common_pch" links to:
    nlohmann_json::nlohmann_json
  but the target was not found.
```

### Root Cause

CPM (CMake Package Manager) failed to properly resolve `nlohmann_json` package due to git operations failing in the CPM cache directory:

```
CMake Warning at .cpmcache/cpm/CPM_0.40.2.cmake:429 (message):
  CPM: Calling git status on folder
  /__w/pytorch2.0_ttnn/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308
  failed
```

### Why This Happens

1. **CPM Git Check**: CPM performs `git status` checks on cached packages to verify integrity
2. **Git Safe Directory**: Git refuses to operate on directories not in `safe.directory` list
3. **Missing Configuration**: The CPM cache directory (`.cpmcache/`) was not added to git's safe.directory list
4. **Target Creation Failure**: Without successful git check, CPM cannot properly create the `nlohmann_json::nlohmann_json` CMake target
5. **CMake Errors**: Multiple targets fail to link because `nlohmann_json::nlohmann_json` doesn't exist

### Affected Targets

The following CMake targets failed to link against `nlohmann_json::nlohmann_json`:
- `metal_common_pch`
- `tt_stl`
- `tt_metal`
- `common`
- `llrt`

## Solution

Add CPM cache directory to git's safe.directory configuration before building:

```bash
git config --global --add safe.directory "${TT_METAL_HOME}/.cpmcache/*"
```

Or add the parent directory pattern:

```bash
git config --global --add safe.directory "${TT_METAL_HOME}/.cpmcache"
```

However, git's `safe.directory` doesn't support wildcards directly. Better approach is to add the specific cache directories after they're created, or add the parent directory.

## Implementation

The fix has been added in the "Build C++ Extension" step, right after configuring git for `TT_METAL_HOME`:

```yaml
# Ensure git is available and configured (needed for version detection and CPM)
git config --global --add safe.directory "${TT_METAL_HOME}" || true
# CPM performs git operations in cache directories - add them to safe.directory
CPM_CACHE_DIR="${TT_METAL_HOME}/.cpmcache"
if [ -d "${CPM_CACHE_DIR}" ]; then
  git config --global --add safe.directory "${CPM_CACHE_DIR}" || true
  # Add all git repositories in CPM cache (CPM checks git status in cached packages)
  find "${CPM_CACHE_DIR}" -type d -name ".git" 2>/dev/null | while read git_dir; do
    cache_dir=$(dirname "$git_dir")
    git config --global --add safe.directory "$cache_dir" || true
  done
fi
export CPM_SOURCE_CACHE="${CPM_CACHE_DIR}"
```

**Note**: We create the cache directory with `mkdir -p` before configuring git, ensuring git can operate on it even before CMake runs. This prevents CPM from failing when it tries to check git status in newly created cache directories.

## Similar Issues

This is the same type of issue we encountered earlier with CPM and git operations. The pattern is:
1. CPM downloads packages to cache directories
2. CPM performs git operations to verify packages
3. Git refuses operations on untrusted directories
4. CMake configuration fails

## Prevention

When working with CPM in CI environments:
1. Always configure `git config --global --add safe.directory` for:
   - The main repository directory
   - CPM cache directories (`.cpmcache/`)
   - Any directories where git operations might occur
2. Consider setting `CPM_SOURCE_CACHE` explicitly to a known location
3. Add git configuration early in the build process, before CMake runs

## Known Warnings (Non-Critical)

### Warning: integer expression expected in install_dependencies.sh

**Symptoms:**
```
./install_dependencies.sh: line 377: [: 22.04.5 LTS (Jammy Jellyfish): integer expression expected
```

**Cause:**
- Bug in `install_dependencies.sh` line 375: uses `$VERSION` instead of `$OS_VERSION`
- `$VERSION` contains full string like "22.04.5 LTS (Jammy Jellyfish)"
- Comparison `[ "$VERSION_NUM" -gt "2404" ]` fails because `VERSION_NUM` becomes invalid string

**Impact:**
- **Non-critical**: Warning only, installation continues successfully
- MPI ULFM installation proceeds regardless of this check
- Script completes with success message

**Action:**
- **No action needed**: Warning can be safely ignored
- If fixing in tt-metal: change line 375 from `$VERSION` to `$OS_VERSION`
- This is a bug in tt-metal submodule, not our code

## Issue: Intermittent CPM Cache Git Failures

### Problem

After a successful build, changing only test files causes CI to fail with the same CPM git error, even though git configuration appears correct.

### Analysis

**Hypothesis**: The git configuration for CPM cache directories may not persist correctly between CI runs, or new cache directories created during CMake execution are not added to `safe.directory` in time.

**Possible Causes**:
1. **Timing Issue**: CPM creates new cache directories during CMake execution, and git tries to check them before they're added to `safe.directory`
2. **Cache Persistence**: CI caches `.cpmcache` between runs, but git configuration may not persist correctly
3. **Subdirectory Path Issue**: Git `safe.directory` requires exact paths - parent directory inclusion may not work for deeply nested subdirectories

### Investigation Required

The current implementation:
- Creates `.cpmcache` directory
- Adds parent directory to `safe.directory`
- Finds existing `.git` directories and adds them

**Issue**: When CPM creates a NEW directory during CMake execution (e.g., `.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308`), it immediately tries to run `git status` before our `find` command can discover and add it.

### Solution Options

1. **Use `git config --global --add safe.directory` with wildcard pattern** (if supported)
2. **Set `GIT_CONFIG_GLOBAL` environment variable** to disable safe.directory checks for CI
3. **Pre-configure all known CPM package paths** before CMake runs
4. **Use `GIT_CONFIG_COUNT` and `GIT_CONFIG_KEY_*` environment variables** to set config without file access
5. **Wrap CMake execution** to monitor and add new directories as they're created (complex)

### Recommended Fix

Since git `safe.directory` checks are security features, the best approach is to ensure all cache directories are added before CMake runs. However, if CPM creates directories dynamically, we need to:

1. **Add the entire workspace to safe.directory** (less secure but more reliable for CI) ✅ **IMPLEMENTED**
2. **Use `GIT_CONFIG_GLOBAL`** to set config via environment variables (alternative approach)
3. **Pre-create expected cache directories** based on CPM package list (complex)

**Implementation**: Added git configuration at the start of "Build C++ Extension" step:

```yaml
# Configure git safe.directory for entire workspace (covers all subdirectories including CPM cache)
# This prevents race conditions when CPM creates new cache directories during CMake execution
git config --global --add safe.directory "${{ github.workspace }}" || true
git config --global --add safe.directory "${TT_METAL_HOME}" || true
```

This ensures that any directory created by CPM during CMake execution is automatically covered by the workspace safe.directory configuration, eliminating the race condition.

**Note**: This issue appears intermittently, suggesting it may be related to CI cache state or the order of operations when CPM downloads packages. The fix addresses this by covering the entire workspace upfront.

### Current Status: Warning Still Appears (Non-Critical)

Even after adding the workspace to `safe.directory`, the CPM warning may still appear:

```
CMake Warning at .cpmcache/cpm/CPM_0.40.2.cmake:429 (message):
  CPM: Calling git status on folder
  /__w/pytorch2.0_ttnn/pytorch2.0_ttnn/torch_ttnn/cpp_extension/third-party/tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308
  failed
```

**Important**: This is a **Warning**, not an **Error**. The build continues successfully despite this warning. CPM can still function even if `git status` fails - it just can't verify package integrity via git.

**Why it still appears**:
- Git's `safe.directory` may require exact paths for deeply nested subdirectories
- CPM creates cache directories dynamically during CMake execution
- Even with workspace in `safe.directory`, git may check subdirectories before they're recognized

**Action**: 
- **No action needed** if build succeeds (warning is harmless)
- If you want to suppress the warning, you could disable CPM's git checks via environment variable (not recommended, as it reduces package verification)
- The warning does not affect build success or functionality

## Issue: Double Free or Corruption in tt-metal 0.60.1

### Problem

When running tests (especially BERT tests), the process crashes with:
```
double free or corruption (!prev)
```

This error occurs during test cleanup, after tests complete successfully. The issue persists even when commenting out most tests and leaving only BERT tests.

### Root Cause Analysis

**Location**: `torch_ttnn/cpp_extension/ttnn_cpp_extension/src/core/TtnnGuard.cpp`

The problem is a known issue in tt-metal 0.60.1 related to device lifecycle management:

```cpp
ttnn::MeshDevice* TtnnGuard::get_open_ttnn_device(c10::DeviceIndex device_index) {
    if (!ttnn_device) {
        ttnn_device = [device_index] {
            auto sp = ttnn::open_mesh_device(device_index);
            // Intentional memory leak to avoid destruction order issues
            // TODO: might be a problem if the device is closed and opened many times
            static auto* keeper = new std::shared_ptr<ttnn::MeshDevice>(std::move(sp));
            return keeper->get();
        }();
    }
    return ttnn_device;
}
```

**The Problem**:
1. `TtnnGuard` uses a **static pointer** to hold the device (`static auto* keeper`)
2. This is intentionally leaked to avoid destruction order issues
3. In pytest, the `device` fixture (scope="session") closes the device at the end of the test session via `ttnn.close_device(device)`
4. When the device is closed, tt-metal's internal cleanup deallocates device resources
5. However, the static pointer in `TtnnGuard` still holds a reference to the closed device
6. When Python's garbage collector or other destructors try to free memory later, they encounter already-freed resources → **double free**

**Why It Happens in BERT Tests**:
- BERT tests create many tensors and complex models
- These tensors hold references to the device through `TtnnTensorImpl`
- When the test session ends, pytest closes the device
- But tensors and the static device pointer are still in memory
- During cleanup, multiple paths try to free the same memory

### Evidence

1. **Comment in code**: "TODO: might be a problem if the device is closed and opened many times"
2. **Known bug**: User reports this is fixed in later tt-metal versions
3. **Intermittent nature**: Appears during cleanup, not during test execution
4. **Session-scoped fixture**: Device is opened once and closed once, matching the static pointer pattern

### Workarounds (Without Upgrading tt-metal)

Since upgrading tt-metal requires PyTorch API changes, here are workarounds:

#### Option 1: Change Device Fixture Scope (Recommended)

Change the device fixture scope from `"session"` to `"function"` or `"module"`:

```python
@pytest.fixture(scope="function")  # Changed from "session"
def device(request):
    # ... existing code ...
    yield device
    ttnn.synchronize_device(device)
    ttnn.close_device(device)
```

**Pros**: 
- Prevents device from being closed while static pointer still references it
- Each test gets a fresh device

**Cons**:
- Slower (device opened/closed for each test)
- May not work if tests need persistent device state

#### Option 2: Explicit Cleanup After Device Close + Use MeshDevice ✅ **IMPLEMENTED**

Two changes:
1. **Use `mesh_device` even for single device** (recommended in tt-metal)
2. **Call `gc.collect()` AFTER closing device** to clear Python references after device resources are freed

```python
@pytest.fixture(scope="session")
def device(request):
    # Use mesh_device even for single device (recommended in tt-metal)
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), dispatch_core_config=dispatch_core_config, l1_small_size=l1_small_size
    )
    
    yield device
    
    ttnn.synchronize_device(device)
    ttnn.close_mesh_device(device)
    
    # Explicit cleanup AFTER closing device
    import gc
    gc.collect()  # Force Python garbage collection to clear tensor references
```

**Why this helps**:
- `mesh_device` uses `shared_ptr` which better manages device lifecycle
- `gc.collect()` after close ensures Python references are cleared after device resources are freed
- This reduces chance of double free from static pointer in `TtnnGuard`

**Pros**: 
- Keeps session scope
- Uses recommended mesh_device API
- Cleanup happens after device resources are freed

**Cons**:
- May not fully solve the static pointer issue (C++ code problem)
- Less reliable than changing fixture scope

#### Option 3: Patch TtnnGuard to Reset Static Pointer

Add a cleanup function to reset the static pointer:

```cpp
// In TtnnGuard.cpp
void TtnnGuard::reset_device() {
    ttnn_device = nullptr;
}
```

Then call it before closing device in pytest fixture.

**Pros**: 
- Addresses root cause
- Clean solution

**Cons**:
- Requires modifying C++ code
- May break if device is still referenced elsewhere

#### Option 4: Skip Cleanup (Not Recommended)

Comment out device closing in fixture (leaves device open):

```python
@pytest.fixture(scope="session")
def device(request):
    # ... open device ...
    yield device
    # ttnn.synchronize_device(device)  # Commented out
    # ttnn.close_device(device)  # Commented out
```

**Pros**: 
- Quick fix
- Avoids double free

**Cons**:
- **Not recommended**: Device resources not properly released
- May cause issues in CI/long-running tests
- Memory leaks

### Recommended Solution

**Short-term**: Use Option 1 (change fixture scope to `"function"`) for tests that fail:
- Update `tests/conftest.py` to use `scope="function"` for the device fixture
- Or create a separate fixture for BERT tests with `scope="function"`

**Long-term**: Upgrade tt-metal when PyTorch API changes are ready (this bug is fixed in later versions)

### Testing the Fix

After applying a workaround:
1. Run BERT tests multiple times
2. Check for double free errors
3. Monitor memory usage (should not leak)
4. Verify tests still pass

## References

- CPM documentation: https://github.com/cpm-cmake/CPM.cmake
- Git safe.directory: https://git-scm.com/docs/git-config#Documentation/git-config.txt-safedirectory
- Previous fix: Added `git config --global --add safe.directory "${TT_METAL_HOME}"` but missed CPM cache subdirectories
- tt-metal issue: Device lifecycle management in version 0.60.1 (fixed in later versions)
