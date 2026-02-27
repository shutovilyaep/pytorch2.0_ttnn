/*
 * TTSim Shim Library
 *
 * UGLY WORKAROUND for TTSim/UMD version incompatibility.
 * See: https://github.com/tenstorrent/ttsim/issues/4
 *
 * This shim provides no-op implementations for missing TTSim symbols
 * while forwarding all other calls to the real TTSim library.
 *
 * Remove this workaround when Tenstorrent releases a fixed TTSim version.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void* real_libttsim = NULL;
static int shim_initialized = 0;

// Function pointers for forwarded symbols
static int (*real_libttsim_init)(void) = NULL;
static void (*real_libttsim_exit)(void) = NULL;
static void (*real_libttsim_tile_rd_bytes)(int, int, uint64_t, void*, uint32_t) = NULL;
static void (*real_libttsim_tile_wr_bytes)(int, int, uint64_t, const void*, uint32_t) = NULL;
static void (*real_libttsim_clock)(int) = NULL;
static uint32_t (*real_libttsim_pci_config_rd32)(uint32_t) = NULL;
static void (*real_libttsim_pci_config_wr32)(uint32_t, uint32_t) = NULL;
static void (*real_libttsim_pci_mem_rd_bytes)(uint64_t, void*, uint32_t) = NULL;
static void (*real_libttsim_pci_mem_wr_bytes)(uint64_t, const void*, uint32_t) = NULL;
static void (*real_libttsim_set_pci_dma_mem_callbacks)(void*, void*) = NULL;

__attribute__((constructor))
static void init_shim(void) {
    if (shim_initialized) return;
    shim_initialized = 1;

    const char* real_path = getenv("TTSIM_SHIM_REAL_LIB");
    if (!real_path) {
        real_path = "/tmp/ttsim/libttsim_real.so";
    }

    fprintf(stderr, "[ttsim_shim] Loading real TTSim library from: %s\n", real_path);

    real_libttsim = dlopen(real_path, RTLD_NOW | RTLD_GLOBAL);
    if (!real_libttsim) {
        fprintf(stderr, "[ttsim_shim] ERROR: Could not load real libttsim: %s\n", dlerror());
        fprintf(stderr, "[ttsim_shim] Set TTSIM_SHIM_REAL_LIB to the path of the real libttsim.so\n");
        return;
    }

    // Load all forwarded symbols
    real_libttsim_init = dlsym(real_libttsim, "libttsim_init");
    real_libttsim_exit = dlsym(real_libttsim, "libttsim_exit");
    real_libttsim_tile_rd_bytes = dlsym(real_libttsim, "libttsim_tile_rd_bytes");
    real_libttsim_tile_wr_bytes = dlsym(real_libttsim, "libttsim_tile_wr_bytes");
    real_libttsim_clock = dlsym(real_libttsim, "libttsim_clock");
    real_libttsim_pci_config_rd32 = dlsym(real_libttsim, "libttsim_pci_config_rd32");
    real_libttsim_pci_config_wr32 = dlsym(real_libttsim, "libttsim_pci_config_wr32");
    real_libttsim_pci_mem_rd_bytes = dlsym(real_libttsim, "libttsim_pci_mem_rd_bytes");
    real_libttsim_pci_mem_wr_bytes = dlsym(real_libttsim, "libttsim_pci_mem_wr_bytes");
    real_libttsim_set_pci_dma_mem_callbacks = dlsym(real_libttsim, "libttsim_set_pci_dma_mem_callbacks");

    fprintf(stderr, "[ttsim_shim] Real TTSim library loaded successfully\n");
    fprintf(stderr, "[ttsim_shim] Providing no-op stubs for: libttsim_tensix_reset_deassert, libttsim_tensix_reset_assert\n");
}

__attribute__((destructor))
static void cleanup_shim(void) {
    if (real_libttsim) {
        dlclose(real_libttsim);
        real_libttsim = NULL;
    }
}

// =============================================================================
// MISSING SYMBOLS - No-op implementations
// These are required by UMD but not exported by TTSim v1.3.6
// =============================================================================

void libttsim_tensix_reset_deassert(int x, int y) {
    // No-op: TTSim v1.3.6 doesn't implement Tensix reset control
    // This function is called by UMD to deassert reset on a Tensix core
    (void)x;
    (void)y;
}

void libttsim_tensix_reset_assert(int x, int y) {
    // No-op: TTSim v1.3.6 doesn't implement Tensix reset control
    // This function is called by UMD to assert reset on a Tensix core
    (void)x;
    (void)y;
}

// =============================================================================
// FORWARDED SYMBOLS - Pass through to real TTSim library
// =============================================================================

int libttsim_init(void) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_init) {
        return real_libttsim_init();
    }
    fprintf(stderr, "[ttsim_shim] WARNING: libttsim_init not found in real library\n");
    return 0;
}

void libttsim_exit(void) {
    if (real_libttsim_exit) {
        real_libttsim_exit();
    }
}

void libttsim_tile_rd_bytes(int x, int y, uint64_t addr, void* data, uint32_t size) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_tile_rd_bytes) {
        real_libttsim_tile_rd_bytes(x, y, addr, data, size);
    } else {
        fprintf(stderr, "[ttsim_shim] WARNING: libttsim_tile_rd_bytes not found in real library\n");
    }
}

void libttsim_tile_wr_bytes(int x, int y, uint64_t addr, const void* data, uint32_t size) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_tile_wr_bytes) {
        real_libttsim_tile_wr_bytes(x, y, addr, data, size);
    } else {
        fprintf(stderr, "[ttsim_shim] WARNING: libttsim_tile_wr_bytes not found in real library\n");
    }
}

void libttsim_clock(int cycles) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_clock) {
        real_libttsim_clock(cycles);
    }
}

uint32_t libttsim_pci_config_rd32(uint32_t addr) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_pci_config_rd32) {
        return real_libttsim_pci_config_rd32(addr);
    }
    return 0;
}

void libttsim_pci_config_wr32(uint32_t addr, uint32_t data) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_pci_config_wr32) {
        real_libttsim_pci_config_wr32(addr, data);
    }
}

void libttsim_pci_mem_rd_bytes(uint64_t addr, void* data, uint32_t size) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_pci_mem_rd_bytes) {
        real_libttsim_pci_mem_rd_bytes(addr, data, size);
    }
}

void libttsim_pci_mem_wr_bytes(uint64_t addr, const void* data, uint32_t size) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_pci_mem_wr_bytes) {
        real_libttsim_pci_mem_wr_bytes(addr, data, size);
    }
}

void libttsim_set_pci_dma_mem_callbacks(void* read_cb, void* write_cb) {
    if (!shim_initialized) init_shim();
    if (real_libttsim_set_pci_dma_mem_callbacks) {
        real_libttsim_set_pci_dma_mem_callbacks(read_cb, write_cb);
    }
}
