# TTSim Shim Library
#
# UGLY WORKAROUND for TTSim/UMD version incompatibility.
# See: https://github.com/tenstorrent/ttsim/issues/4
#
# Remove this workaround when Tenstorrent releases a fixed TTSim version.

cmake_minimum_required(VERSION 3.16)
project(ttsim_shim C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build as shared library
add_library(ttsim_shim SHARED ttsim_shim.c)

# Link with dl for dlopen/dlsym
target_link_libraries(ttsim_shim PRIVATE ${CMAKE_DL_LIBS})

# Export all symbols
set_target_properties(ttsim_shim PROPERTIES
    C_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
)

# Install to build directory
install(TARGETS ttsim_shim
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build
)
