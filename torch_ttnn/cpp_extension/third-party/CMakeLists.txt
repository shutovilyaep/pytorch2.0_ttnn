# Configure the tt-metal build used by the C++ extension. These options mirror
# the defaults in the upstream repository but are expressed as cache variables so
# they can be overridden by users.
set(CMAKE_DISABLE_PRECOMPILE_HEADERS TRUE CACHE BOOL "Disable PCH for tt-metal")
set(ENABLE_CCACHE TRUE CACHE BOOL "Enable ccache for tt-metal")
set(TT_UNITY_BUILDS ON CACHE BOOL "Build tt-metal with unity builds")
set(WITH_PYTHON_BINDINGS ON CACHE BOOL "Build tt-metal Python bindings")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build tt-metal shared libraries")

set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/tt-metal/.cpmcache" CACHE STRING "Path to CPM source cache")

add_subdirectory(tt-metal)

