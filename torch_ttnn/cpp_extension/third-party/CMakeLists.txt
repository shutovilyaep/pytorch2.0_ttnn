# Some flags for tt-metal -- ccache + unity build + make sure that cpm cache stays in tt-metal dir
set(CMAKE_DISABLE_PRECOMPILE_HEADERS TRUE)
set(ENABLE_CCACHE TRUE)
set(TT_UNITY_BUILDS ON)
set(WITH_PYTHON_BINDINGS ON)
set(BUILD_SHARED_LIBS OFF)  # Build static libraries to avoid runtime loading issues
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache" CACHE STRING "Path to CPM source cache")

add_subdirectory(tt-metal)

