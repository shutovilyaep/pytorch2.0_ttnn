# Some flags for tt-metal -- ccache + unity build + make sure that cpm cache stays in tt-metal dir
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME environment variable is not set")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# TODO: figure out the way to fetch those from ttnn
# Glob to find needed includes from cpm cache
file(GLOB TTNN_INCLUDE_DIRS 
    "${TT_METAL_HOME}/.cpmcache/reflect/*"
    "${TT_METAL_HOME}/.cpmcache/fmt/*/include"
    "${TT_METAL_HOME}/.cpmcache/magic_enum/*/include"
    "${TT_METAL_HOME}/.cpmcache/nlohmann_json/*/include"
    "${TT_METAL_HOME}/.cpmcache/boost/*/*/*/include"
)

# Determine include roots depending on layout (wheel vs submodule)
set(_TTNN_INC_CANDIDATES
    # ttnn public API headers (decorators.hpp, device.hpp, tensor.hpp, etc.)
    ${TT_METAL_HOME}/ttnn/api
    ${TT_METAL_HOME}/cpp
    ${TT_METAL_HOME}/ttnn/cpp
    ${TT_METAL_HOME}/ttnn
    ${TT_METAL_HOME}/tt_metal/api
    ${TT_METAL_HOME}/tt_metal/third_party
    ${TT_METAL_HOME}/tt_metal/third_party/umd
    ${TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    ${TT_METAL_HOME}/tt_metal/hostdevcommon/api
    ${TT_METAL_HOME}/tt_metal/third_party/tracy/public
    ${TT_METAL_HOME}/tt_stl
    ${TT_METAL_HOME}/build/include
)
set(TTNN_INCLUDE_DIRS)
foreach(dir IN LISTS _TTNN_INC_CANDIDATES)
    if(EXISTS ${dir})
        list(APPEND TTNN_INCLUDE_DIRS ${dir})
    endif()
endforeach()
list(APPEND TTNN_INCLUDE_DIRS ${TTNN_INCLUDE_DIRS})

# Add specific third-party includes discovered in CPM cache (tt-logger)
if(EXISTS ${TT_METAL_HOME}/.cpmcache/tt-logger)
    file(GLOB _TT_LOGGER_INC "${TT_METAL_HOME}/.cpmcache/tt-logger/*/include")
    list(APPEND TTNN_INCLUDE_DIRS ${_TT_LOGGER_INC})
endif()

# Now wrap all the headers and .so files nicely into one target
if(NOT TARGET Metalium::TTNN)
    # Prefer an explicitly provided lib directory (env or CMake var),
    # otherwise use the submodule build output
    if(DEFINED TTNN_LIB_DIR)
        set(METALIUM_LIB_PATH "${TTNN_LIB_DIR}")
    elseif(DEFINED ENV{TTNN_LIB_DIR})
        set(METALIUM_LIB_PATH "$ENV{TTNN_LIB_DIR}")
    else()
        set(METALIUM_LIB_PATH "${TT_METAL_HOME}/build/lib")
    endif()
    find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    find_library(DEVICE_LIBRARY NAMES "device" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    # Wheel may ship as _ttnncpp.so or _ttnn.so
    find_library(TTNN_LIBRARY NAMES "_ttnncpp.so" "_ttnn.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)

    if(TT_METAL_LIBRARY)
        add_library(Metalium::Metal SHARED IMPORTED GLOBAL)
        set_target_properties(Metalium::Metal PROPERTIES
            IMPORTED_LOCATION "${TT_METAL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TTNN_INCLUDE_DIRS}"
        )
        target_link_libraries(
        Metalium::Metal
        INTERFACE
            ${DEVICE_LIBRARY}
        )
        message(STATUS "Successfully found libtt_metal.so at ${TT_METAL_LIBRARY}")
    else()
        message(FATAL_ERROR "libtt_metal.so not found in ${METALIUM_LIB_PATH}")
    endif()
    if(TTNN_LIBRARY)
        add_library(Metalium::TTNN SHARED IMPORTED)
        set_target_properties(
            Metalium::TTNN
            PROPERTIES
                IMPORTED_LOCATION "${TTNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TTNN_INCLUDE_DIRS}"
        )
        message(STATUS "Successfully found TTNN shared library at ${TTNN_LIBRARY}")
    else()
        message(FATAL_ERROR "TTNN shared library (_ttnncpp.so or _ttnn.so) not found in ${METALIUM_LIB_PATH}")
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()
