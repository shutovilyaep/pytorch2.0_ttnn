# Some flags for tt-metal -- ccache + unity build + make sure that cpm cache stays in tt-metal dir
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME environment variable is not set")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# Prefer relying on tt-metal exported CMake targets instead of manual include hacks
set(TTNN_INCLUDE_DIRS "")

set(TTNN_INCLUDE_DIRS
    ${TT_METAL_HOME}/ttnn/cpp
    ${TT_METAL_HOME}/ttnn
    ${TT_METAL_HOME}/ttnn/api
    ${TT_METAL_HOME}/tt_metal/api
    ${TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    ${TT_METAL_HOME}/tt_metal/hostdevcommon/api
    ${TT_METAL_HOME}/tt_metal/third_party/tracy/public
    ${TT_METAL_HOME}/tt_stl
)

# Now wrap all the headers and .so files nicely into one target
if(NOT TARGET Metalium::TTNN)
    # Prefer build_Release if present, otherwise fall back to build
    if(EXISTS "${TT_METAL_HOME}/build_Release/lib")
        set(METALIUM_LIB_PATH "${TT_METAL_HOME}/build_Release/lib")
    else()
        set(METALIUM_LIB_PATH "${TT_METAL_HOME}/build/lib")
    endif()
    find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    find_library(DEVICE_LIBRARY NAMES "device" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    find_library(TTNN_LIBRARY NAMES "_ttnn.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)

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
        message(STATUS "Successfully found _ttnn.so at ${TTNN_LIBRARY}")
    else()
        message(FATAL_ERROR "_ttnn.so not found in ${METALIUM_LIB_PATH}")
    endif()

    # Also expose TT::STL/other deps via tt-metal exported config. Ensure its prefix
    # (which contains lib/cmake/{fmt,spdlog,...}) is on CMAKE_PREFIX_PATH so find_dependency works.
    if(EXISTS "${TT_METAL_HOME}/tt_stl/CMakeLists.txt")
        if(EXISTS "${TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            set(_TT_METAL_PREFIX "${TT_METAL_HOME}/build_Release")
            set(_TT_METAL_MODULE_DIR "${_TT_METAL_PREFIX}/lib/cmake/tt-metalium")
        elseif(EXISTS "${TT_METAL_HOME}/build/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            set(_TT_METAL_PREFIX "${TT_METAL_HOME}/build")
            set(_TT_METAL_MODULE_DIR "${_TT_METAL_PREFIX}/lib/cmake/tt-metalium")
        endif()

        if(DEFINED _TT_METAL_PREFIX)
            # Ensure dependencies like fmt/spdlog are discoverable
            list(PREPEND CMAKE_PREFIX_PATH "${_TT_METAL_PREFIX}")
            # Force using the exported config under lib/cmake to avoid the root-level shim
            find_package(tt-metalium CONFIG REQUIRED PATHS "${_TT_METAL_MODULE_DIR}" NO_DEFAULT_PATH)
        endif()
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()
