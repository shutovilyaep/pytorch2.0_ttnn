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

    # Also expose TT::STL if tt_stl is installed in this tree
    if(EXISTS "${TT_METAL_HOME}/tt_stl/CMakeLists.txt")
        # If tt-metal exported config is present, include it to define TT::STL
        if(EXISTS "${TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build_Release")
            find_package(tt-metalium CONFIG REQUIRED PATHS "${TT_METAL_HOME}/build_Release" NO_DEFAULT_PATH)
        elseif(EXISTS "${TT_METAL_HOME}/build/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build")
            find_package(tt-metalium CONFIG REQUIRED PATHS "${TT_METAL_HOME}/build" NO_DEFAULT_PATH)
        endif()
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()
