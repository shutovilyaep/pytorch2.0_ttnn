# Some flags for tt-metal -- ccache + unity build + make sure that cpm cache stays in tt-metal dir
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME environment variable is not set")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# TODO: figure out the way to fetch those from ttnn
# Glob to find needed includes from cpm cache
# Select single include dir per third-party to avoid version mixing
file(GLOB _REFLECT_DIRS "${TT_METAL_HOME}/.cpmcache/reflect/*/")
list(SORT _REFLECT_DIRS)
list(REVERSE _REFLECT_DIRS)
list(GET _REFLECT_DIRS 0 REFLECT_INCLUDE_DIR)

file(GLOB _FMT_DIRS "${TT_METAL_HOME}/.cpmcache/fmt/*/include")
list(SORT _FMT_DIRS)
list(REVERSE _FMT_DIRS)
list(GET _FMT_DIRS 0 FMT_INCLUDE_DIR)

file(GLOB _MAGIC_ENUM_DIRS "${TT_METAL_HOME}/.cpmcache/magic_enum/*/include")
list(SORT _MAGIC_ENUM_DIRS)
list(REVERSE _MAGIC_ENUM_DIRS)
list(GET _MAGIC_ENUM_DIRS 0 MAGIC_ENUM_INCLUDE_DIR)

file(GLOB _NLOHMANN_DIRS "${TT_METAL_HOME}/.cpmcache/nlohmann_json/*/include")
list(SORT _NLOHMANN_DIRS)
list(REVERSE _NLOHMANN_DIRS)
list(GET _NLOHMANN_DIRS 0 NLOHMANN_INCLUDE_DIR)

# Boost: find the actual include dir containing boost/core/span.hpp (avoid tools/* includes)
file(GLOB_RECURSE _BOOST_SPAN_HEADERS "${TT_METAL_HOME}/.cpmcache/boost/**/include/boost/core/span.hpp")
list(SORT _BOOST_SPAN_HEADERS)
list(REVERSE _BOOST_SPAN_HEADERS)
list(LENGTH _BOOST_SPAN_HEADERS _BOOST_SPAN_LEN)
if(_BOOST_SPAN_LEN GREATER 0)
    list(GET _BOOST_SPAN_HEADERS 0 _BOOST_SPAN_HEADER)
    get_filename_component(_BOOST_INCLUDE_DIR ${_BOOST_SPAN_HEADER} DIRECTORY) # .../include/boost/core
    get_filename_component(_BOOST_INCLUDE_DIR ${_BOOST_INCLUDE_DIR} DIRECTORY) # .../include/boost
    get_filename_component(BOOST_INCLUDE_DIR ${_BOOST_INCLUDE_DIR} DIRECTORY)  # .../include
else()
    # Fallback: best-effort include dir
    file(GLOB _BOOST_DIRS "${TT_METAL_HOME}/.cpmcache/boost/*/*/*/include")
    list(SORT _BOOST_DIRS)
    list(REVERSE _BOOST_DIRS)
    list(GET _BOOST_DIRS 0 BOOST_INCLUDE_DIR)
endif()

file(GLOB _TT_LOGGER_DIRS "${TT_METAL_HOME}/.cpmcache/tt-logger/*/include")
list(SORT _TT_LOGGER_DIRS)
list(REVERSE _TT_LOGGER_DIRS)
list(GET _TT_LOGGER_DIRS 0 TT_LOGGER_INCLUDE_DIR)

file(GLOB _SPDLOG_DIRS "${TT_METAL_HOME}/.cpmcache/spdlog/*/include")
list(SORT _SPDLOG_DIRS)
list(REVERSE _SPDLOG_DIRS)
list(GET _SPDLOG_DIRS 0 SPDLOG_INCLUDE_DIR)

set(TTNN_THIRD_PARTY_INCLUDE_DIRS
    ${REFLECT_INCLUDE_DIR}
    ${FMT_INCLUDE_DIR}
    ${MAGIC_ENUM_INCLUDE_DIR}
    ${NLOHMANN_INCLUDE_DIR}
    ${BOOST_INCLUDE_DIR}
    ${TT_LOGGER_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
)

set(TTNN_INCLUDE_DIRS
    ${TT_METAL_HOME}/ttnn/cpp
    ${TT_METAL_HOME}/ttnn
    ${TT_METAL_HOME}/ttnn/api
    ${TT_METAL_HOME}/tt_metal/api
    ${TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    ${TT_METAL_HOME}/tt_metal/hostdevcommon/api
    ${TT_METAL_HOME}/tt_metal/third_party/tracy/public
    ${TT_METAL_HOME}/tt_stl
    ${TTNN_THIRD_PARTY_INCLUDE_DIRS}
)

# Now wrap all the headers and .so files nicely into one target
if(NOT TARGET Metalium::TTNN)
    set(METALIUM_LIB_PATH "${TT_METAL_HOME}/build/lib")
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
else()
    message(STATUS "Metalium targets already exists")
endif()
