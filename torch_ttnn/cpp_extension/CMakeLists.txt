cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)

project(ttnn_device_extension CXX)

# Options
option(ENABLE_LOCAL_TT_METAL_BUILD "Enable local TT_Metal build" OFF)
option(ENABLE_SUBMODULE_TT_METAL_BUILD "Enable submodule TT_Metal build" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/tt-metal/cmake")
include(utils)
include(CPM)

# Prepare tt-metal build
# Если включен сбор подмодуля, собираем tt-metal как часть проекта.
# Иначе используем предсобранные артефакты через find_prebuilt_ttnn.
if (ENABLE_SUBMODULE_TT_METAL_BUILD)
    add_subdirectory(third-party)
else()
    include(find_prebuilt_ttnn)
endif()

check_ubuntu_version_at_least(24 IS_UBUNTU_GE_24)
message(STATUS "Ubuntu version > 24: ${IS_UBUNTU_GE_24}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# NOTE: ttnn reauires to downrade python version to 3.10, which break find_package
# TODO: Remove this once ttnn is updated to support python 3.12+
if (IS_UBUNTU_GE_24)
    include(find_python_workaround_ubuntu24)
else()
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(PythonLibs COMPONENTS Interpreter Development REQUIRED)
endif()

find_package(Torch REQUIRED)

set(TTNN_CPP_EXTENSION_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/open_registration_extension.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/core/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/core/TtnnCustomAllocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/core/TtnnGuard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/core/TtnnTensorImpl.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/ops/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/ops/creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/ops/unary.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/utils/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/src/utils/vector_utils.cpp

)

add_library(ttnn_device_extension SHARED ${TTNN_CPP_EXTENSION_SOURCES})
target_compile_options(ttnn_device_extension PRIVATE
    -DTORCH_EXTENSION_NAME=ttnn_device_extension
    -DTORCH_API_INCLUDE_EXTENSION_H
)
target_compile_definitions(ttnn_device_extension PRIVATE NTEST)
if(DEFINED TTNN_EXT_CXX11_ABI)
    # Pin ABI used by the extension to match torch/tt-metal choice without forcing GCC-only flags
    target_compile_definitions(ttnn_device_extension PRIVATE _GLIBCXX_USE_CXX11_ABI=${TTNN_EXT_CXX11_ABI})
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(ttnn_device_extension PRIVATE DISABLE_NAMESPACE_STATIC_ASSERT)
endif()
target_include_directories(ttnn_device_extension PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/include
    ${PYTHON_INCLUDE_DIRS}
)


# Needed for torch_python
target_link_directories(ttnn_device_extension PRIVATE "${TORCH_INSTALL_PREFIX}/lib")

# Bring Boost headers via CPM (match tt-metal third_party integration)
# Requires CPM to be available from included cmake/utils
CPMAddPackage(
    NAME Boost
    VERSION 1.86.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
    URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
    OPTIONS
        "BOOST_ENABLE_CMAKE ON"
        "BOOST_SKIP_INSTALL_RULES ON"
        "BUILD_SHARED_LIBS OFF"
        "BOOST_INCLUDE_LIBRARIES core\;container\;smart_ptr\;interprocess\;asio\;lockfree"
    FIND_PACKAGE_ARGUMENTS "CONFIG REQUIRED"
)

if(TARGET Boost::headers)
    target_link_libraries(ttnn_device_extension PRIVATE Boost::headers)
endif()
if(TARGET Boost::core)
    target_link_libraries(ttnn_device_extension PRIVATE Boost::core)
endif()

# Reflect and others (header-only) in case sources include headers directly
CPMAddPackage(NAME reflect GITHUB_REPOSITORY boost-ext/reflect GIT_TAG v1.2.6)
if(reflect_ADDED AND NOT TARGET Reflect::Reflect)
    add_library(reflect INTERFACE)
    add_library(Reflect::Reflect ALIAS reflect)
    target_include_directories(reflect SYSTEM INTERFACE "$<BUILD_INTERFACE:${reflect_SOURCE_DIR}>")
endif()
CPMAddPackage(NAME magic_enum GITHUB_REPOSITORY Neargye/magic_enum GIT_TAG v0.9.7)
if(TARGET magic_enum)
    target_link_libraries(ttnn_device_extension PRIVATE magic_enum::magic_enum)
endif()
CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json GIT_TAG v3.11.3)
if(TARGET nlohmann_json)
    target_link_libraries(ttnn_device_extension PRIVATE nlohmann_json::nlohmann_json)
endif()

# Enchantum (match tt-metal/main/third_party/CMakeLists.txt)
CPMAddPackage(
    NAME enchantum
    GIT_REPOSITORY https://github.com/ZXShady/enchantum.git
    GIT_TAG 8ca5b0eb7e7ebe0252e5bc6915083f1dd1b8294e
    OPTIONS "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)
if(TARGET enchantum::enchantum)
    target_link_libraries(ttnn_device_extension PRIVATE enchantum::enchantum)
else()
    # Fallback include dirs if target not exported
    if(DEFINED enchantum_SOURCE_DIR)
        target_include_directories(ttnn_device_extension SYSTEM PRIVATE ${enchantum_SOURCE_DIR}/include)
    endif()
endif()

# tt-logger headers (for assert.hpp include)
CPMAddPackage(
    NAME tt-logger
    GITHUB_REPOSITORY tenstorrent/tt-logger
    VERSION 1.1.3
    OPTIONS
        "TT_LOGGER_INSTALL OFF"
        "TT_LOGGER_BUILD_TESTING OFF"
)
if(TARGET tt-logger::tt-logger)
    target_link_libraries(ttnn_device_extension PRIVATE tt-logger::tt-logger)
elseif(DEFINED tt_logger_SOURCE_DIR)
    target_include_directories(ttnn_device_extension SYSTEM PRIVATE ${tt_logger_SOURCE_DIR}/include)
endif()

# Bring fmt (header-only) matching tt-metal version
CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.1.4)
if(TARGET fmt::fmt-header-only)
    target_link_libraries(ttnn_device_extension PRIVATE fmt::fmt-header-only)
endif()

# Bring spdlog matching tt-metal version and use external fmt header-only
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME spdlog-dev)
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.15.2
    OPTIONS
        "SPDLOG_FMT_EXTERNAL_HO ON"
        "SPDLOG_INSTALL ON"
)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${DEFAULT_COMPONENT_NAME})
if(TARGET spdlog::spdlog_header_only)
    target_link_libraries(ttnn_device_extension PRIVATE spdlog::spdlog_header_only)
endif()

target_link_libraries(ttnn_device_extension PUBLIC
    Metalium::Metal
    Metalium::TTNN
    Metalium::TTNN_CPP
    fmt::fmt-header-only
    spdlog::spdlog_header_only
    Reflect::Reflect
)

target_link_libraries(ttnn_device_extension PUBLIC 
    ${TORCH_LIBRARIES} 
    ${PYTHON_LIBRARIES}
    torch_python
)

if(NOT DEFINED OUTPUT_NAME)
  set(OUTPUT_NAME "ttnn_device_extension")
endif()

set_target_properties(ttnn_device_extension PROPERTIES
    PREFIX ""
    SUFFIX ""
    OUTPUT_NAME ${OUTPUT_NAME}
    BUILD_RPATH "${TT_METAL_HOME}/build/lib;${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TT_METAL_HOME}/build/lib;${TORCH_INSTALL_PREFIX}/lib"
)

if (BUILD_EXAMPLES)
    message(STATUS "Building examples")
    add_subdirectory(ttnn_cpp_extension/examples)
endif()