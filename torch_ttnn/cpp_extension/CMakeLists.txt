cmake_minimum_required(VERSION 3.23)

# If no toolchain file is provided, try to use the clang based toolchains that
# ship with tt-metal. Fall back to the copy committed with this project.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "")
    if(DEFINED ENV{TT_METAL_HOME})
        set(_tt_toolchain_candidates "x86_64-linux-clang-17-libstdcpp-toolchain.cmake" "x86_64-linux-clang-17-libcpp-toolchain.cmake")
        if(EXISTS "/etc/os-release")
            file(READ "/etc/os-release" _tt_os_release)
            if(_tt_os_release MATCHES "VERSION_ID=\"?20.04\"?")
                list(REVERSE _tt_toolchain_candidates)
            endif()
        endif()
        foreach(_tt_toolchain_name IN LISTS _tt_toolchain_candidates)
            set(_tt_toolchain "$ENV{TT_METAL_HOME}/cmake/${_tt_toolchain_name}")
            if(EXISTS "${_tt_toolchain}")
                set(CMAKE_TOOLCHAIN_FILE "${_tt_toolchain}" CACHE FILEPATH "Toolchain file" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "")
        set(_fallback_toolchain "${CMAKE_CURRENT_SOURCE_DIR}/cmake/x86_64-linux-torch-toolchain.cmake")
        if(EXISTS "${_fallback_toolchain}")
            set(CMAKE_TOOLCHAIN_FILE "${_fallback_toolchain}" CACHE FILEPATH "Toolchain file" FORCE)
        endif()
    endif()
endif()

project(ttnn_device_extension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(utils)

# Honour TT_METAL_HOME passed in through the environment when configuring the
# third-party build.
if(DEFINED ENV{TT_METAL_HOME} AND NOT DEFINED TT_METAL_HOME)
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH "Path to the tt-metal checkout")
endif()

set(VERSION_NUMERIC "" CACHE STRING "tt-metal version used to build the extension")
if(VERSION_NUMERIC STREQUAL "")
    if(DEFINED ENV{TT_METAL_VERSION} AND NOT "$ENV{TT_METAL_VERSION}" STREQUAL "")
        set(VERSION_NUMERIC "$ENV{TT_METAL_VERSION}" CACHE STRING "tt-metal version used to build the extension" FORCE)
    elseif(DEFINED TT_METAL_HOME AND EXISTS "${TT_METAL_HOME}/.git")
        execute_process(
            COMMAND git describe --abbrev=0 --tags
            WORKING_DIRECTORY "${TT_METAL_HOME}"
            OUTPUT_VARIABLE _tt_metal_tag
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT _tt_metal_tag STREQUAL "")
            string(REGEX REPLACE "^v" "" _tt_metal_version "${_tt_metal_tag}")
            set(VERSION_NUMERIC "${_tt_metal_version}" CACHE STRING "tt-metal version used to build the extension" FORCE)
        endif()
    endif()
endif()
if(VERSION_NUMERIC STREQUAL "")
    set(VERSION_NUMERIC "0.60.1" CACHE STRING "tt-metal version used to build the extension" FORCE)
endif()
message(STATUS "Using tt-metal VERSION_NUMERIC=${VERSION_NUMERIC}")

add_subdirectory(third-party)

check_ubuntu_version_at_least(24 IS_UBUNTU_GE_24)
message(STATUS "Ubuntu version >= 24: ${IS_UBUNTU_GE_24}")

if(IS_UBUNTU_GE_24)
    include(find_python_workaround_ubuntu24)
else()
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
    set(PYTHON_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python_LIBRARIES}")
endif()

# Ensure imported targets are available even when the Ubuntu 24 workaround ran.
if(NOT TARGET Python::Module)
    add_library(Python::Module UNKNOWN IMPORTED)
    set_target_properties(Python::Module PROPERTIES
        IMPORTED_LOCATION "${PYTHON_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}"
    )
endif()

if(NOT TARGET Python::Python)
    add_library(Python::Python UNKNOWN IMPORTED)
    set_target_properties(Python::Python PROPERTIES
        IMPORTED_LOCATION "${PYTHON_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}"
    )
endif()

if(PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT _torch_prefix STREQUAL "")
        string(REPLACE ":" ";" _torch_prefix_list "${_torch_prefix}")
        foreach(_prefix IN LISTS _torch_prefix_list)
            list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
        endforeach()
        list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE STRING "CMake search prefixes" FORCE)
    endif()
endif()

find_package(Torch REQUIRED CONFIG)

# Derive ABI flags from Torch if they are not explicitly provided.
set(ttnn_torch_abi_flags "")
if(DEFINED ENV{TORCH_ABI_FLAGS} AND NOT "$ENV{TORCH_ABI_FLAGS}" STREQUAL "")
    string(REPLACE " " ";" ttnn_torch_abi_flags "$ENV{TORCH_ABI_FLAGS}")
else()
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__config__.show())"
        OUTPUT_VARIABLE _torch_config
        ERROR_QUIET
    )
    if(_torch_config MATCHES "-D_GLIBCXX_USE_CXX11_ABI=([01])")
        set(ttnn_torch_abi_flags "-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_MATCH_1}")
        message(STATUS "Detected Torch ABI flag: ${ttnn_torch_abi_flags}")
    endif()
endif()

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

add_library(ttnn_device_extension MODULE ${TTNN_CPP_EXTENSION_SOURCES})
set_target_properties(ttnn_device_extension PROPERTIES
    CXX_STANDARD 20
    PREFIX ""
    SUFFIX ""
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

if(ttnn_torch_abi_flags)
    target_compile_options(ttnn_device_extension PRIVATE ${ttnn_torch_abi_flags})
endif()

target_compile_definitions(ttnn_device_extension PRIVATE
    FMT_HEADER_ONLY
    TORCH_EXTENSION_NAME=ttnn_device_extension
    TORCH_API_INCLUDE_EXTENSION_H
    NTEST
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(ttnn_device_extension PRIVATE DISABLE_NAMESPACE_STATIC_ASSERT)
endif()

target_include_directories(ttnn_device_extension PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/include
)

target_link_libraries(ttnn_device_extension
    PRIVATE
        TT::Metalium
        TTNN::CPP
        Python::Module
        Python::Python
)

if(TARGET Torch::Torch)
    target_link_libraries(ttnn_device_extension PRIVATE Torch::Torch)
endif()

if(TARGET Torch::Python)
    target_link_libraries(ttnn_device_extension PRIVATE Torch::Python)
elseif(TARGET torch_python)
    target_link_libraries(ttnn_device_extension PRIVATE torch_python)
endif()

set(_ttnn_output_name "ttnn_device_extension")
if(DEFINED OUTPUT_NAME AND NOT OUTPUT_NAME STREQUAL "")
    set(_ttnn_output_name "${OUTPUT_NAME}")
elseif(DEFINED ENV{OUTPUT_NAME} AND NOT "$ENV{OUTPUT_NAME}" STREQUAL "")
    set(_ttnn_output_name "$ENV{OUTPUT_NAME}")
endif()
set_target_properties(ttnn_device_extension PROPERTIES OUTPUT_NAME "${_ttnn_output_name}")

if(TTNN_BUILD_EXAMPLES)
    add_subdirectory(ttnn_cpp_extension/examples)
endif()

install(TARGETS ttnn_device_extension
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension")

# Always install tt-metal shared libraries if they exist. This ensures runtime
# dependencies are available via INSTALL_RPATH "$ORIGIN".
if(TARGET tt_metal)
    get_target_property(_tt_metal_type tt_metal TYPE)
    if(_tt_metal_type STREQUAL "SHARED_LIBRARY")
        install(TARGETS tt_metal
                LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension")
    else()
        set(_tt_metal_lib_dirs
            "${TT_METAL_HOME}/build_Release/lib"
            "${TT_METAL_HOME}/build/lib"
            "${TT_METAL_HOME}/build_Release/tt_stl"
            "${TT_METAL_HOME}/build/tt_stl"
        )
        foreach(_lib_dir IN LISTS _tt_metal_lib_dirs)
            if(EXISTS "${_lib_dir}/libtt_metal.so")
                install(FILES "${_lib_dir}/libtt_metal.so"
                        DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                        OPTIONAL)
                message(STATUS "Will copy libtt_metal.so from ${_lib_dir}")
            endif()
            if(EXISTS "${_lib_dir}/libtt_stl.so")
                install(FILES "${_lib_dir}/libtt_stl.so"
                        DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                        OPTIONAL)
                message(STATUS "Will copy libtt_stl.so from ${_lib_dir}")
            endif()
        endforeach()
    endif()
endif()

if(TARGET ttnncpp)
    get_target_property(_ttnncpp_type ttnncpp TYPE)
    if(_ttnncpp_type STREQUAL "SHARED_LIBRARY")
        install(TARGETS ttnncpp
                LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension")
    endif()
endif()

