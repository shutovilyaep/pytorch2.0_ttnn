cmake_minimum_required(VERSION 3.18)

# If no toolchain file is provided, try to select a sensible default early,
# before the first project()/compiler detection happens.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "")
    if(DEFINED ENV{TT_METAL_HOME} AND EXISTS "$ENV{TT_METAL_HOME}/cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{TT_METAL_HOME}/cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake" CACHE FILEPATH "Toolchain file" FORCE)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/x86_64-linux-torch-toolchain.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/x86_64-linux-torch-toolchain.cmake" CACHE FILEPATH "Toolchain file" FORCE)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)

project(ttnn_device_extension CXX)

# Options
option(BUILD_EXAMPLES "Build examples" OFF)

# Set up Python and Torch paths from environment if available
if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    set(TORCH_INSTALL_PREFIX $ENV{TORCH_INSTALL_PREFIX})
endif()

if(DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()

if(DEFINED ENV{CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $ENV{CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

if(DEFINED ENV{OUTPUT_NAME})
    set(OUTPUT_NAME $ENV{OUTPUT_NAME})
endif()

# Propagate TT_METAL_HOME to cmake if provided
if(DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_HOME $ENV{TT_METAL_HOME})
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(utils)

# Prepare tt-metal build
add_subdirectory(third-party)

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

# Detect Torch ABI flags after Torch is found
if(DEFINED ENV{TORCH_ABI_FLAGS})
    set(TORCH_ABI_FLAGS $ENV{TORCH_ABI_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_ABI_FLAGS}")
else()
    # Try to detect from Python/Torch config
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__config__.show())"
        OUTPUT_VARIABLE TORCH_CONFIG
        ERROR_QUIET
    )
    if(TORCH_CONFIG MATCHES "-D_GLIBCXX_USE_CXX11_ABI=([01])")
        set(TORCH_ABI_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_MATCH_1}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_ABI_FLAGS}")
        message(STATUS "Detected Torch ABI flag: ${TORCH_ABI_FLAGS}")
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

add_library(ttnn_device_extension SHARED ${TTNN_CPP_EXTENSION_SOURCES})
target_compile_options(ttnn_device_extension PRIVATE
    -DFMT_HEADER_ONLY
    -DTORCH_EXTENSION_NAME=ttnn_device_extension
    -DTORCH_API_INCLUDE_EXTENSION_H
)
target_compile_definitions(ttnn_device_extension PRIVATE NTEST)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(ttnn_device_extension PRIVATE DISABLE_NAMESPACE_STATIC_ASSERT)
endif()
target_include_directories(ttnn_device_extension PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_cpp_extension/include
    ${PYTHON_INCLUDE_DIRS}
)

# Needed for torch_python
target_link_directories(ttnn_device_extension PRIVATE "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(ttnn_device_extension PUBLIC
    TT::Metalium
    TTNN::CPP
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
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

if (BUILD_EXAMPLES)
    message(STATUS "Building examples")
    add_subdirectory(ttnn_cpp_extension/examples)
endif()

# Install the extension library so it's included in the wheel
install(TARGETS ttnn_device_extension
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
)

# Always install tt-metal shared libraries if they exist (even with static linking)
# This ensures runtime dependencies are available via INSTALL_RPATH "$ORIGIN"
# Libraries are copied from tt-metal build directory to the package directory
if(TARGET tt_metal)
    get_target_property(tt_metal_type tt_metal TYPE)
    if(tt_metal_type STREQUAL "SHARED_LIBRARY")
        install(TARGETS tt_metal
                LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
        )
    else()
        # For static builds, copy shared libraries from tt-metal build directory if they exist
        # This handles cases where tt-metal was built with shared libs but extension links statically
        # Check both build_Release/lib and build/lib directories
        set(TT_METAL_LIB_DIRS
            "${TT_METAL_HOME}/build_Release/lib"
            "${TT_METAL_HOME}/build/lib"
            "${TT_METAL_HOME}/build_Release/tt_stl"
            "${TT_METAL_HOME}/build/tt_stl"
        )
        
        foreach(LIB_DIR ${TT_METAL_LIB_DIRS})
            if(EXISTS "${LIB_DIR}/libtt_metal.so")
                install(FILES "${LIB_DIR}/libtt_metal.so"
                        DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                        OPTIONAL
                )
                message(STATUS "Will copy libtt_metal.so from ${LIB_DIR}")
            endif()
            if(EXISTS "${LIB_DIR}/libtt_stl.so")
                install(FILES "${LIB_DIR}/libtt_stl.so"
                        DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                        OPTIONAL
                )
                message(STATUS "Will copy libtt_stl.so from ${LIB_DIR}")
            endif()
        endforeach()
    endif()
endif()

# Install ttnncpp library if it's a shared library
if(TARGET ttnncpp)
    get_target_property(ttnncpp_type ttnncpp TYPE)
    if(ttnncpp_type STREQUAL "SHARED_LIBRARY")
        install(TARGETS ttnncpp
                LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
                RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/torch_ttnn_cpp_extension"
        )
    endif()
endif()