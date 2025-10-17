import os
import sys
import subprocess
import importlib.util
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
import sysconfig


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=".", cmake_args=None, **kwargs):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)
        self.cmake_args = cmake_args or []


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        build_dir = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_dir, exist_ok=True)

        # Configure CMake
        try:
            import torch  # Lazy import to avoid PEP 517 import-time failures
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required to build this extension. Please install torch before building."
            ) from exc
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DTORCH_INSTALL_PREFIX={sysconfig.get_paths()['purelib']}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))}",
            f"-DOUTPUT_NAME={os.path.basename(self.get_ext_fullpath(ext.name))}",
            f"-G",
            f"Ninja",
        ]
        extra_cmake_flags = os.environ.get("CMAKE_FLAGS", "")
        # Support semicolon-separated flags from environment
        extra_cmake_flags = [f for f in extra_cmake_flags.split(";") if f]

        # Load ABI flags helper from local file without relying on import-time package layout
        torch_cxx_flags = []
        try:
            util_path = os.path.join(os.path.dirname(__file__), "utils", "get_torch_abi_flags.py")
            spec = importlib.util.spec_from_file_location("_ttnn_get_torch_abi_flags", util_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                torch_cxx_flags = module.get_torch_abi_related_compiler_flags()
        except Exception as _exc:
            # Fall back silently if helper is unavailable; build may still succeed
            torch_cxx_flags = []
        if torch_cxx_flags:
            flags_str = " ".join(torch_cxx_flags)
            extra_cmake_flags.append(f"-DCMAKE_CXX_FLAGS={flags_str}")

        # Propagate TT_METAL_HOME to cmake if provided
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if tt_metal_home:
            cmake_args.append(f"-DTT_METAL_HOME={tt_metal_home}")

        if extra_cmake_flags:
            cmake_args.extend(extra_cmake_flags)

        cmake_args.extend(ext.cmake_args)

        # Build the extension
        subprocess.check_call(["cmake", ext.source_dir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)

        # Copy the extension to the correct location
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)


setup(
    name="torch_ttnn_cpp_extension",
    version="0.1.0",
    packages=find_namespace_packages(include=["torch_ttnn*"]),
    ext_modules=[
        CMakeExtension(
            name="ttnn_device_extension",
            source_dir=".",
            cmake_args=[],
        ),
    ],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
