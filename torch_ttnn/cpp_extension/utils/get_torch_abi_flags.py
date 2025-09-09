import torch
import re


def get_torch_abi_related_compiler_flags():
    """Extract compiler flags relaed to abi from torch.__config__.show()"""
    config_str = torch.__config__.show()

    # Extract C++ flags
    cxx_flags = []

    # Look for ABI flag, critical for compatibility
    abi_match = re.search(r"-D_GLIBCXX_USE_CXX11_ABI=(\d)", config_str)
    if abi_match:
        cxx_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={abi_match.group(1)}")

    fabi_match = re.search(r"-fabi-version=(\d+)", config_str)
    # -fabi-version is GCC-specific and breaks Clang; skip if using Clang
    import os
    def _is_clang(val: str) -> bool:
        v = val or ""
        v = v.strip().lower()
        return "clang" in v
    use_clang = _is_clang(os.environ.get("CXX")) or _is_clang(os.environ.get("CC"))
    if fabi_match and not use_clang:
        cxx_flags.append(f"-fabi-version={fabi_match.group(1)}")

    return cxx_flags


if __name__ == "__main__":
    flags = get_torch_abi_related_compiler_flags()
    for flag in flags:
        print(flag, end=" ")
