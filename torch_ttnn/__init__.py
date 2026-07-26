# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# The interface of this backend

import sys
import platform

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def _check_ttnn_requirements():
    """Check if the environment meets ttnn requirements."""
    if sys.platform != "linux":
        raise ImportError("ttnn is only supported on Linux. " "Current platform: {}".format(platform.platform()))

    python_version = sys.version_info
    # 3.10 is the historical CI pin; 3.12 is used for host-matched eager packaging
    # when manylinux auditwheel wheels are unusable on the build host.
    if python_version.major != 3 or python_version.minor not in (10, 12):
        raise ImportError(
            "ttnn requires Python 3.10 or 3.12. "
            "Current Python version: {}.{}".format(python_version.major, python_version.minor)
        )


def _shim_ttnn_compat(ttnn_mod):
    """Fill APIs removed from newer ttnn that torch_ttnn still references."""
    if not hasattr(ttnn_mod, "format_output_tensor"):
        # Pin 8dfb324 dropped format_output_tensor; to_layout is the replacement.
        ttnn_mod.format_output_tensor = ttnn_mod.to_layout


try:
    _check_ttnn_requirements()
    import ttnn as _ttnn

    _shim_ttnn_compat(_ttnn)
except ImportError as e:
    if "ttnn" not in str(e):
        raise
    print(
        "\nError: ttnn package is not installed.\n"
        "\n"
        "For PyPI users, install with:\n"
        "  pip install torch-ttnn-shutov[pypi]\n"
        "\n"
        "For development builds, ttnn is built from source:\n"
        "  cd ${TT_METAL_HOME} && ./create_venv.sh\n"
        "\n"
        "Note: ttnn is only supported on Linux with Python 3.10 or 3.12.\n"
        "Current system: {}\n"
        "Current Python: {}.{}\n".format(platform.platform(), sys.version_info.major, sys.version_info.minor)
    )
    raise

from .backend import ttnn_backend as backend
from .backend import TorchTtnnOption
import torch as _torch

_torch._dynamo.backends.registry.register_backend(name="ttnn", compiler_fn=backend)

# To wrap the ttnn ops
from .passes.lowering import target_wrappers
