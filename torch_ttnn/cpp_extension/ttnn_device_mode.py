import torch
from torch.overrides import TorchFunctionMode
import logging
import sys
from pathlib import Path

# Import ttnn_device_extension module
# First try direct import (works for regular installs)
try:
    import ttnn_device_extension as ttnn_module
except ImportError:
    # For editable installs, the module is installed in site-packages/torch_ttnn_cpp_extension/ttnn_device_extension
    # but editable install overrides the package, so we need to load it manually
    import site
    import importlib.util
    from importlib.machinery import ExtensionFileLoader

    _imported = False

    for site_dir in site.getsitepackages():
        ext_dir = Path(site_dir) / "torch_ttnn_cpp_extension"
        ext_file = ext_dir / "ttnn_device_extension"

        if ext_file.exists():
            # Try direct import by adding to sys.path first
            if str(ext_dir) not in sys.path:
                sys.path.insert(0, str(ext_dir))
            try:
                import ttnn_device_extension as ttnn_module

                _imported = True
                break
            except ImportError:
                # Direct import failed, use ExtensionFileLoader to load the .so file
                try:
                    loader = ExtensionFileLoader("ttnn_device_extension", str(ext_file))
                    spec = importlib.util.spec_from_loader("ttnn_device_extension", loader)
                    if spec and spec.loader:
                        ttnn_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(ttnn_module)
                        _imported = True
                        break
                except Exception as e:
                    logging.debug(f"Failed to load via ExtensionFileLoader: {e}")
                    pass
                # Remove from path if import failed
                if str(ext_dir) in sys.path:
                    sys.path.remove(str(ext_dir))

    if not _imported:
        raise ImportError(
            "Could not import ttnn_device_extension. "
            "Please ensure the C++ extension is built and installed correctly. "
            f"Searched in: {[str(Path(d) / 'torch_ttnn_cpp_extension') for d in site.getsitepackages()]}"
        )

logging.info("Using pre-built ttnn_device_extension")

torch.utils.rename_privateuse1_backend("ttnn")


# The user will globally enable the below mode when calling this API
def enable_ttnn_device():
    m = TtnnDeviceMode()
    m.__enter__()
    # If you want the mode to never be disabled, then this function shouldn't return anything.
    return m


# This is a simple TorchFunctionMode class that:
# (a) Intercepts all torch.* calls
# (b) Checks for kwargs of the form `device="ttnn:i"`
# (c) Turns those into custom device objects: `device=ttnn_module.custom_device(i)`
# (d) Forwards the call along into pytorch.
class TtnnDeviceMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if "device" in kwargs and "ttnn" in kwargs["device"]:
            device_and_idx = kwargs["device"].split(":")
            if len(device_and_idx) == 1:
                # Case 1: No index specified
                kwargs["device"] = ttnn_module.open_torch_device()
            else:
                # Case 2: The user specified a device index.
                device_idx = int(device_and_idx[1])
                kwargs["device"] = ttnn_module.open_torch_device(device_idx)
        with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)
