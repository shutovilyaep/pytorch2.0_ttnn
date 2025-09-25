import torch
from torch.overrides import TorchFunctionMode
import logging

import ttnn_device_extension as ttnn_module

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
        if "device" in kwargs and isinstance(kwargs["device"], str) and "ttnn" in kwargs["device"]:
            # Device lifecycle is handled by TTNN/TT-Metal mesh APIs; do not open here.
            # Allow torch to receive the PrivateUse1 device string as-is.
            # Users should obtain a proper device via higher-level APIs (e.g., tests fixture).
            pass
        with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)
