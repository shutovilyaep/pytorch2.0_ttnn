
#include <tt-metalium/assert.hpp>
#include <c10/core/Device.h>

#include "ttnn_cpp_extension/core/TtnnGuard.hpp"
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

#include "ttnn_cpp_extension/utils/device.hpp"
#include "ttnn_cpp_extension/utils/extension_utils.hpp"

// This function can be used when the TTNN device is initialized separately,
// for example, `device = ttnn.open_mesh_device(MeshShape(1,1))`. Pass that
// device object to this function so that the cpp extension can use it.
c10::Device as_torch_device(std::shared_ptr<ttnn::MeshDevice> ttnn_device) {
    LOGGING("");
    // TODO: Lacks a proper mesh support. We need to have mapping (shape, offset) -> index.
    // It's quiet difficult to do since c10::DeviceIndex is int8
    auto index = ttnn_device->get_device(0, 0)->id();
    auto device = c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(index));
    if (TtnnGuard::ttnn_device == nullptr) {
        // Keep a raw pointer view inside the guard while preserving ownership via shared_ptr
        static std::shared_ptr<ttnn::MeshDevice> keeper;
        keeper = std::move(ttnn_device);
        TtnnGuard::ttnn_device = keeper.get();
    }
    return device;
}

// Removed open/close helpers: Mesh device lifecycle managed by TT-Metal/TTNN

// Get the underlying TTNN tensor from a Torch tensor
ttnn::Tensor get_ttnn_tensor(at::Tensor& tensor) {
    LOGGING("");
    at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(tensor.unsafeGetTensorImpl());
    auto ttnn_tensor = tensor_impl->get_ttnn_tensor();
    return ttnn_tensor;
}
