#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <memory>

#include <ttnn/operations/core/core.hpp>

c10::Device as_torch_device(std::shared_ptr<ttnn::MeshDevice> ttnn_device);

// Get the underlying TTNN tensor from a Torch tensor
ttnn::Tensor get_ttnn_tensor(at::Tensor& tensor);
