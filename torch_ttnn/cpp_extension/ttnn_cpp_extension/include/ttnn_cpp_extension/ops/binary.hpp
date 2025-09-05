#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

namespace tt_eager::ops::binary {
// Signatures matching registrations in open_registration_extension.cpp for add/sub Tensor and out variants
at::Tensor ttnn_add_tensor(const at::Tensor& input, const at::Tensor& other);
at::Tensor& ttnn_add_out(const at::Tensor& input, const at::Tensor& other, at::Tensor& out);

at::Tensor ttnn_sub_tensor(const at::Tensor& input, const at::Tensor& other);
at::Tensor& ttnn_sub_out(const at::Tensor& input, const at::Tensor& other, at::Tensor& out);
}  // namespace tt_eager::ops::binary
