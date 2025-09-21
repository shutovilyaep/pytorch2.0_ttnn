#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

namespace tt_eager::ops::binary {
// Signatures matching PyTorch schemas (alpha included)
at::Tensor ttnn_add_tensor(const at::Tensor& input, const at::Tensor& other, const c10::Scalar& alpha);
at::Tensor& ttnn_add_out(const at::Tensor& input, const at::Tensor& other, const c10::Scalar& alpha, at::Tensor& out);

at::Tensor ttnn_sub_tensor(const at::Tensor& input, const at::Tensor& other, const c10::Scalar& alpha);
at::Tensor& ttnn_sub_out(const at::Tensor& input, const at::Tensor& other, const c10::Scalar& alpha, at::Tensor& out);
}  // namespace tt_eager::ops::binary
