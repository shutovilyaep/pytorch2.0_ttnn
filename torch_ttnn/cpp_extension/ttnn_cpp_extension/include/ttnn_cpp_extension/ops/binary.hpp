#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::binary {

// Kernel aliases for aten::add / aten::sub with scalar alpha handling
using ttnn_add = tt_eager::ext::binary_with_scalar_kernel<ttnn::add>;
using ttnn_sub = tt_eager::ext::binary_with_scalar_kernel<ttnn::subtract>;

}  // namespace tt_eager::ops::binary


