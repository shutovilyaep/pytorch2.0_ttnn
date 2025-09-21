#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::binary {

// Wrapper aliases for aten::add / aten::sub with scalar alpha handling
using add = tt_eager::ext::binary_with_scalar_wrapper<ttnn::add>;
using sub = tt_eager::ext::binary_with_scalar_wrapper<ttnn::subtract>;

}  // namespace tt_eager::ops::binary


