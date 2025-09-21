#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::binary {

// Wrapper aliases for TTNN binary operations

// add/sub: aten semantics support alpha -> use b-scaled wrapper
using ttnn_add = tt_eager::ext::binary_b_scaled_wrapper<ttnn::add>;
using ttnn_sub = tt_eager::ext::binary_b_scaled_wrapper<ttnn::subtract>;

// arithmetic
using ttnn_multiply = tt_eager::ext::binary_wrapper<ttnn::multiply>;
using ttnn_divide = tt_eager::ext::binary_wrapper<ttnn::divide>;
using ttnn_ldexp = tt_eager::ext::binary_wrapper<ttnn::ldexp>;
using ttnn_logaddexp = tt_eager::ext::binary_wrapper<ttnn::logaddexp>;
using ttnn_logaddexp2 = tt_eager::ext::binary_wrapper<ttnn::logaddexp2>;
using ttnn_squared_difference = tt_eager::ext::binary_wrapper<ttnn::squared_difference>;
using ttnn_logical_right_shift = tt_eager::ext::binary_wrapper<ttnn::logical_right_shift>;
using ttnn_xlogy = tt_eager::ext::binary_wrapper<ttnn::xlogy>;

// logical
using ttnn_logical_and = tt_eager::ext::binary_wrapper<ttnn::logical_and>;
using ttnn_logical_or = tt_eager::ext::binary_wrapper<ttnn::logical_or>;
using ttnn_logical_xor = tt_eager::ext::binary_wrapper<ttnn::logical_xor>;

// relational
using ttnn_eq = tt_eager::ext::binary_wrapper<ttnn::eq>;
using ttnn_ne = tt_eager::ext::binary_wrapper<ttnn::ne>;
using ttnn_ge = tt_eager::ext::binary_wrapper<ttnn::ge>;
using ttnn_gt = tt_eager::ext::binary_wrapper<ttnn::gt>;
using ttnn_le = tt_eager::ext::binary_wrapper<ttnn::le>;
using ttnn_lt = tt_eager::ext::binary_wrapper<ttnn::lt>;

}  // namespace tt_eager::ops::binary


