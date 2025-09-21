#pragma once

#include <ATen/core/Tensor.h>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::unary {

// Simple unary ops (no extra parameters)
using ttnn_abs = tt_eager::ext::unary_wrapper<ttnn::abs>;
using ttnn_neg = tt_eager::ext::unary_wrapper<ttnn::neg>;
using ttnn_reciprocal = tt_eager::ext::unary_wrapper<ttnn::reciprocal>;
using ttnn_sqrt = tt_eager::ext::unary_wrapper<ttnn::sqrt>;
using ttnn_rsqrt = tt_eager::ext::unary_wrapper<ttnn::rsqrt>;
using ttnn_square = tt_eager::ext::unary_wrapper<ttnn::square>;
using ttnn_sin = tt_eager::ext::unary_wrapper<ttnn::sin>;
using ttnn_cos = tt_eager::ext::unary_wrapper<ttnn::cos>;
using ttnn_tan = tt_eager::ext::unary_wrapper<ttnn::tan>;
using ttnn_sinh = tt_eager::ext::unary_wrapper<ttnn::sinh>;
using ttnn_cosh = tt_eager::ext::unary_wrapper<ttnn::cosh>;
using ttnn_tanh = tt_eager::ext::unary_wrapper<ttnn::tanh>;
using ttnn_floor = tt_eager::ext::unary_wrapper<ttnn::floor>;
using ttnn_ceil = tt_eager::ext::unary_wrapper<ttnn::ceil>;
using ttnn_trunc = tt_eager::ext::unary_wrapper<ttnn::trunc>;
using ttnn_frac = tt_eager::ext::unary_wrapper<ttnn::frac>;
using ttnn_bitwise_not = tt_eager::ext::unary_wrapper<ttnn::bitwise_not>;
using ttnn_logical_not = tt_eager::ext::unary_wrapper<ttnn::logical_not>;
using ttnn_sign = tt_eager::ext::unary_wrapper<ttnn::sign>;
using ttnn_signbit = tt_eager::ext::unary_wrapper<ttnn::signbit>;
using ttnn_i0 = tt_eager::ext::unary_wrapper<ttnn::i0>;
using ttnn_erf = tt_eager::ext::unary_wrapper<ttnn::erf>;
using ttnn_erfc = tt_eager::ext::unary_wrapper<ttnn::erfc>;
using ttnn_erfinv = tt_eager::ext::unary_wrapper<ttnn::erfinv>;
using ttnn_exp = tt_eager::ext::unary_wrapper<ttnn::exp>;
using ttnn_log = tt_eager::ext::unary_wrapper<ttnn::log>;
using ttnn_log10 = tt_eager::ext::unary_wrapper<ttnn::log10>;
using ttnn_log2 = tt_eager::ext::unary_wrapper<ttnn::log2>;
using ttnn_log1p = tt_eager::ext::unary_wrapper<ttnn::log1p>;

}  // namespace tt_eager::ops::unary
