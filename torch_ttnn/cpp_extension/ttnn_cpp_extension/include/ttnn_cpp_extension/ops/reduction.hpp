#pragma once

#include <ATen/core/Tensor.h>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::reduction {

using ttnn_sum = tt_eager::ext::reduction_wrapper<ttnn::sum>;
using ttnn_mean = tt_eager::ext::reduction_wrapper<ttnn::mean>;
using ttnn_max_reduce = tt_eager::ext::reduction_wrapper<ttnn::max>;
using ttnn_min_reduce = tt_eager::ext::reduction_wrapper<ttnn::min>;
using ttnn_std_reduce = tt_eager::ext::reduction_wrapper<ttnn::std>;
using ttnn_var_reduce = tt_eager::ext::reduction_wrapper<ttnn::var>;

}  // namespace tt_eager::ops::reduction


