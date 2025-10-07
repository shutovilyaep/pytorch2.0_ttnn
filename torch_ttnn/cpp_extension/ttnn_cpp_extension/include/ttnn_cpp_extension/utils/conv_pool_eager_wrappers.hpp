#pragma once

#include "ttnn_cpp_extension/utils/eager_common.hpp"

#include <ttnn/operations/conv/conv1d/conv1d.hpp>
#include <ttnn/operations/conv/conv2d/conv2d.hpp>
#include <ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp>
#include <ttnn/operations/experimental/conv3d/conv3d.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>
#include <ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp>

namespace tt_eager::ext {

// This header is the home for convolution and pooling eager wrappers.
// As part of the split, all conv/pool logic will live here,
// and the previous monolithic eager_wrap.hpp will be removed.
// Tuple helpers and output-dimension calculators will be defined here alongside wrappers.

} // namespace tt_eager::ext



