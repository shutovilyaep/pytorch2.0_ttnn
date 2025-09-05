#include <c10/util/Optional.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"

#include "ttnn_cpp_extension/ops/binary.hpp"
#include "ttnn_cpp_extension/ops/creation.hpp"

#include "ttnn_cpp_extension/utils/extension_utils.hpp"
#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::binary {
    // TODO: not required to write ttnn_add_tensor, ttnn_add_out - should be possible to pass these directly in torch_ttnn/cpp_extension/ttnn_cpp_extension/src/open_registration_extension.cpp
    using add_k = tt_eager::ext::binary_kernel<ttnn::add>;

    inline at::Tensor ttnn_add_tensor(const at::Tensor& a, const at::Tensor& b) {
        return add_k::func(a, b);
    }

    inline at::Tensor& ttnn_add_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        return add_k::func_out(a, b, out);
    }

    using sub_k = tt_eager::ext::binary_kernel<ttnn::subtract>;

    inline at::Tensor ttnn_sub_tensor(const at::Tensor& a, const at::Tensor& b) {
        return sub_k::func(a, b);
    }

    inline at::Tensor& ttnn_sub_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        return sub_k::func_out(a, b, out);
    }

}  // namespace tt_eager::ops::binary
