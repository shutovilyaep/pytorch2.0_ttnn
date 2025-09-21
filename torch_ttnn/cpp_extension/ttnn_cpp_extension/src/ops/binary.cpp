#include <c10/util/Optional.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "ttnn_cpp_extension/ops/binary.hpp"
#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::binary {
// TODO: not required to write ttnn_add_tensor, ttnn_add_out - should be possible to pass these directly in
// torch_ttnn/cpp_extension/ttnn_cpp_extension/src/open_registration_extension.cpp
using add_k = tt_eager::ext::binary_with_scalar_kernel<ttnn::add>;

at::Tensor ttnn_add_tensor(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
    return add_k::func(a, b, alpha);
}

at::Tensor& ttnn_add_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
    return add_k::func_out(a, b, alpha, out);
}

using sub_k = tt_eager::ext::binary_with_scalar_kernel<ttnn::subtract>;

at::Tensor ttnn_sub_tensor(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
    return sub_k::func(a, b, alpha);
}

at::Tensor& ttnn_sub_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
    return sub_k::func_out(a, b, alpha, out);
}

}  // namespace tt_eager::ops::binary
