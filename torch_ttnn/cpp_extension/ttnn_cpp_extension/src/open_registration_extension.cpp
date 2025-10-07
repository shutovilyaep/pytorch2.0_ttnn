#include <ATen/native/DispatchStub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "ttnn_cpp_extension/utils/device.hpp"

#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"
#include "ttnn_cpp_extension/core/copy.hpp"

#include "ttnn_cpp_extension/ops/creation.hpp"

#include "ttnn_cpp_extension/utils/unary_eager_wrappers.hpp"
#include "ttnn_cpp_extension/utils/unary_eager_register.hpp"
#include "ttnn_cpp_extension/utils/binary_eager_wrappers.hpp"
#include "ttnn_cpp_extension/utils/random_eager_wrappers.hpp"
#include "ttnn_cpp_extension/utils/reduction_eager_wrappers.hpp"
#include "ttnn_cpp_extension/utils/conv_pool_eager_wrappers.hpp"

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/eltwise/complex_unary/complex_unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>
#include <ttnn/operations/eltwise/binary/binary_composite.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/bernoulli/bernoulli.hpp>
#include <ttnn/operations/uniform/uniform.hpp>
#include <ttnn/operations/moreh/moreh_dot/moreh_dot.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <string>

// Register custom allocator. Only used to create dummy Torch tensor object.
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &get_ttnn_custom_allocator());

namespace {
// Generic convolution dispatcher matching aten.convolution/overrideable
// Signature must match native_functions.yaml schema
static at::Tensor aten_convolution_dispatch(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups) {
    const int64_t dim = input.dim();
    TORCH_CHECK(dim == weight.dim(), "convolution: input and weight must have same rank");
    if (!transposed) {
        if (dim == 3) {
            return tt_eager::ext::conv1d_aten::invoke(input, weight, bias, stride, padding, dilation, groups);
        } else if (dim == 4) {
            return tt_eager::ext::conv2d_aten::invoke(input, weight, bias, stride, padding, dilation, groups);
        } else if (dim == 5) {
            return tt_eager::ext::conv3d_aten::invoke(input, weight, bias, stride, padding, dilation, groups);
        }
        TORCH_CHECK(false, "convolution: unsupported input dim=", dim, " (expected 3,4,5)");
    } else {
        TORCH_CHECK(dim != 3, "convolution: transposed 1D not yet supported on TTNN");
        if (dim == 4) {
            return tt_eager::ext::conv_transpose2d_aten::invoke(
                input, weight, bias, stride, padding, output_padding, groups, dilation);
        }
        TORCH_CHECK(false, "convolution: transposed dim=", dim, " not yet supported on TTNN");
    }
}

// Helper templates for concise unary registrations
// - register_unary_base_out_inplace: registers base, base.out, base_
// - register_unary_base_out:         registers base, base.out
// - register_unary_base_inplace:     registers base, base_
template <template <auto> class Wrapper, auto Op>
static inline void register_unary_base_out_inplace(torch::Library& m, const std::string& base) {
    const std::string out = base + ".out";
    const std::string inplace = base + "_";
    m.impl(base.c_str(), TORCH_FN(Wrapper<Op>::invoke));
    m.impl(out.c_str(), TORCH_FN(Wrapper<Op>::invoke_into));
    m.impl(inplace.c_str(), TORCH_FN(Wrapper<Op>::invoke_inplace));
}

template <template <auto> class Wrapper, auto Op>
static inline void register_unary_base_out(torch::Library& m, const std::string& base) {
    const std::string out = base + ".out";
    m.impl(base.c_str(), TORCH_FN(Wrapper<Op>::invoke));
    m.impl(out.c_str(), TORCH_FN(Wrapper<Op>::invoke_into));
}

template <template <auto> class Wrapper, auto Op>
static inline void register_unary_base_inplace(torch::Library& m, const std::string& base) {
    const std::string inplace = base + "_";
    m.impl(base.c_str(), TORCH_FN(Wrapper<Op>::invoke));
    m.impl(inplace.c_str(), TORCH_FN(Wrapper<Op>::invoke_inplace));
}

static inline void register_core_creation_and_copy(torch::Library& m) {
    // =========================
    // Core ops: creation and copy
    // =========================
    // From Pytorch's NamesRegistrations.cpp
    // schema: empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device?
    // device=None, bool? pin_memory=None) -> Tensor
    m.impl("aten::empty_strided", &tt_eager::ops::create::custom_empty_strided);
    // schema: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
    // bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    m.impl("empty.memory_format", &tt_eager::ops::create::custom_empty_memory_format);
    // schema: _copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
    m.impl("_copy_from", &ttnn_copy_from);
    // Pending TTNN core ops to register (from ttnn_ops_grouped.txt)
    // ttnn::to_dtype
    // ttnn::to_layout
    // ttnn::to_memory_config
}

static inline void register_binary_ops(torch::Library& m) {
    // =========================
    // Binary ops
    // =========================
    // schema: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
    m.impl("add.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke_into));
    // schema: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    m.impl("add.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke));
    // schema: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    m.impl("add.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::add>::invoke));
    // schema: add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    m.impl("add_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::add_>::invoke_inplace));
    // schema: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    m.impl("add_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke_inplace));
    // _add_relu.* = relu(add.Tensor with alpha=1)
    // Match aten schema: (Tensor, Tensor, Scalar alpha=1)
    using AddReluAlphaWrapper = tt_eager::ext::binary_tensor_tensor_alpha_then_unary<ttnn::addalpha, ttnn::relu>;
    // schema: _add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    m.impl("_add_relu.Tensor", TORCH_FN(AddReluAlphaWrapper::invoke));
    // schema: _add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
    m.impl("_add_relu.out", TORCH_FN(AddReluAlphaWrapper::invoke_into));
    // schema: _add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    m.impl("_add_relu_.Tensor", TORCH_FN(AddReluAlphaWrapper::invoke_inplace));

    // schema: sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
    m.impl("sub.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke_into));
    // schema: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    m.impl("sub.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke));
    // schema: sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    m.impl("sub.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::subtract>::invoke));
    // schema: sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    m.impl(
        "sub_.Scalar",
        TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::subtract_>::invoke_inplace));
    // schema: sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    m.impl("sub_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke_inplace));
    // rsub: reverse subtract
    // rsub.Tensor: rsub(self, other, alpha) = other - alpha*self
    // rsub.Scalar: rsub(self, other, alpha) = other - alpha*self
    // schema: rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    m.impl(
        "rsub.Tensor",
        TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha_swapped<ttnn::subalpha>::invoke));  // TODO: to check
    // schema: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    m.impl(
        "rsub.Scalar",
        TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::rsub>::invoke));  // TODO: to check

    // Arithmetic ops
    // schema: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("mul.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply>::invoke_into));
    // schema: mul.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("mul.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply>::invoke));
    // schema: mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("mul_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply_>::invoke_inplace));

    // schema: div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("div.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide>::invoke_into));
    // schema: div.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("div.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide>::invoke));
    // schema: div.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("div.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::divide>::invoke));
    // schema: div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
    m.impl("div_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::divide_>::invoke_inplace));
    // schema: div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("div_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide_>::invoke_inplace));

    // schema: floor_divide(Tensor self, Tensor other) -> Tensor
    m.impl("floor_divide", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke));
    // schema: floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("floor_divide.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::floor_div>::invoke));
    // schema: floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("floor_divide.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke_into));
    // schema: floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
    m.impl("floor_divide_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::floor_div>::invoke_inplace));
    // schema: floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("floor_divide_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke_inplace));
    // true_divide.Scalar
    // true_divide.out
    // true_divide_.Scalar
    // true_divide_.Tensor
    // (handled via divide) no direct ttnn::true_divide
    // schema: pow.Scalar(Scalar self, Tensor exponent) -> Tensor
    m.impl("pow.Scalar", TORCH_FN(tt_eager::ext::binary_scalar_tensor_as_tensor<ttnn::pow>::invoke));
    // schema: pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("pow.Scalar_out", TORCH_FN(tt_eager::ext::binary_scalar_tensor_as_tensor<ttnn::pow>::invoke_into));
    // schema: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
    m.impl("pow.Tensor_Scalar", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke));
    // schema: pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("pow.Tensor_Scalar_out", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke_into));
    // schema: pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
    m.impl("pow_.Scalar", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke_inplace));
    // schema: pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
    m.impl("pow_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::pow>::invoke_inplace));
    // schema: nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("nextafter.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::nextafter>::invoke_into));
    // schema: nextafter(Tensor self, Tensor other) -> Tensor
    m.impl("nextafter", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::nextafter>::invoke));
    // schema: dot(Tensor self, Tensor tensor) -> Tensor
    m.impl("dot", TORCH_FN(tt_eager::ext::binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke));
    // schema: dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("dot.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke_into));
    // schema: hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("hypot.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::hypot>::invoke_into));
    // schema: hypot(Tensor self, Tensor other) -> Tensor
    m.impl("hypot", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::hypot>::invoke));

    // schema: matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("matmul.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::matmul>::invoke_into));
    // schema: matmul(Tensor self, Tensor other) -> Tensor
    m.impl("matmul", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::matmul>::invoke));
    // mm
    // mm.out
    // mv
    // mv.out
    // bmm
    // bmm.out

    // Logical ops
    // schema: logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logical_and.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and>::invoke_into));
    // schema: logical_and(Tensor self, Tensor other) -> Tensor
    m.impl("logical_and", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and>::invoke));
    // schema: logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("logical_and_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and_>::invoke_inplace));

    // schema: logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logical_or.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or>::invoke_into));
    // schema: logical_or(Tensor self, Tensor other) -> Tensor
    m.impl("logical_or", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or>::invoke));
    // schema: logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("logical_or_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or_>::invoke_inplace));

    // schema: logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logical_xor.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor>::invoke_into));
    // schema: logical_xor(Tensor self, Tensor other) -> Tensor
    m.impl("logical_xor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor>::invoke));
    // schema: logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("logical_xor_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor_>::invoke_inplace));

    // Trigonometric binary ops
    // schema: atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("atan2.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke_into));
    // schema: atan2(Tensor self, Tensor other) -> Tensor
    m.impl("atan2", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke));
    // schema: atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
    m.impl("atan2_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke_inplace));

    // Relational ops (Tensor versions only)
    // schema: eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("eq.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::eq>::invoke_into));
    // schema: eq.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("eq.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::eq>::invoke));

    // schema: eq.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("eq.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::eq>::invoke));
    // schema: eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("eq.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::eq>::invoke_into));

    // schema: ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("ne.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ne>::invoke_into));
    // schema: ne.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("ne.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ne>::invoke));

    // schema: ne.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("ne.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ne>::invoke));
    // schema: ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("ne.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ne>::invoke_into));

    // schema: ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("ge.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ge>::invoke_into));
    // schema: ge.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("ge.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ge>::invoke));

    // schema: ge.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("ge.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ge>::invoke));
    // schema: ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("ge.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ge>::invoke_into));

    // schema: gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("gt.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::gt>::invoke_into));
    // schema: gt.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("gt.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::gt>::invoke));

    // schema: gt.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("gt.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::gt>::invoke));
    // schema: gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("gt.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::gt>::invoke_into));

    // schema: le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("le.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::le>::invoke_into));
    // schema: le.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("le.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::le>::invoke));

    // schema: le.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("le.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::le>::invoke));
    // schema: le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("le.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::le>::invoke_into));

    // schema: lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("lt.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::lt>::invoke_into));
    // schema: lt.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("lt.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::lt>::invoke));

    // schema: lt.Scalar(Tensor self, Scalar other) -> Tensor
    m.impl("lt.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::lt>::invoke));
    // schema: lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("lt.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::lt>::invoke_into));

    // Special ops
    // schema: logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logaddexp.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp>::invoke_into));
    // schema: logaddexp(Tensor self, Tensor other) -> Tensor
    m.impl("logaddexp", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp>::invoke));
    // schema: logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logaddexp2.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp2>::invoke_into));
    // schema: logaddexp2(Tensor self, Tensor other) -> Tensor
    m.impl("logaddexp2", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp2>::invoke));
}

static inline void register_reductions(torch::Library& m) {
    // =========================
    // Reductions
    // =========================
    // Sum
    // schema: sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum", TORCH_FN(tt_eager::ext::reduction_all<ttnn::sum>::invoke));
    // schema: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum.dim_IntList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke));
    // schema: sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out)
    // -> Tensor(a!)
    m.impl("sum.IntList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_into));
    // schema: sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum.dim_DimnameList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames));
    // schema: sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("sum.DimnameList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames_into));

    // Mean
    // schema: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean", TORCH_FN(tt_eager::ext::reduction_all<ttnn::mean>::invoke));
    // schema: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke));
    // schema: mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("mean.out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_into));
    // schema: mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames));
    // schema: mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!)
    // out) -> Tensor(a!)
    m.impl("mean.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames_into));

    // Max / Min (value-only reductions; aten::max/min no dtype)
    // schema: max(Tensor self) -> Tensor
    m.impl("max", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::max>::invoke));
    // schema: min(Tensor self) -> Tensor
    m.impl("min", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::min>::invoke));

    // max/min with indices along dim (return (values, indices))
    using MaxPair = tt_eager::ext::reduction_dim_pair<ttnn::max, ttnn::experimental::argmax>;
    // schema: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("max.dim", TORCH_FN(MaxPair::invoke));
    // schema: max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) ->
    // (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.dim_max", TORCH_FN(MaxPair::invoke_into));
    // schema: max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("max.names_dim", TORCH_FN(MaxPair::invoke_dimname));
    // schema: max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values)
    // -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.names_dim_max", TORCH_FN(MaxPair::invoke_dimname_into));

    using MinPair = tt_eager::ext::reduction_dim_pair<ttnn::min, ttnn::experimental::argmin>;
    // schema: min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("min.dim", TORCH_FN(MinPair::invoke));
    // schema: min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) ->
    // (Tensor(a!) values, Tensor(b!) indices)
    m.impl("min.dim_min", TORCH_FN(MinPair::invoke_into));
    // schema: min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("min.names_dim", TORCH_FN(MinPair::invoke_dimname));
    // schema: min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!)
    // min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("min.names_dim_min", TORCH_FN(MinPair::invoke_dimname_into));

    // Std / Var
    // Base (all-elements) with unbiased flag default (correction)
    // schema: var(Tensor self, bool unbiased=True) -> Tensor
    m.impl("var", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::var>::invoke));
    // schema: std(Tensor self, bool unbiased=True) -> Tensor
    m.impl("std", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::std>::invoke));

    // schema: var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("var.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::var>::invoke_into));
    // schema: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("var.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke));
    // schema: var.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) ->
    // Tensor
    m.impl("var.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames));
    // schema: var.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl(
        "var.correction_names_out",
        TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames_into));
    // schema: var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("var.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_into));
    // schema: var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke));
    // schema: var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames));
    // schema: var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("var.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames_into));

    // schema: std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("std.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::std>::invoke_into));
    // schema: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("std.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke));
    // schema: std.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) ->
    // Tensor
    m.impl("std.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames));
    // schema: std.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl(
        "std.correction_names_out",
        TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames_into));
    // schema: std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("std.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_into));
    // schema: std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke));
    // schema: std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames));
    // schema: std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("std.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames_into));
}

static inline void register_random_ops(torch::Library& m) {
    // =========================
    // Random
    // =========================
    // schema: bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
    m.impl("bernoulli", TORCH_FN(tt_eager::ext::unary_random_seeded<ttnn::bernoulli>::invoke));
    // schema: bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
    m.impl("bernoulli.out", TORCH_FN(tt_eager::ext::unary_random_seeded<ttnn::bernoulli>::invoke_into));
    // bernoulli_.Tensor
    // bernoulli_.float
    // cauchy_
    // exponential_
    // geometric_
    // normal_
    // random_ family: use ttnn::rand creator semantics to match PyTorch behavior
    // schema: random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
    m.impl("random_", TORCH_FN(tt_eager::ext::random_like_rand::invoke_inplace));
    // schema: random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
    m.impl("random_.from", TORCH_FN(tt_eager::ext::random_like_rand::invoke_from_inplace));
    // schema: random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
    m.impl("random_.to", TORCH_FN(tt_eager::ext::random_like_rand::invoke_to_inplace));
    // uniform_
    // schema: uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
    m.impl("uniform_", TORCH_FN(tt_eager::ext::unary_random_uniform<ttnn::uniform>::invoke_inplace));
}

static inline void register_conv_and_pool_ops(torch::Library& m) {
    // =========================
    // Convolution ops
    // =========================
    // From native_functions.yaml, available schemas (signatures below for reference):
    // - convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation,
    // bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
    // - convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[]
    // dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
    // - _convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation,
    // bool transposed, SymInt[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled,
    // bool allow_tf32) -> Tensor
    // - _convolution_mode(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, str padding, SymInt[] dilation,
    // SymInt groups) -> Tensor
    // - conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1]
    // dilation=1, SymInt groups=1) -> Tensor
    // - conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2]
    // dilation=1, SymInt groups=1) -> Tensor
    // - conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3]
    // dilation=1, SymInt groups=1) -> Tensor
    // - conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, str padding="valid",
    // SymInt[1] dilation=1, SymInt groups=1) -> Tensor
    // - conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid",
    // SymInt[2] dilation=1, SymInt groups=1) -> Tensor
    // - conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, str padding="valid",
    // SymInt[3] dilation=1, SymInt groups=1) -> Tensor
    // - conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
    // - conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0,
    // SymInt[1] output_padding=0, SymInt groups=1, SymInt[1] dilation=1) -> Tensor
    // - conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0,
    // SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
    // - conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0,
    // SymInt[3] output_padding=0, SymInt groups=1, SymInt[3] dilation=1) -> Tensor

    // Implemented via TTNN wrappers (Conv):
    // conv1d
    m.impl("conv1d", TORCH_FN(tt_eager::ext::conv1d_aten::invoke));
    // conv2d
    m.impl("conv2d", TORCH_FN(tt_eager::ext::conv2d_aten::invoke));
    // conv3d (uses ttnn::experimental::conv3d)
    m.impl("conv3d", TORCH_FN(tt_eager::ext::conv3d_aten::invoke));
    // conv_transpose2d.input
    m.impl("conv_transpose2d.input", TORCH_FN(tt_eager::ext::conv_transpose2d_aten::invoke));

    // Register generic convolution entry points
    m.impl("convolution", TORCH_FN(aten_convolution_dispatch));
    m.impl("convolution_overrideable", TORCH_FN(aten_convolution_dispatch));

    // Pooling (2D):
    // max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool
    // ceil_mode=False)
    m.impl("max_pool2d", TORCH_FN(tt_eager::ext::max_pool2d_aten::invoke));
    // avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool
    // count_include_pad=True)
    m.impl("avg_pool2d", TORCH_FN(tt_eager::ext::avg_pool2d_aten::invoke));
    // adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
    m.impl("adaptive_avg_pool2d", TORCH_FN(tt_eager::ext::adaptive_avg_pool2d_aten::invoke));

    // Not implemented yet (reserved):
    // m.impl("_convolution", ...);
    // m.impl("_convolution_mode", ...);
    // m.impl("conv1d.padding", ...);
    // m.impl("conv2d.padding", ...);
    // m.impl("conv3d.padding", ...);
    // m.impl("conv_tbc", ...);
    // m.impl("conv_transpose1d", ...);
    // m.impl("conv_transpose3d.input", ...);
}
}  // namespace

// This macro registers the kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at
// http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    using namespace tt_eager::ext;

    register_core_creation_and_copy(m);
    register_unary_ops(m);
    register_binary_ops(m);
    register_conv_and_pool_ops(m);
    register_reductions(m);
    register_random_ops(m);
}

// This macro registers helper functions associated with the ttnn_device_mode module that can be used in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("as_torch_device", &as_torch_device, "get torch device from existing ttnn device");
    m.def("get_ttnn_tensor", &get_ttnn_tensor, "open ttnn device and get torch device");
}

// Fallbacks
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) { m.fallback(torch::CppFunction::makeFallthrough()); }

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) { m.fallback(torch::CppFunction::makeFallthrough()); }
