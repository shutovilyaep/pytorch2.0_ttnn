#include <ATen/native/DispatchStub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "ttnn_cpp_extension/utils/device.hpp"

#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"
#include "ttnn_cpp_extension/core/copy.hpp"

#include "ttnn_cpp_extension/ops/creation.hpp"

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"
// TODO: NOW: FIXME временно отключено: autograd_wrap.hpp вызывает ошибки компиляции при парсинге шаблонов
// #include "ttnn_cpp_extension/utils/autograd_wrap.hpp"

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

// Register custom allocator. Only used to create dummy Torch tensor object.
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &get_ttnn_custom_allocator());


namespace {

static inline void register_core_creation_and_copy(torch::Library& m) {
    // =========================
    // Core ops: creation and copy
    // =========================
    // From Pytorch's NamesRegistrations.cpp
    m.impl("aten::empty_strided", &tt_eager::ops::create::custom_empty_strided);
    m.impl("empty.memory_format", &tt_eager::ops::create::custom_empty_memory_format);
    m.impl("_copy_from", &ttnn_copy_from);
    // Pending TTNN core ops to register (from ttnn_ops_grouped.txt)
    // ttnn::to_dtype
    // ttnn::to_layout
    // ttnn::to_memory_config
}

static inline void register_unary_ops(torch::Library& m) {
    // =========================
    // Unary ops
    // =========================
    m.impl("abs", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke));
    m.impl("abs.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_into));
    m.impl("abs_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_inplace));
    // alias: absolute -> abs
    m.impl("absolute", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke));
    m.impl("absolute.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_into));
    m.impl("absolute_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_inplace));
    m.impl("neg", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke));
    m.impl("neg.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke_into));
    m.impl("neg_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke_inplace));
    m.impl("reciprocal", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke));
    m.impl("reciprocal.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke_into));
    m.impl("reciprocal_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke_inplace));
    m.impl("sqrt", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke));
    m.impl("sqrt.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke_into));
    m.impl("sqrt_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke_inplace));
    m.impl("rsqrt", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke));
    m.impl("rsqrt.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke_into));
    m.impl("rsqrt_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke_inplace));
    m.impl("square", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke));
    m.impl("square.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke_into));
    m.impl("square_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke_inplace));
    m.impl("sin", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke));
    m.impl("sin.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke_into));
    m.impl("sin_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke_inplace));
    m.impl("cos", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke));
    m.impl("cos.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke_into));
    m.impl("cos_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke_inplace));
    m.impl("tan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke));
    m.impl("tan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke_into));
    m.impl("tan_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke_inplace));
    m.impl("sinh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke));
    m.impl("sinh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke_into));
    m.impl("sinh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke_inplace));
    m.impl("cosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke));
    m.impl("cosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke_into));
    m.impl("cosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke_inplace));
    m.impl("tanh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke));
    m.impl("tanh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke_into));
    m.impl("tanh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke_inplace));
    m.impl("floor", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke));
    m.impl("floor.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke_into));
    m.impl("floor_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke_inplace));
    m.impl("ceil", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke));
    m.impl("ceil.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke_into));
    m.impl("ceil_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke_inplace));
    m.impl("trunc", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke));
    m.impl("trunc.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke_into));
    m.impl("trunc_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke_inplace));
    m.impl("frac", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke));
    m.impl("frac.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke_into));
    m.impl("frac_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke_inplace));
    m.impl("bitwise_not", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke));
    m.impl("bitwise_not.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke_into));
    m.impl("bitwise_not_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke_inplace));
    m.impl("logical_not", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke));
    m.impl("logical_not.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke_into));
    m.impl("logical_not_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke_inplace));
    m.impl("sign", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke));
    m.impl("sign.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke_into));
    m.impl("sign_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke_inplace));
    m.impl("signbit", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::signbit>::invoke));
    m.impl("signbit.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::signbit>::invoke_into));
    m.impl("i0", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke));
    m.impl("i0.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke_into));
    m.impl("i0_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke_inplace));
    m.impl("erf", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke));
    m.impl("erf.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke_into));
    m.impl("erf_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke_inplace));
    m.impl("erfc", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke));
    m.impl("erfc.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke_into));
    m.impl("erfc_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke_inplace));
    m.impl("erfinv", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke));
    m.impl("erfinv.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke_into));
    m.impl("erfinv_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke_inplace));
    m.impl("exp", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke));
    m.impl("exp.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke_into));
    m.impl("exp_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke_inplace));
    m.impl("log", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke));
    m.impl("log.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke_into));
    m.impl("log_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke_inplace));
    m.impl("log10", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke));
    m.impl("log10.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke_into));
    m.impl("log10_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke_inplace));
    m.impl("log2", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke));
    m.impl("log2.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke_into));
    m.impl("log2_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke_inplace));
    m.impl("log1p", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke));
    m.impl("log1p.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke_into));
    m.impl("log1p_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke_inplace));
    m.impl("acos", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke));
    m.impl("acos.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke_into));
    m.impl("acos_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke_inplace));
    m.impl("acosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke));
    m.impl("acosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_into));
    m.impl("acosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_inplace));
    m.impl("angle", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke)); // TODO: check
    m.impl("angle.out", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke_into)); // TODO: check
    m.impl("angle_", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke_inplace)); // TODO: check
    // alias: arccosh -> acosh
    m.impl("arccosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke));
    m.impl("arccosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_into));
    m.impl("arccosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_inplace));
    m.impl("asin", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke));
    m.impl("asin.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke_into));
    m.impl("asin_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke_inplace));
    
    m.impl("asinh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke));
    m.impl("asinh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke_into));
    m.impl("asinh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke_inplace));
    m.impl("atan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke));
    m.impl("atan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke_into));
    m.impl("atan_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke_inplace));
    
    m.impl("atanh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke));
    m.impl("atanh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke_into));
    m.impl("atanh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke_inplace));
    m.impl("conj", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke));
    m.impl("conj.out", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke_into));
    m.impl("conj_", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke_inplace));
    // ttnn::conj_bw
    m.impl("deg2rad", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke));
    m.impl("deg2rad.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke_into));
    m.impl("deg2rad_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke_inplace));
    
    m.impl("digamma", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke));
    m.impl("digamma.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke_into));
    m.impl("digamma_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke_inplace));
    
    m.impl("expm1", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke));
    m.impl("expm1.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke_into));
    m.impl("expm1_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke_inplace));
    
    // imag
    m.impl("isfinite", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isfinite>::invoke));
    m.impl("isfinite.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isfinite>::invoke_into));
    m.impl("isinf", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isinf>::invoke));
    m.impl("isinf.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isinf>::invoke_into));
    m.impl("isnan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isnan>::invoke));
    m.impl("isnan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isnan>::invoke_into));
    
    m.impl("lgamma", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke));
    m.impl("lgamma.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke_into));
    m.impl("lgamma_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke_inplace));
    
    m.impl("rad2deg", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke));
    m.impl("rad2deg.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke_into));
    m.impl("rad2deg_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke_inplace));
    
    
    m.impl("relu", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::relu>::invoke));
    m.impl("relu_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::relu>::invoke_inplace));
    // real

    m.impl("round", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke));
    m.impl("round.out", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke_into));
    m.impl("round.decimals", TORCH_FN(tt_eager::ext::unary_tensor_opt_int<ttnn::round>::invoke_decimals));
    m.impl("round.decimals_out", TORCH_FN(tt_eager::ext::unary_tensor_opt_int<ttnn::round>::invoke_decimals_into));
    m.impl("round_", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke_inplace));
    // ttnn::round_bw
    m.impl("sigmoid", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke));
    m.impl("sigmoid.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke_into));
    m.impl("sigmoid_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke_inplace));
    
}

static inline void register_binary_ops(torch::Library& m) {
    // =========================
    // Binary ops
    // =========================
    m.impl("add.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke_into));
    m.impl("add.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke));
    m.impl("add.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::add>::invoke));
    m.impl("add_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::add_>::invoke_inplace));
    m.impl("add_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke_inplace));
    // _add_relu.* = relu(add.Tensor with alpha=1)
    // Match aten schema: (Tensor, Tensor, Scalar alpha=1)
    using AddReluAlphaWrapper = tt_eager::ext::binary_tensor_tensor_alpha_then_unary<ttnn::addalpha, ttnn::relu>;
    m.impl("_add_relu.Tensor", TORCH_FN(AddReluAlphaWrapper::invoke));
    m.impl("_add_relu.out", TORCH_FN(AddReluAlphaWrapper::invoke_into));
    m.impl("_add_relu_.Tensor", TORCH_FN(AddReluAlphaWrapper::invoke_inplace));

    m.impl("sub.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke_into));
    m.impl("sub.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke));
    m.impl("sub.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::subtract>::invoke));
    m.impl("sub_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::subtract_>::invoke_inplace));
    m.impl("sub_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke_inplace));
    // rsub: reverse subtract
    // rsub.Tensor: rsub(self, other, alpha) = other - alpha*self
    // rsub.Scalar: rsub(self, other, alpha) = other - alpha*self
    m.impl("rsub.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha_swapped<ttnn::subalpha>::invoke)); // TODO: to check
    m.impl("rsub.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::rsub>::invoke)); // TODO: to check

    // Arithmetic ops
    m.impl("mul.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply>::invoke_into));
    m.impl("mul.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply>::invoke));
    m.impl("mul_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply_>::invoke_inplace));

    m.impl("div.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide>::invoke_into));
    m.impl("div.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide>::invoke));
    m.impl("div.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::divide>::invoke));
    m.impl("div_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::divide_>::invoke_inplace));
    m.impl("div_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::divide_>::invoke_inplace));

    m.impl("floor_divide", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke));
    m.impl("floor_divide.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::floor_div>::invoke));
    m.impl("floor_divide.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke_into));
    m.impl("floor_divide_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float<ttnn::floor_div>::invoke_inplace));
    m.impl("floor_divide_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::floor_div>::invoke_inplace));
    // true_divide.Scalar
    // true_divide.out
    // true_divide_.Scalar
    // true_divide_.Tensor
    // (handled via divide) no direct ttnn::true_divide
    m.impl("pow.Scalar", TORCH_FN(tt_eager::ext::binary_scalar_tensor_as_tensor<ttnn::pow>::invoke));
    m.impl("pow.Scalar_out", TORCH_FN(tt_eager::ext::binary_scalar_tensor_as_tensor<ttnn::pow>::invoke_into));
    m.impl("pow.Tensor_Scalar", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke));
    m.impl("pow.Tensor_Scalar_out", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke_into));
    m.impl("pow_.Scalar", TORCH_FN(tt_eager::ext::unary_tensor_int<ttnn::power>::invoke_inplace));
    m.impl("pow_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::pow>::invoke_inplace));
    m.impl("nextafter.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::nextafter>::invoke_into));
    m.impl("nextafter", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::nextafter>::invoke));
    m.impl("dot", TORCH_FN(tt_eager::ext::binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke));
    m.impl("dot.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke_into));
    m.impl("hypot.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::hypot>::invoke_into));
    m.impl("hypot", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::hypot>::invoke));
    
    m.impl("matmul.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::matmul>::invoke_into));
    m.impl("matmul", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::matmul>::invoke));
    // mm
    // mm.out
    // mv
    // mv.out
    // bmm
    // bmm.out

    // Logical ops
    m.impl("logical_and.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and>::invoke_into));
    m.impl("logical_and", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and>::invoke));
    m.impl("logical_and_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_and_>::invoke_inplace));

    m.impl("logical_or.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or>::invoke_into));
    m.impl("logical_or", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or>::invoke));
    m.impl("logical_or_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_or_>::invoke_inplace));

    m.impl("logical_xor.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor>::invoke_into));
    m.impl("logical_xor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor>::invoke));
    m.impl("logical_xor_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logical_xor_>::invoke_inplace));

    // Trigonometric binary ops
    m.impl("atan2.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke_into));
    m.impl("atan2", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke));
    m.impl("atan2_", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::atan2>::invoke_inplace));
    

    // Relational ops (Tensor versions only)
    m.impl("eq.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::eq>::invoke_into));
    m.impl("eq.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::eq>::invoke));

    m.impl("eq.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::eq>::invoke));
    m.impl("eq.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::eq>::invoke_into));

    m.impl("ne.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ne>::invoke_into));
    m.impl("ne.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ne>::invoke));
    
    m.impl("ne.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ne>::invoke));
    m.impl("ne.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ne>::invoke_into));

    m.impl("ge.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ge>::invoke_into));
    m.impl("ge.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::ge>::invoke));
    
    m.impl("ge.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ge>::invoke));
    m.impl("ge.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::ge>::invoke_into));

    m.impl("gt.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::gt>::invoke_into));
    m.impl("gt.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::gt>::invoke));
    
    m.impl("gt.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::gt>::invoke));
    m.impl("gt.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::gt>::invoke_into));

    m.impl("le.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::le>::invoke_into));
    m.impl("le.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::le>::invoke));

    m.impl("le.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::le>::invoke));
    m.impl("le.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::le>::invoke_into));

    m.impl("lt.Tensor_out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::lt>::invoke_into));
    m.impl("lt.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::lt>::invoke));

    m.impl("lt.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::lt>::invoke));
    m.impl("lt.Scalar_out", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::lt>::invoke_into));

    // Special ops
    m.impl("logaddexp.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp>::invoke_into));
    m.impl("logaddexp", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp>::invoke));
    m.impl("logaddexp2.out", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp2>::invoke_into));
    m.impl("logaddexp2", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::logaddexp2>::invoke));
}

static inline void register_reductions(torch::Library& m) {
    // =========================
    // Reductions
    // =========================
    // Sum
    m.impl("sum", TORCH_FN(tt_eager::ext::reduction_all<ttnn::sum>::invoke));
    m.impl("sum.dim_IntList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke));
    m.impl("sum.IntList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_into));
    m.impl("sum.dim_DimnameList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames));
    m.impl("sum.DimnameList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames_into));

    // Mean
    m.impl("mean", TORCH_FN(tt_eager::ext::reduction_all<ttnn::mean>::invoke));
    m.impl("mean.dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke));
    m.impl("mean.out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_into));
    m.impl("mean.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames));
    m.impl("mean.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames_into));

    // Max / Min (value-only reductions; aten::max/min no dtype)
    m.impl("max", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::max>::invoke));
    m.impl("min", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::min>::invoke));

    // max/min with indices along dim (return (values, indices))
    using MaxPair = tt_eager::ext::reduction_dim_pair<ttnn::max, ttnn::experimental::argmax>;
    m.impl("max.dim", TORCH_FN(MaxPair::invoke));
    m.impl("max.dim_max", TORCH_FN(MaxPair::invoke_into));
    m.impl("max.names_dim", TORCH_FN(MaxPair::invoke_dimname));
    m.impl("max.names_dim_max", TORCH_FN(MaxPair::invoke_dimname_into));

    using MinPair = tt_eager::ext::reduction_dim_pair<ttnn::min, ttnn::experimental::argmin>;
    m.impl("min.dim", TORCH_FN(MinPair::invoke));
    m.impl("min.dim_min", TORCH_FN(MinPair::invoke_into));
    m.impl("min.names_dim", TORCH_FN(MinPair::invoke_dimname));
    m.impl("min.names_dim_min", TORCH_FN(MinPair::invoke_dimname_into));

    // Std / Var
    // Base (all-elements) with unbiased flag default (correction)
    m.impl("var", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::var>::invoke));
    m.impl("std", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::std>::invoke));

    m.impl("var.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::var>::invoke_into));
    m.impl("var.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke));
    m.impl("var.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames));
    m.impl("var.correction_names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames_into));
    m.impl("var.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_into));
    m.impl("var.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke));
    m.impl("var.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames));
    m.impl("var.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames_into));


    m.impl("std.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::std>::invoke_into));
    m.impl("std.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke));
    m.impl("std.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames));
    m.impl("std.correction_names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames_into));
    m.impl("std.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_into));
    m.impl("std.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke));
    m.impl("std.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames));
    m.impl("std.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames_into));
}

static inline void register_random_ops(torch::Library& m) {
    // =========================
    // Random
    // =========================
    m.impl("bernoulli", TORCH_FN(tt_eager::ext::unary_random_seeded<ttnn::bernoulli>::invoke));
    m.impl("bernoulli.out", TORCH_FN(tt_eager::ext::unary_random_seeded<ttnn::bernoulli>::invoke_into));
    // bernoulli_.Tensor
    // bernoulli_.float
    // cauchy_
    // exponential_
    // geometric_
    // normal_
    // random_ family: use ttnn::rand creator semantics to match PyTorch behavior
    m.impl("random_", TORCH_FN(tt_eager::ext::random_like_rand::invoke_inplace));
    m.impl("random_.from", TORCH_FN(tt_eager::ext::random_like_rand::invoke_from_inplace));
    m.impl("random_.to", TORCH_FN(tt_eager::ext::random_like_rand::invoke_to_inplace));
    // uniform_
    m.impl("uniform_", TORCH_FN(tt_eager::ext::unary_random_uniform<ttnn::uniform>::invoke_inplace));
}

} // namespace

// This macro registers the kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at
// http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    register_core_creation_and_copy(m);
    register_unary_ops(m);
    register_binary_ops(m);
    register_reductions(m);
    register_random_ops(m);
}

// This macro registers helper functions associated with the ttnn_device_mode module that can be used in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("as_torch_device", &as_torch_device, "get torch device from existing ttnn device");
    m.def("get_ttnn_tensor", &get_ttnn_tensor, "open ttnn device and get torch device");
}

// Fallbacks
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}
