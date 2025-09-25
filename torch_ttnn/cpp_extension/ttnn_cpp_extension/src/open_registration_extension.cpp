#include <ATen/native/DispatchStub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "ttnn_cpp_extension/utils/device.hpp"

#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"
#include "ttnn_cpp_extension/core/copy.hpp"

#include "ttnn_cpp_extension/ops/creation.hpp"

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/bernoulli/bernoulli.hpp>

// Register custom allocator. Only used to create dummy Torch tensor object.
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &get_ttnn_custom_allocator());


// This macro registers the kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at
// http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
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
    // =========================
    // Unary ops
    // =========================
    m.impl("abs", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::abs>::invoke));
    m.impl("abs.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::abs>::invoke_out));
    // abs_
    // absolute
    // absolute.out
    // absolute_
    m.impl("neg", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::neg>::invoke));
    m.impl("neg.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::neg>::invoke_out));
    // neg_
    m.impl("reciprocal", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::reciprocal>::invoke));
    m.impl("reciprocal.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::reciprocal>::invoke_out));
    // reciprocal_
    m.impl("sqrt", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sqrt>::invoke));
    m.impl("sqrt.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sqrt>::invoke_out));
    // sqrt_
    m.impl("rsqrt", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::rsqrt>::invoke));
    m.impl("rsqrt.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::rsqrt>::invoke_out));
    // rsqrt_
    m.impl("square", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::square>::invoke));
    m.impl("square.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::square>::invoke_out));
    // square_
    m.impl("sin", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sin>::invoke));
    m.impl("sin.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sin>::invoke_out));
    // sin_
    m.impl("cos", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::cos>::invoke));
    m.impl("cos.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::cos>::invoke_out));
    // cos_
    m.impl("tan", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::tan>::invoke));
    m.impl("tan.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::tan>::invoke_out));
    // tan_
    m.impl("sinh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sinh>::invoke));
    m.impl("sinh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sinh>::invoke_out));
    // sinh_
    m.impl("cosh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::cosh>::invoke));
    m.impl("cosh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::cosh>::invoke_out));
    // cosh_
    m.impl("tanh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::tanh>::invoke));
    m.impl("tanh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::tanh>::invoke_out));
    // tanh_
    m.impl("floor", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::floor>::invoke));
    m.impl("floor.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::floor>::invoke_out));
    // floor_
    m.impl("ceil", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::ceil>::invoke));
    m.impl("ceil.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::ceil>::invoke_out));
    // ceil_
    m.impl("trunc", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::trunc>::invoke));
    m.impl("trunc.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::trunc>::invoke_out));
    // trunc_
    m.impl("frac", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::frac>::invoke));
    m.impl("frac.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::frac>::invoke_out));
    // frac_
    m.impl("bitwise_not", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::bitwise_not>::invoke));
    m.impl("bitwise_not.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::bitwise_not>::invoke_out));
    // bitwise_not_
    m.impl("logical_not", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::logical_not>::invoke));
    m.impl("logical_not.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::logical_not>::invoke_out));
    // logical_not_
    m.impl("sign", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sign>::invoke));
    m.impl("sign.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sign>::invoke_out));
    // sign_
    m.impl("signbit", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::signbit>::invoke));
    m.impl("signbit.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::signbit>::invoke_out));
    m.impl("i0", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::i0>::invoke));
    m.impl("i0.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::i0>::invoke_out));
    // i0_
    m.impl("erf", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erf>::invoke));
    m.impl("erf.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erf>::invoke_out));
    // erf_
    m.impl("erfc", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erfc>::invoke));
    m.impl("erfc.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erfc>::invoke_out));
    // erfc_
    m.impl("erfinv", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erfinv>::invoke));
    m.impl("erfinv.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::erfinv>::invoke_out));
    // erfinv_
    m.impl("exp", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::exp>::invoke));
    m.impl("exp.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::exp>::invoke_out));
    // exp_
    m.impl("log", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log>::invoke));
    m.impl("log.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log>::invoke_out));
    // log_
    m.impl("log10", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log10>::invoke));
    m.impl("log10.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log10>::invoke_out));
    // log10_
    m.impl("log2", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log2>::invoke));
    m.impl("log2.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log2>::invoke_out));
    // log2_
    m.impl("log1p", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log1p>::invoke));
    m.impl("log1p.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::log1p>::invoke_out));
    // log1p_
    m.impl("acos", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::acos>::invoke));
    m.impl("acos.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::acos>::invoke_out));
    // acos_
    m.impl("acosh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::acosh>::invoke));
    m.impl("acosh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::acosh>::invoke_out));
    // acosh_
    // angle
    // angle.out
    // ttnn::angle
    // arccosh
    // arccosh.out
    // arccosh_
    m.impl("asin", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::asin>::invoke));
    m.impl("asin.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::asin>::invoke_out));
    // asin_
    // ttnn::asin_bw
    m.impl("asinh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::asinh>::invoke));
    m.impl("asinh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::asinh>::invoke_out));
    // asinh_
    m.impl("atan", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::atan>::invoke));
    m.impl("atan.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::atan>::invoke_out));
    // atan2
    // atan2.out
    // ttnn::atan2
    // atan2_
    // ttnn::atan2_bw
    // atan_
    // ttnn::atan_bw
    m.impl("atanh", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::atanh>::invoke));
    m.impl("atanh.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::atanh>::invoke_out));
    // atanh_
    // conj
    // ttnn::conj
    // ttnn::conj_bw
    m.impl("deg2rad", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::deg2rad>::invoke));
    m.impl("deg2rad.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::deg2rad>::invoke_out));
    // deg2rad_
    // digamma
    // digamma.out
    // digamma_
    // ttnn::digamma
    // ttnn::digamma_bw
    m.impl("expm1", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::expm1>::invoke));
    m.impl("expm1.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::expm1>::invoke_out));
    // expm1_
    // ttnn::expm1_bw
    // imag
    // isfinite
    // isinf
    // isnan
    // lgamma
    // lgamma.out
    // lgamma_
    // log_
    // log10_
    // log1p_
    // log2_
    m.impl("rad2deg", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::rad2deg>::invoke));
    m.impl("rad2deg.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::rad2deg>::invoke_out));
    // rad2deg_
    // ttnn::rad2deg_bw
    // relu
    // relu_
    // real
    m.impl("round", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::round>::invoke));
    m.impl("round.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::round>::invoke_out));
    // round_
    // ttnn::round_bw
    m.impl("sigmoid", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sigmoid>::invoke));
    m.impl("sigmoid.out", TORCH_FN(tt_eager::ext::unary_wrapper<ttnn::sigmoid>::invoke_out));
    // sigmoid_
    // ttnn::sigmoid_bw

    // Pending TTNN unary ops to register (from ttnn_ops_grouped.txt)
    // ttnn::abs_bw
    // ttnn::add_sfpu
    // ttnn::alt_complex_rotate90
    // ttnn::angle
    // ttnn::asin_bw
    // ttnn::asinh_bw
    // ttnn::atan_bw
    // ttnn::atanh_bw
    // ttnn::cbrt
    // ttnn::ceil_bw
    // ttnn::celu
    // ttnn::celu_bw
    // ttnn::clamp
    // ttnn::clamp_bw
    // ttnn::clamp_tss
    // ttnn::clip
    // ttnn::clip_bw
    // ttnn::conj
    // ttnn::cos_bw
    // ttnn::cosh_bw
    // ttnn::deg2rad_bw
    // ttnn::digamma
    // ttnn::digamma_bw
    // ttnn::div_no_nan_bw
    // ttnn::div_sfpu
    // ttnn::elu
    // ttnn::elu_bw
    // ttnn::eq_unary
    // ttnn::eqz
    // ttnn::erf_bw
    // ttnn::erfc_bw
    // ttnn::erfinv_bw
    // ttnn::exp2
    // ttnn::exp2_bw
    // ttnn::exp_bw
    // ttnn::expm1_bw
    // ttnn::fill
    // ttnn::fill_bw
    // ttnn::fill_zero_bw
    // ttnn::floor_bw
    // ttnn::frac_bw
    // ttnn::ge_unary
    // ttnn::geglu
    // ttnn::gelu
    // ttnn::gelu_bw
    // ttnn::gez
    // ttnn::glu
    // ttnn::gt_unary
    // ttnn::gtz
    // ttnn::hardshrink
    // ttnn::hardshrink_bw
    // ttnn::hardsigmoid
    // ttnn::hardsigmoid_bw
    // ttnn::hardswish
    // ttnn::hardswish_bw
    // ttnn::hardtanh
    // ttnn::hardtanh_bw
    // ttnn::heaviside
    // ttnn::i0_bw
    // ttnn::i1
    // ttnn::identity
    // ttnn::imag
    // ttnn::is_imag
    // ttnn::is_real
    // ttnn::isfinite
    // ttnn::isinf
    // ttnn::isnan
    // ttnn::isneginf
    // ttnn::isposinf
    // ttnn::le_unary
    // ttnn::leaky_relu
    // ttnn::leaky_relu_bw
    // ttnn::lez
    // ttnn::lgamma
    // ttnn::lgamma_bw
    // ttnn::log10_bw
    // ttnn::log1p_bw
    // ttnn::log2_bw
    // ttnn::log_bw
    // ttnn::log_sigmoid
    // ttnn::log_sigmoid_bw
    // ttnn::logical_not_
    // ttnn::logit
    // ttnn::logit_bw
    // ttnn::logiteps_bw
    // ttnn::lt_unary
    // ttnn::ltz
    // ttnn::mish
    // ttnn::mul_sfpu
    // ttnn::multigammaln
    // ttnn::multigammaln_bw
    // ttnn::ne_unary
    // ttnn::neg_bw
    // ttnn::nez
    // ttnn::normalize_global
    // ttnn::normalize_hw
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::operation_name
    // ttnn::polar
    // ttnn::polygamma
    // ttnn::polygamma_bw
    // ttnn::pow_bw
    // ttnn::power
    // ttnn::prelu_sfpu
    // ttnn::prim::tanh_accurate
    // ttnn::prim::unary
    // ttnn::prod_bw
    // ttnn::rad2deg_bw
    // ttnn::rdiv
    // ttnn::rdiv_bw
    // ttnn::real
    // ttnn::reciprocal_bw
    // ttnn::reglu
    // ttnn::relu
    // ttnn::relu6
    // ttnn::relu6_bw
    // ttnn::relu_bw
    // ttnn::relu_max
    // ttnn::relu_min
    // ttnn::repeat_bw
    // ttnn::round_bw
    // ttnn::rpow
    // ttnn::rpow_bw
    // ttnn::rsqrt_bw
    // ttnn::selu
    // ttnn::selu_bw
    // ttnn::sigmoid_accurate
    // ttnn::sigmoid_bw
    // ttnn::sign_bw
    // ttnn::silu
    // ttnn::silu_bw
    // ttnn::sin_bw
    // ttnn::sinh_bw
    // ttnn::softplus
    // ttnn::softplus_bw
    // ttnn::softshrink
    // ttnn::softshrink_bw
    // ttnn::softsign
    // ttnn::softsign_bw
    // ttnn::sqrt_bw
    // ttnn::square_bw
    // ttnn::std_hw
    // ttnn::sub_sfpu
    // ttnn::swiglu
    // ttnn::swish
    // ttnn::tan_bw
    // ttnn::tanh_accurate
    // ttnn::tanh_bw
    // ttnn::tanhshrink
    // ttnn::tanhshrink_accurate
    // ttnn::tanhshrink_bw
    // ttnn::threshold
    // ttnn::threshold_bw
    // ttnn::tiled_prod
    // ttnn::tril
    // ttnn::triu
    // ttnn::trunc_bw
    // ttnn::unary_chain
    // ttnn::unary_remainder
    // ttnn::var_hw


    // =========================
    // Binary ops
    // =========================
    m.impl("add.out", TORCH_FN(tt_eager::ext::binary_alpha_wrapper<ttnn::addalpha>::invoke_out));
    m.impl("add.Tensor", TORCH_FN(tt_eager::ext::binary_alpha_wrapper<ttnn::addalpha>::invoke));
    // add.Scalar
    // add_.Scalar
    // add_.Tensor
    // _add_relu.Tensor
    // _add_relu.out
    // _add_relu_.Tensor

    m.impl("sub.out", TORCH_FN(tt_eager::ext::binary_alpha_wrapper<ttnn::subalpha>::invoke_out));
    m.impl("sub.Tensor", TORCH_FN(tt_eager::ext::binary_alpha_wrapper<ttnn::subalpha>::invoke));
    // sub.Scalar
    // sub_.Scalar
    // sub_.Tensor
    // rsub.Scalar
    // rsub.Tensor

    // Arithmetic ops
    m.impl("mul.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::multiply>::invoke_out));
    m.impl("mul.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::multiply>::invoke));
    // mul_.Tensor

    m.impl("div.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::divide>::invoke_out));
    m.impl("div.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::divide>::invoke));
    // div.Scalar
    // div_.Scalar
    // div_.Tensor
    // floor_divide
    // floor_divide.Scalar
    // floor_divide.out
    // floor_divide_.Scalar
    // floor_divide_.Tensor
    // ttnn::floor_div
    // true_divide.Scalar
    // true_divide.out
    // true_divide_.Scalar
    // true_divide_.Tensor
    // (handled via divide) no direct ttnn::true_divide
    // pow.Scalar
    // pow.Scalar_out
    // pow.Tensor_Scalar
    // pow.Tensor_Scalar_out
    // pow.Tensor_Tensor
    // pow.Tensor_Tensor_out
    // pow_.Scalar
    // pow_.Tensor
    // ttnn::pow
    // nextafter
    // nextafter.out
    // nextafter_
    // ttnn::nextafter
    // dot
    // dot.out
    // hypot
    // hypot.out
    // hypot_
    // ttnn::hypot
    // ttnn::hypot_bw
    // matmul
    // matmul.out
    // mm
    // mm.out
    // mv
    // mv.out
    // bmm
    // bmm.out

    // Logical ops
    m.impl("logical_and.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_and>::invoke_out));
    m.impl("logical_and", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_and>::invoke));
    // logical_and_
    // ttnn::logical_and_

    m.impl("logical_or.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_or>::invoke_out));
    m.impl("logical_or", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_or>::invoke));
    // logical_or_
    // ttnn::logical_or_

    m.impl("logical_xor.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_xor>::invoke_out));
    m.impl("logical_xor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_xor>::invoke));
    // logical_xor_
    // ttnn::logical_xor_

    // Relational ops (Tensor versions only)
    m.impl("eq.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::eq>::invoke_out));
    m.impl("eq.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::eq>::invoke));
    // eq.Scalar
    // eq.Scalar_out

    m.impl("ne.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::ne>::invoke_out));
    m.impl("ne.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::ne>::invoke));
    // ne.Scalar
    // ne.Scalar_out

    m.impl("ge.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::ge>::invoke_out));
    m.impl("ge.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::ge>::invoke));
    // ge.Scalar
    // ge.Scalar_out

    m.impl("gt.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::gt>::invoke_out));
    m.impl("gt.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::gt>::invoke));
    // gt.Scalar
    // gt.Scalar_out

    m.impl("le.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::le>::invoke_out));
    m.impl("le.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::le>::invoke));
    // le.Scalar
    // le.Scalar_out

    m.impl("lt.Tensor_out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::lt>::invoke_out));
    m.impl("lt.Tensor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::lt>::invoke));
    // lt.Scalar
    // lt.Scalar_out

    // Special ops
    m.impl("logaddexp.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logaddexp>::invoke_out));
    m.impl("logaddexp", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logaddexp>::invoke));
    m.impl("logaddexp2.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logaddexp2>::invoke_out));
    m.impl("logaddexp2", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logaddexp2>::invoke));

    // Pending TTNN binary ops to register (from ttnn_ops_grouped.txt)
    // ttnn::add_
    // ttnn::add_bw
    // ttnn::assign_bw
    // ttnn::atan2_bw
    // ttnn::bias_gelu
    // ttnn::bias_gelu_
    // ttnn::bias_gelu_bw
    // ttnn::bitwise_and
    // ttnn::bitwise_left_shift
    // ttnn::bitwise_or
    // ttnn::bitwise_right_shift
    // ttnn::bitwise_xor
    // ttnn::concat_bw
    // ttnn::div_bw
    // ttnn::div_no_nan
    // ttnn::divide_
    // ttnn::eq_
    // ttnn::floor_div
    // ttnn::fmod
    // ttnn::fmod_bw
    // ttnn::gcd
    // ttnn::ge_
    // ttnn::gt_
    // ttnn::hypot
    // ttnn::hypot_bw
    // ttnn::isclose
    // ttnn::lcm
    // ttnn::ldexp_
    // ttnn::ldexp_bw
    // ttnn::le_
    // ttnn::logaddexp2_
    // ttnn::logaddexp2_bw
    // ttnn::logaddexp_
    // ttnn::logaddexp_bw
    // ttnn::logical_and_
    // ttnn::logical_left_shift
    // ttnn::logical_or_
    // ttnn::logical_right_shift
    // ttnn::logical_xor_
    // ttnn::lt_
    // ttnn::max_bw
    // ttnn::maximum
    // ttnn::min_bw
    // ttnn::minimum
    // ttnn::mul_bw
    // ttnn::multiply_
    // ttnn::ne_
    // ttnn::nextafter
    // ttnn::outer
    // ttnn::polyval
    // ttnn::pow
    // ttnn::prelu
    // ttnn::prim::binary
    // ttnn::prim::binary_ng
    // ttnn::remainder
    // ttnn::remainder_bw
    // ttnn::rsub
    // ttnn::rsub_
    // ttnn::rsub_bw
    // ttnn::squared_difference
    // ttnn::squared_difference_
    // ttnn::sub_bw
    // ttnn::xlogy_bw

    // =========================
    // Reductions
    // =========================
    // Sum
    // sum
    // sum.DimnameList_out
    // sum.IntList_out
    // sum.dim_DimnameList
    // sum.dim_IntList
    // ttnn::sum

    // Mean
    // mean
    // mean.dim
    // mean.names_dim
    // mean.names_out
    // mean.out
    // ttnn::mean

    // Max / Min
    // max
    // max.dim
    // max.dim_max
    // max.names_dim
    // max.names_dim_max
    // ttnn::max
    // min
    // min.dim
    // min.dim_min
    // min.names_dim
    // min.names_dim_min
    // ttnn::min

    // Std / Var
    // std
    // std.dim
    // std.names_dim
    // std.names_out
    // std.out
    // std.correction
    // std.correction_out
    // std.correction_names
    // std.correction_names_out
    // ttnn::std
    // var
    // var.dim
    // var.names_dim
    // var.names_out
    // var.out
    // var.correction
    // var.correction_out
    // var.correction_names
    // var.correction_names_out
    // ttnn::var
    
    // Core tensor ops (shape/view/manipulation)
    // alias
    // align_as
    // align_tensors
    // align_to
    // align_to.ellipsis_idx
    // as_strided
    // clone
    // ttnn::clone
    // ttnn::prim::clone
    // contiguous
    // diagonal
    // diagonal.Dimname
    // narrow
    // narrow.Tensor
    // rename
    // rename_
    // reshape
    // ttnn::reshape
    // ttnn::reshape_on_device
    // resize_
    // resize_as_
    // select.Dimname
    // select.int
    // size.Dimname
    // size.int
    // slice.Tensor
    // ttnn::slice
    // squeeze
    // ttnn::squeeze
    // squeeze.dim
    // squeeze.dimname
    // stride.Dimname
    // stride.int
    // t
    // transpose.Dimname
    // transpose.int
    // ttnn::transpose
    // unbind.Dimname
    // unbind.int
    // unflatten.Dimname
    // unflatten.int
    // unsafe_chunk
    // unsafe_split.Tensor
    // ttnn::split
    // unsafe_split_with_sizes

    // Pending TTNN core tensor/data movement ops to register
    // ttnn::assign
    // ttnn::bcast
    // ttnn::chunk
    // ttnn::clone
    // ttnn::concat
    // ttnn::copy
    // ttnn::expand
    // ttnn::fill_implicit_tile_padding
    // ttnn::fill_ones_rm
    // ttnn::fill_rm
    // ttnn::gather
    // ttnn::indexed_fill
    // ttnn::interleaved_to_sharded
    // ttnn::interleaved_to_sharded_partial
    // ttnn::move
    // ttnn::nonzero
    // ttnn::pad
    // ttnn::permute
    // ttnn::prim::clone
    // ttnn::prim::fold
    // ttnn::prim::gather
    // ttnn::prim::moe_expert_token_remap
    // ttnn::prim::permute
    // ttnn::prim::scatter
    // ttnn::prim::sort
    // ttnn::prim::typecast
    // ttnn::repeat
    // ttnn::repeat_interleave
    // ttnn::reshape
    // ttnn::reshape_on_device
    // ttnn::reshard
    // ttnn::roll
    // ttnn::scatter
    // ttnn::sharded_to_interleaved
    // ttnn::sharded_to_interleaved_partial
    // ttnn::slice
    // ttnn::sort
    // ttnn::split
    // ttnn::squeeze
    // ttnn::stack
    // ttnn::tilize
    // ttnn::tilize_with_zero_padding
    // ttnn::tosa_gather
    // ttnn::tosa_scatter
    // ttnn::transpose
    // ttnn::typecast
    // ttnn::unsqueeze
    // ttnn::untilize
    // ttnn::untilize_with_unpadding
    // ttnn::view

    // Creation / like-ops
    // empty_like
    // ttnn::moreh_full_like / ttnn::full_like
    // full_like
    // ttnn::moreh_full / ttnn::full
    // ones_like
    // rand_like
    // randn_like
    // vander
    // zeros_like

    // Indexing / filling
    // copy_
    // fill_.Scalar
    // fill_.Tensor
    // index_fill.Dimname_Scalar
    // index_fill.Dimname_Tensor
    // index_fill.int_Scalar
    // index_fill.int_Tensor
    // index_fill_.Dimname_Scalar
    // index_fill_.Dimname_Tensor
    // index_fill_.int_Scalar
    // index_fill_.int_Tensor
    // masked_fill.Scalar
    // masked_fill.Tensor
    // masked_fill_.Scalar
    // masked_fill_.Tensor
    // masked_select
    // masked_select.out

    // Reductions / scans
    // all
    // any
    // cummax
    // cummax.dimname
    // cummax.dimname_out
    // cummax.out
    // cummin
    // cummin.dimname
    // cummin.dimname_out
    // cummin.out
    // cumprod
    // cumprod.dimname
    // cumprod.dimname_out
    // cumprod.out
    // cumsum
    // cumsum.dimname
    // cumsum.dimname_out
    // cumsum.out
    // kthvalue
    // kthvalue.dimname
    // kthvalue.dimname_out
    // kthvalue.values
    // logcumsumexp
    // logcumsumexp.dimname
    // logcumsumexp.dimname_out
    // logcumsumexp.out
    // logsumexp
    // logsumexp.names
    // logsumexp.names_out
    // logsumexp.out
    
    
    // median
    // median.dim
    // median.dim_values
    // median.names_dim
    // median.names_dim_values
    
    // prod
    // prod.Dimname_out
    // prod.dim_Dimname
    // prod.dim_int
    // prod.int_out
    

    // Probability / statistics
    // polygamma
    // polygamma.out
    // polygamma_
    

    // Random
    m.impl("bernoulli", TORCH_FN(tt_eager::ext::random_wrapper<ttnn::bernoulli>::invoke));
    m.impl("bernoulli.out", TORCH_FN(tt_eager::ext::random_wrapper<ttnn::bernoulli>::invoke_out));
    // bernoulli_.Tensor
    // bernoulli_.float
    // cauchy_
    // exponential_
    // geometric_
    // normal_
    // random_
    // random_.from
    // random_.to
    // uniform_

    // Pending TTNN random ops to register
    // ttnn::prim::rand
    // ttnn::prim::uniform
    // ttnn::rand
    // ttnn::uniform

    // Math helpers (clamp and friends)
    // clamp
    // clamp.Tensor
    // clamp.Tensor_out
    // clamp.out
    // clamp_
    // clamp_.Tensor
    // clamp_max
    // clamp_max.Tensor
    // clamp_max.Tensor_out
    // clamp_max.out
    // clamp_max_
    // clamp_max_.Tensor
    // clamp_min
    // clamp_min.Tensor
    // clamp_min.Tensor_out
    // clamp_min.out
    // clamp_min_
    // clamp_min_.Tensor

    // Pooling / distance
    // _cdist_forward
    // cdist
    // max_pool1d
    // max_pool1d_with_indices
    // max_pool2d
    // max_pool2d_with_indices
    // ttnn::max_pool2d / ttnn::avg_pool2d / ttnn::global_avg_pool2d
    // max_pool3d
    // max_pool3d_with_indices

    // Softmax / dropout / threshold
    // _fused_dropout
    // dropout
    // dropout_
    // native_dropout
    // softmax.Dimname
    // softmax.int
    // ttnn::softmax / ttnn::softmax_in_place
    // threshold
    // threshold.out
    // threshold_
    // ttnn::threshold / ttnn::threshold_bw
    // _sparse_log_softmax.Dimname
    // _sparse_log_softmax.int
    // _sparse_softmax.Dimname
    // _sparse_softmax.int

    // Pending TTNN normalization ops to register
    // ttnn::batch_norm
    // ttnn::group_norm
    // ttnn::layer_norm
    // ttnn::layer_norm_post_all_gather
    // ttnn::layer_norm_pre_all_gather
    // ttnn::prim::batch_norm
    // ttnn::prim::running_statistics
    // ttnn::prim::softmax
    // ttnn::rms_norm
    // ttnn::scale_causal_mask_hw_dims_softmax_in_place
    // ttnn::scale_mask_softmax
    // ttnn::scale_mask_softmax_in_place
    // ttnn::softmax
    // ttnn::softmax_in_place

    // Tensor lists / concat / split
    // cat
    // ttnn::concat
    // cat.names
    // cat.names_out
    // cat.out
    // chunk
    // ttnn::chunk
    // split.Tensor
    // split_with_sizes
    // tensor_split.indices
    // tensor_split.sections
    // tensor_split.tensor_indices_or_sections

    // Type / device / names
    // _local_scalar_dense
    // _to_copy
    // equal
    // is_coalesced
    // is_complex
    // is_floating_point
    // is_inference
    // is_nonzero
    // is_pinned
    // is_same_size
    // is_signed
    // item
    // output_nr
    // real
    // refine_names
    // result_type.Scalar
    // result_type.Scalar_Tensor
    // result_type.Tensor
    // to.device
    // to.dtype
    // to.dtype_layout
    // ttnn::to_dtype / ttnn::to_layout / ttnn::to_memory_config
}

// This macro registers helper functions associated with the ttnn_device_mode module that can be used in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("as_torch_device", &as_torch_device, "get torch device from existing ttnn device");
    m.def("get_ttnn_tensor", &get_ttnn_tensor, "open ttnn device and get torch device");
}
