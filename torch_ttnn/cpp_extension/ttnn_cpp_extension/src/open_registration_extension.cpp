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
    // schema: empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    m.impl("aten::empty_strided", &tt_eager::ops::create::custom_empty_strided);
    // schema: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    m.impl("empty.memory_format", &tt_eager::ops::create::custom_empty_memory_format);
    // schema: _copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
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
    // schema: abs(Tensor self) -> Tensor
    m.impl("abs", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke));
    // schema: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("abs.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_into));
    // schema: abs_(Tensor(a!) self) -> Tensor(a!)
    m.impl("abs_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_inplace));
    // alias: absolute -> abs
    // schema: absolute(Tensor self) -> Tensor
    m.impl("absolute", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke));
    // schema: absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("absolute.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_into));
    // schema: absolute_(Tensor(a!) self) -> Tensor(a!)
    m.impl("absolute_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke_inplace));
    // schema: neg(Tensor self) -> Tensor
    m.impl("neg", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke));
    // schema: neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("neg.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke_into));
    // schema: neg_(Tensor(a!) self) -> Tensor(a!)
    m.impl("neg_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::neg>::invoke_inplace));
    // schema: reciprocal(Tensor self) -> Tensor
    m.impl("reciprocal", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke));
    // schema: reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("reciprocal.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke_into));
    // schema: reciprocal_(Tensor(a!) self) -> Tensor(a!)
    m.impl("reciprocal_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::reciprocal>::invoke_inplace));
    // schema: sqrt(Tensor self) -> Tensor
    m.impl("sqrt", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke));
    // schema: sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("sqrt.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke_into));
    // schema: sqrt_(Tensor(a!) self) -> Tensor(a!)
    m.impl("sqrt_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sqrt>::invoke_inplace));
    // schema: rsqrt(Tensor self) -> Tensor
    m.impl("rsqrt", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke));
    // schema: rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("rsqrt.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke_into));
    // schema: rsqrt_(Tensor(a!) self) -> Tensor(a!)
    m.impl("rsqrt_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rsqrt>::invoke_inplace));
    // schema: square(Tensor self) -> Tensor
    m.impl("square", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke));
    // schema: square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("square.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke_into));
    // schema: square_(Tensor(a!) self) -> Tensor(a!)
    m.impl("square_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::square>::invoke_inplace));
    // schema: sin(Tensor self) -> Tensor
    m.impl("sin", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke));
    // schema: sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("sin.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke_into));
    // schema: sin_(Tensor(a!) self) -> Tensor(a!)
    m.impl("sin_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sin>::invoke_inplace));
    // schema: cos(Tensor self) -> Tensor
    m.impl("cos", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke));
    // schema: cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("cos.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke_into));
    // schema: cos_(Tensor(a!) self) -> Tensor(a!)
    m.impl("cos_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cos>::invoke_inplace));
    // schema: tan(Tensor self) -> Tensor
    m.impl("tan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke));
    // schema: tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("tan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke_into));
    // schema: tan_(Tensor(a!) self) -> Tensor(a!)
    m.impl("tan_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tan>::invoke_inplace));
    // schema: sinh(Tensor self) -> Tensor
    m.impl("sinh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke));
    // schema: sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("sinh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke_into));
    // schema: sinh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("sinh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sinh>::invoke_inplace));
    // schema: cosh(Tensor self) -> Tensor
    m.impl("cosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke));
    // schema: cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("cosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke_into));
    // schema: cosh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("cosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::cosh>::invoke_inplace));
    // schema: tanh(Tensor self) -> Tensor
    m.impl("tanh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke));
    // schema: tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("tanh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke_into));
    // schema: tanh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("tanh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::tanh>::invoke_inplace));
    // schema: floor(Tensor self) -> Tensor
    m.impl("floor", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke));
    // schema: floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("floor.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke_into));
    // schema: floor_(Tensor(a!) self) -> Tensor(a!)
    m.impl("floor_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::floor>::invoke_inplace));
    // schema: ceil(Tensor self) -> Tensor
    m.impl("ceil", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke));
    // schema: ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("ceil.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke_into));
    // schema: ceil_(Tensor(a!) self) -> Tensor(a!)
    m.impl("ceil_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::ceil>::invoke_inplace));
    // schema: trunc(Tensor self) -> Tensor
    m.impl("trunc", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke));
    // schema: trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("trunc.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke_into));
    // schema: trunc_(Tensor(a!) self) -> Tensor(a!)
    m.impl("trunc_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::trunc>::invoke_inplace));
    // schema: frac(Tensor self) -> Tensor
    m.impl("frac", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke));
    // schema: frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("frac.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke_into));
    // schema: frac_(Tensor(a!) self) -> Tensor(a!)
    m.impl("frac_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::frac>::invoke_inplace));
    // schema: bitwise_not(Tensor self) -> Tensor
    m.impl("bitwise_not", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke));
    // schema: bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("bitwise_not.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke_into));
    // schema: bitwise_not_(Tensor(a!) self) -> Tensor(a!)
    m.impl("bitwise_not_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::bitwise_not>::invoke_inplace));
    // schema: logical_not(Tensor self) -> Tensor
    m.impl("logical_not", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke));
    // schema: logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("logical_not.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke_into));
    // schema: logical_not_(Tensor(a!) self) -> Tensor(a!)
    m.impl("logical_not_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::logical_not>::invoke_inplace));
    // schema: sign(Tensor self) -> Tensor
    m.impl("sign", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke));
    // schema: sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("sign.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke_into));
    // schema: sign_(Tensor(a!) self) -> Tensor(a!)
    m.impl("sign_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sign>::invoke_inplace));
    // schema: signbit(Tensor self) -> Tensor
    m.impl("signbit", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::signbit>::invoke));
    // schema: signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("signbit.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::signbit>::invoke_into));
    // schema: i0(Tensor self) -> Tensor
    m.impl("i0", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke));
    // schema: i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("i0.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke_into));
    // schema: i0_(Tensor(a!) self) -> Tensor(a!)
    m.impl("i0_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::i0>::invoke_inplace));
    // schema: erf(Tensor self) -> Tensor
    m.impl("erf", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke));
    // schema: erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("erf.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke_into));
    // schema: erf_(Tensor(a!) self) -> Tensor(a!)
    m.impl("erf_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erf>::invoke_inplace));
    // schema: erfc(Tensor self) -> Tensor
    m.impl("erfc", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke));
    // schema: erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("erfc.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke_into));
    // schema: erfc_(Tensor(a!) self) -> Tensor(a!)
    m.impl("erfc_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfc>::invoke_inplace));
    // schema: erfinv(Tensor self) -> Tensor
    m.impl("erfinv", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke));
    // schema: erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("erfinv.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke_into));
    // schema: erfinv_(Tensor(a!) self) -> Tensor(a!)
    m.impl("erfinv_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::erfinv>::invoke_inplace));
    // schema: exp(Tensor self) -> Tensor
    m.impl("exp", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke));
    // schema: exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("exp.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke_into));
    // schema: exp_(Tensor(a!) self) -> Tensor(a!)
    m.impl("exp_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::exp>::invoke_inplace));
    // schema: log(Tensor self) -> Tensor
    m.impl("log", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke));
    // schema: log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("log.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke_into));
    // schema: log_(Tensor(a!) self) -> Tensor(a!)
    m.impl("log_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log>::invoke_inplace));
    // schema: log10(Tensor self) -> Tensor
    m.impl("log10", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke));
    // schema: log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("log10.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke_into));
    // schema: log10_(Tensor(a!) self) -> Tensor(a!)
    m.impl("log10_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log10>::invoke_inplace));
    // schema: log2(Tensor self) -> Tensor
    m.impl("log2", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke));
    // schema: log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("log2.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke_into));
    // schema: log2_(Tensor(a!) self) -> Tensor(a!)
    m.impl("log2_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log2>::invoke_inplace));
    // schema: log1p(Tensor self) -> Tensor
    m.impl("log1p", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke));
    // schema: log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("log1p.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke_into));
    // schema: log1p_(Tensor(a!) self) -> Tensor(a!)
    m.impl("log1p_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::log1p>::invoke_inplace));
    // schema: acos(Tensor self) -> Tensor
    m.impl("acos", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke));
    // schema: acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("acos.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke_into));
    // schema: acos_(Tensor(a!) self) -> Tensor(a!)
    m.impl("acos_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acos>::invoke_inplace));
    // schema: acosh(Tensor self) -> Tensor
    m.impl("acosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke));
    // schema: acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("acosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_into));
    // schema: acosh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("acosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_inplace));
    // schema: angle(Tensor self) -> Tensor
    m.impl("angle", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke)); // TODO: check
    // schema: angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("angle.out", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke_into)); // TODO: check
    m.impl("angle_", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::angle>::invoke_inplace)); // TODO: check
    // alias: arccosh -> acosh
    // schema: arccosh(Tensor self) -> Tensor
    m.impl("arccosh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke));
    // schema: arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("arccosh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_into));
    // schema: arccosh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("arccosh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::acosh>::invoke_inplace));
    // schema: asin(Tensor self) -> Tensor
    m.impl("asin", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke));
    // schema: asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("asin.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke_into));
    // schema: asin_(Tensor(a!) self) -> Tensor(a!)
    m.impl("asin_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asin>::invoke_inplace));
    
    // schema: asinh(Tensor self) -> Tensor
    m.impl("asinh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke));
    // schema: asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("asinh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke_into));
    // schema: asinh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("asinh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::asinh>::invoke_inplace));
    // schema: atan(Tensor self) -> Tensor
    m.impl("atan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke));
    // schema: atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("atan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke_into));
    // schema: atan_(Tensor(a!) self) -> Tensor(a!)
    m.impl("atan_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atan>::invoke_inplace));
    
    // schema: atanh(Tensor self) -> Tensor
    m.impl("atanh", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke));
    // schema: atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("atanh.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke_into));
    // schema: atanh_(Tensor(a!) self) -> Tensor(a!)
    m.impl("atanh_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::atanh>::invoke_inplace));
    // schema: conj(Tensor(a) self) -> Tensor(a)
    m.impl("conj", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke));
    m.impl("conj.out", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke_into));
    m.impl("conj_", TORCH_FN(tt_eager::ext::complex_unary_from_real<ttnn::conj>::invoke_inplace));
    // ttnn::conj_bw
    // schema: deg2rad(Tensor self) -> Tensor
    m.impl("deg2rad", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke));
    // schema: deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("deg2rad.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke_into));
    // schema: deg2rad_(Tensor(a!) self) -> Tensor(a!)
    m.impl("deg2rad_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::deg2rad>::invoke_inplace));
    
    // schema: digamma(Tensor self) -> Tensor
    m.impl("digamma", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke));
    // schema: digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("digamma.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke_into));
    // schema: digamma_(Tensor(a!) self) -> Tensor(a!)
    m.impl("digamma_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::digamma>::invoke_inplace));
    
    // schema: expm1(Tensor self) -> Tensor
    m.impl("expm1", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke));
    // schema: expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("expm1.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke_into));
    // schema: expm1_(Tensor(a!) self) -> Tensor(a!)
    m.impl("expm1_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::expm1>::invoke_inplace));
    
    // imag
    // schema: isfinite(Tensor self) -> Tensor
    m.impl("isfinite", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isfinite>::invoke));
    m.impl("isfinite.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isfinite>::invoke_into));
    // schema: isinf(Tensor self) -> Tensor
    m.impl("isinf", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isinf>::invoke));
    m.impl("isinf.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isinf>::invoke_into));
    // schema: isnan(Tensor self) -> Tensor
    m.impl("isnan", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isnan>::invoke));
    m.impl("isnan.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::isnan>::invoke_into));
    
    // schema: lgamma(Tensor self) -> Tensor
    m.impl("lgamma", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke));
    // schema: lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("lgamma.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke_into));
    // schema: lgamma_(Tensor(a!) self) -> Tensor(a!)
    m.impl("lgamma_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::lgamma>::invoke_inplace));
    
    // schema: rad2deg(Tensor self) -> Tensor
    m.impl("rad2deg", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke));
    // schema: rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("rad2deg.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke_into));
    // schema: rad2deg_(Tensor(a!) self) -> Tensor(a!)
    m.impl("rad2deg_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::rad2deg>::invoke_inplace));
    
    
    // schema: relu(Tensor self) -> Tensor
    m.impl("relu", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::relu>::invoke));
    // schema: relu_(Tensor(a!) self) -> Tensor(a!)
    m.impl("relu_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::relu>::invoke_inplace));
    // real

    // schema: round(Tensor self) -> Tensor
    m.impl("round", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke));
    // schema: round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("round.out", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke_into));
    // schema: round.decimals(Tensor self, *, int decimals) -> Tensor
    m.impl("round.decimals", TORCH_FN(tt_eager::ext::unary_tensor_opt_int<ttnn::round>::invoke_decimals));
    // schema: round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
    m.impl("round.decimals_out", TORCH_FN(tt_eager::ext::unary_tensor_opt_int<ttnn::round>::invoke_decimals_into));
    // schema: round_(Tensor(a!) self) -> Tensor(a!)
    m.impl("round_", TORCH_FN(tt_eager::ext::unary_tensor_opt_int_none<ttnn::round>::invoke_inplace));
    // ttnn::round_bw
    // schema: sigmoid(Tensor self) -> Tensor
    m.impl("sigmoid", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke));
    // schema: sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("sigmoid.out", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke_into));
    // schema: sigmoid_(Tensor(a!) self) -> Tensor(a!)
    m.impl("sigmoid_", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::sigmoid>::invoke_inplace));
    
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
    m.impl("sub_.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::subtract_>::invoke_inplace));
    // schema: sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    m.impl("sub_.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::subalpha>::invoke_inplace));
    // rsub: reverse subtract
    // rsub.Tensor: rsub(self, other, alpha) = other - alpha*self
    // rsub.Scalar: rsub(self, other, alpha) = other - alpha*self
    // schema: rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    m.impl("rsub.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha_swapped<ttnn::subalpha>::invoke)); // TODO: to check
    // schema: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    m.impl("rsub.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_float_with_alpha_adapter<ttnn::rsub>::invoke)); // TODO: to check

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
    // schema: sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    m.impl("sum.IntList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_into));
    // schema: sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum.dim_DimnameList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames));
    // schema: sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    m.impl("sum.DimnameList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames_into));

    // Mean
    // schema: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean", TORCH_FN(tt_eager::ext::reduction_all<ttnn::mean>::invoke));
    // schema: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke));
    // schema: mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    m.impl("mean.out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_into));
    // schema: mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames));
    // schema: mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
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
    // schema: max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.dim_max", TORCH_FN(MaxPair::invoke_into));
    // schema: max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("max.names_dim", TORCH_FN(MaxPair::invoke_dimname));
    // schema: max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.names_dim_max", TORCH_FN(MaxPair::invoke_dimname_into));

    using MinPair = tt_eager::ext::reduction_dim_pair<ttnn::min, ttnn::experimental::argmin>;
    // schema: min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("min.dim", TORCH_FN(MinPair::invoke));
    // schema: min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("min.dim_min", TORCH_FN(MinPair::invoke_into));
    // schema: min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("min.names_dim", TORCH_FN(MinPair::invoke_dimname));
    // schema: min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("min.names_dim_min", TORCH_FN(MinPair::invoke_dimname_into));

    // Std / Var
    // Base (all-elements) with unbiased flag default (correction)
    // schema: var(Tensor self, bool unbiased=True) -> Tensor
    m.impl("var", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::var>::invoke));
    // schema: std(Tensor self, bool unbiased=True) -> Tensor
    m.impl("std", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::std>::invoke));

    // schema: var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("var.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::var>::invoke_into));
    // schema: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("var.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke));
    // schema: var.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("var.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames));
    // schema: var.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
    m.impl("var.correction_names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames_into));
    // schema: var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
    m.impl("var.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_into));
    // schema: var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke));
    // schema: var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames));
    // schema: var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("var.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames_into));


    // schema: std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    m.impl("std.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::std>::invoke_into));
    // schema: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("std.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke));
    // schema: std.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("std.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames));
    // schema: std.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
    m.impl("std.correction_names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames_into));
    // schema: std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
    m.impl("std.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_into));
    // schema: std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke));
    // schema: std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames));
    // schema: std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
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
