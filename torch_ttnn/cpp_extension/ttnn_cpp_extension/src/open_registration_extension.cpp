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
    // acos
    // acos.out
    // acos_
    // acosh
    // acosh.out
    // acosh_
    // angle
    // angle.out
    // arccosh
    // arccosh.out
    // arccosh_
    // asin
    // asin.out
    // asin_
    // asinh
    // asinh.out
    // asinh_
    // atan
    // atan.out
    // atan2
    // atan2.out
    // atan2_
    // atan_
    // atanh
    // atanh.out
    // atanh_
    // conj
    // deg2rad
    // deg2rad.out
    // deg2rad_
    // digamma
    // digamma.out
    // digamma_
    // expm1
    // expm1.out
    // expm1_
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
    // rad2deg
    // rad2deg.out
    // rad2deg_
    // relu
    // relu_
    // real
    // round
    // round.out
    // round_
    // sigmoid
    // sigmoid.out
    // sigmoid_

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
    // true_divide.Scalar
    // true_divide.out
    // true_divide_.Scalar
    // true_divide_.Tensor
    // pow.Scalar
    // pow.Scalar_out
    // pow.Tensor_Scalar
    // pow.Tensor_Scalar_out
    // pow.Tensor_Tensor
    // pow.Tensor_Tensor_out
    // pow_.Scalar
    // pow_.Tensor
    // nextafter
    // nextafter.out
    // nextafter_
    // dot
    // dot.out
    // hypot
    // hypot.out
    // hypot_
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

    m.impl("logical_or.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_or>::invoke_out));
    m.impl("logical_or", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_or>::invoke));
    // logical_or_

    m.impl("logical_xor.out", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_xor>::invoke_out));
    m.impl("logical_xor", TORCH_FN(tt_eager::ext::binary_wrapper<ttnn::logical_xor>::invoke));
    // logical_xor_

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

    // =========================
    // Reductions
    // =========================
    // Sum
    // sum
    // sum.DimnameList_out
    // sum.IntList_out
    // sum.dim_DimnameList
    // sum.dim_IntList

    // Mean
    // mean
    // mean.dim
    // mean.names_dim
    // mean.names_out
    // mean.out

    // Max / Min
    // max
    // max.dim
    // max.dim_max
    // max.names_dim
    // max.names_dim_max
    // min
    // min.dim
    // min.dim_min
    // min.names_dim
    // min.names_dim_min

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
    // std_mean
    // std_mean.dim
    // std_mean.names_dim
    // std_mean.correction
    // std_mean.correction_names
    // var
    // var.dim
    // var.names_dim
    // var.names_out
    // var.out
    // var.correction
    // var.correction_out
    // var.correction_names
    // var.correction_names_out
    // var_mean
    // var_mean.dim
    // var_mean.names_dim
    // var_mean.correction
    // var_mean.correction_names
    
    // Core tensor ops (shape/view/manipulation)
    // alias
    // align_as
    // align_tensors
    // align_to
    // align_to.ellipsis_idx
    // as_strided
    // clone
    // contiguous
    // diagonal
    // diagonal.Dimname
    // narrow
    // narrow.Tensor
    // rename
    // rename_
    // reshape
    // resize_
    // resize_as_
    // select.Dimname
    // select.int
    // size.Dimname
    // size.int
    // slice.Tensor
    // squeeze
    // squeeze.dim
    // squeeze.dimname
    // stride.Dimname
    // stride.int
    // t
    // transpose.Dimname
    // transpose.int
    // unbind.Dimname
    // unbind.int
    // unflatten.Dimname
    // unflatten.int
    // unsafe_chunk
    // unsafe_split.Tensor
    // unsafe_split_with_sizes

    // Creation / like-ops
    // empty_like
    // full_like
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
    // max_pool3d
    // max_pool3d_with_indices

    // Softmax / dropout / threshold
    // _fused_dropout
    // dropout
    // dropout_
    // native_dropout
    // softmax.Dimname
    // softmax.int
    // threshold
    // threshold.out
    // threshold_
    // _sparse_log_softmax.Dimname
    // _sparse_log_softmax.int
    // _sparse_softmax.Dimname
    // _sparse_softmax.int

    // Tensor lists / concat / split
    // cat
    // cat.names
    // cat.names_out
    // cat.out
    // chunk
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

    // TODO: to add other ops here.
    // FUTURETODO: to generate this part via CMake
}

// This macro registers helper functions associated with the ttnn_device_mode module that can be used in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("as_torch_device", &as_torch_device, "get torch device from existing ttnn device");
    m.def("get_ttnn_tensor", &get_ttnn_tensor, "open ttnn device and get torch device");
}
