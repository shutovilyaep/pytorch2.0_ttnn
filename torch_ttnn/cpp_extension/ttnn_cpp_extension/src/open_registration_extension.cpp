#include <ATen/native/DispatchStub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "ttnn_cpp_extension/utils/device.hpp"

#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"
#include "ttnn_cpp_extension/core/copy.hpp"

#include "ttnn_cpp_extension/ops/creation.hpp"

#include "ttnn_cpp_extension/utils/unary_eager_register.hpp"
#include "ttnn_cpp_extension/utils/binary_eager_register.hpp"
#include "ttnn_cpp_extension/utils/random_eager_wrappers.hpp"
#include "ttnn_cpp_extension/utils/reduction_eager_register.hpp"
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
