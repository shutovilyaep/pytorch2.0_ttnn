#pragma once

#include <c10/util/Optional.h>
// #include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
// #include <fmt/format.h>
// #include <ttnn/operations/core/core.hpp>

namespace tt_eager::ext {


/* TODO: Switch to concepts

template <class F>
concept TNNBinary = requires(F f, const at::Tensor& a, const at::Tensor& b) {
    { f(a, b) } -> std::same_as<ttnn::Tensor>;
};

template <class F>
concept TNNUnary = requires(F f, const at::Tensor& a) {
    { f(a) } -> std::same_as<ttnn::Tensor>;
};

*/

// Helper functions
inline ttnn::Tensor to_ttnn_tile_checked(const at::Tensor& t, const char* arg_name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, arg_name, " must be on TTNN device");

    at::TtnnTensorImpl* impl = static_cast<at::TtnnTensorImpl*>(t.unsafeGetTensorImpl());
    auto tt = impl->get_ttnn_tensor();
    if (tt.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, tt.device());
    }

    return tt;
}

inline at::Tensor make_empty_like_tt(const at::Tensor& t) {
    return tt_eager::ops::create::custom_empty_memory_format(
        t.sizes(),
        c10::optional<at::ScalarType>(t.scalar_type()),
        c10::nullopt,  // layout
        c10::optional<at::Device>(t.device()),
        c10::nullopt  // pin_memory
    );
}

// TODO: parameter order might be confusing, to think about
inline at::Tensor& write_from_ttnn(at::Tensor& out, const at::Tensor& like, const ttnn::Tensor& result) {
    auto* out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(like);
    out_impl->set_ttnn_tensor(result);
    return out;
}

// Invokers
struct binary {
    template <class TNNBinary>
    static at::Tensor run(const at::Tensor& a, const at::Tensor& b, TNNBinary&& op) {
        at::Tensor out = make_empty_like_tt(a);
        out_(a, b, out, std::forward<TNNBinary>(op));
        return out;
    }

    template <class TNNBinary>
    static at::Tensor& out_(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, TNNBinary&& op) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a");
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");
        auto result = op(a_tile, b_tile);

        return write_from_ttnn(out, a, result);
    }


/* Concepts version ~:
    template <TNNBinary F>
    static at::Tensor run(const at::Tensor& a, const at::Tensor& b, F&& op) {
        at::Tensor out = make_empty_like_tt(a);
        out_(a, b, out, std::forward<F>(op));
        return out;
    }

    template <TNNBinary F>
    static at::Tensor& out_(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, F&& op) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a");
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");
        auto result = op(a_tile, b_tile);

        return write_from_ttnn(out, a, result);
    }
*/


};  // namespace binary

// "Kernel" generators (TODO: not kernels, just a wrapper for the invoker)
template <auto TTNN_BINARY>
struct binary_kernel {
    static at::Tensor func(const at::Tensor& a, const at::Tensor& b) {  // TODO: scalar alpha?
        return binary::run(a, b, TTNN_BINARY);
    }

    static at::Tensor& func_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        return binary::out_(a, b, out, TTNN_BINARY);
    }
};


}  // namespace tt_eager::ext
