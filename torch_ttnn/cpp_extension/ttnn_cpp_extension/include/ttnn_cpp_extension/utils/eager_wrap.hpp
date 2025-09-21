#pragma once

#include <c10/util/Optional.h>
// #include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>
#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"
#include "ttnn_cpp_extension/ops/creation.hpp"
// #include <fmt/format.h>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

namespace tt_eager::ext {


template <class F>
concept TTNNBinary = requires(F f, const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { f(a, b) } -> std::same_as<ttnn::Tensor>;
};

template <class F>
concept TTNNUnary = requires(F f, const ttnn::Tensor& a) {
    { f(a) } -> std::same_as<ttnn::Tensor>;
};

// Helper functions
inline ttnn::Tensor to_ttnn_tile_checked(const at::Tensor& t, const char* arg_name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, arg_name, " must be on TTNN device");

    at::TtnnTensorImpl* impl = static_cast<at::TtnnTensorImpl*>(t.unsafeGetTensorImpl());
    auto tt = impl->get_ttnn_tensor();
    if (tt.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT);
        // tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, tt.device());
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
    template <TTNNBinary Op>
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, Op&& op) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, out, std::forward<Op>(op));
        return out;
    }

    template <TTNNBinary Op>
    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, Op&& op) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a"); // TODO: viaraible names are not correct, placeholders
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");
        auto result = op(a_tile, b_tile);

        return write_from_ttnn(out, a, result);
    }


/* Concepts version ~:
    template <TTNNBinary F>
    static at::Tensor run(const at::Tensor& a, const at::Tensor& b, F&& op) {
        at::Tensor out = make_empty_like_tt(a);
        out_(a, b, out, std::forward<F>(op));
        return out;
    }

    template <TTNNBinary F>
    static at::Tensor& out_(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, F&& op) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a");
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");
        auto result = op(a_tile, b_tile);

        return write_from_ttnn(out, a, result);
    }
*/


};  // struct binary

// Wrappers (these are not kernels, just thin wrappers over the invoker)
template <auto TTNN_BINARY>
    requires TTNNBinary<decltype(TTNN_BINARY)>
struct binary_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        return binary::invoke(a, b, TTNN_BINARY);
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        return binary::invoke_out(a, b, out, TTNN_BINARY);
    }
};

// Binary wrapper that applies scalar alpha to the second operand and then executes the binary op
template <auto TTNN_BINARY>
    requires TTNNBinary<decltype(TTNN_BINARY)>
struct binary_with_scalar_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        return binary::invoke(a, b, [&](const ttnn::Tensor& a_tile, const ttnn::Tensor& b_tile) {
            const double alpha_value = alpha.toDouble();
            if (alpha_value == 1.0) {
                return TTNN_BINARY(a_tile, b_tile);
            }
            return TTNN_BINARY(a_tile, ttnn::multiply(b_tile, static_cast<float>(alpha_value)));
        });
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
        return binary::invoke_out(a, b, out, [&](const ttnn::Tensor& a_tile, const ttnn::Tensor& b_tile) {
            const double alpha_value = alpha.toDouble();
            if (alpha_value == 1.0) {
                return TTNN_BINARY(a_tile, b_tile);
            }
            return TTNN_BINARY(a_tile, ttnn::multiply(b_tile, static_cast<float>(alpha_value)));
        });
    }
};


}  // namespace tt_eager::ext
