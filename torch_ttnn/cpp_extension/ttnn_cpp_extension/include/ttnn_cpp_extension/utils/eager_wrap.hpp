#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema

#include <concepts>
#include <optional>
#include <variant>
#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>

#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"
#include "ttnn_cpp_extension/ops/creation.hpp"
// #include <fmt/format.h>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace tt_eager::ext {


//===========================
//   Concepts (C++20)
//===========================

// Unary op concept (first)
template <auto Op>
concept TTNNUnaryFn = requires(const ttnn::Tensor& a) {
    { Op(a) } -> std::same_as<ttnn::Tensor>;
};

// Binary op concept
template <auto Op>
concept TTNNBinaryFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { Op(a, b) } -> std::same_as<ttnn::Tensor>;
};

template <auto TTNN_BINARY_ALPHA>
concept TTNNBinaryAlphaFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b, float alpha) {
    { TTNN_BINARY_ALPHA(a, b, alpha) } -> std::same_as<ttnn::Tensor>;
};

// Helper functions
inline ttnn::Tensor tileify(const at::Tensor& t) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "Tensor must be on TTNN device");

    at::TtnnTensorImpl* impl = static_cast<at::TtnnTensorImpl*>(t.unsafeGetTensorImpl());
    auto tt = impl->get_ttnn_tensor();
    if (tt.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT);
        // tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, tt.device());
    }

    return tt;
}

inline at::Tensor make_empty_like_tt(
    const at::Tensor& t,
    c10::optional<at::ScalarType> dtype_override = c10::nullopt) {
    c10::optional<at::ScalarType> dtype_opt = dtype_override.has_value()
        ? c10::optional<at::ScalarType>(*dtype_override)
        : c10::optional<at::ScalarType>(t.scalar_type());
    return tt_eager::ops::create::custom_empty_memory_format(
        t.sizes(),
        dtype_opt,
        c10::nullopt,
        c10::optional<at::Device>(t.device()),
        c10::nullopt);
}

// TODO: parameter order might be confusing, to think about
inline at::Tensor& write_from_ttnn(at::Tensor& out, const at::Tensor& like, const ttnn::Tensor& result) {
    auto* out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(like);
    out_impl->set_ttnn_tensor(result);
    return out;
}

// Uniry logic Invoker
template <auto Op>
    requires TTNNUnaryFn<Op>
struct unary_logic {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor result = Op(a_tile);
        return write_from_ttnn(out, a, result);
    }
};  // struct unary_logic

// Unary wrapper with fixed signatures for dispatcher; forwards to logic
template <auto TTNN_UNARY>
    requires TTNNUnaryFn<TTNN_UNARY>
using unary_wrapper = unary_logic<TTNN_UNARY>;

template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_logic {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        ttnn::Tensor result = Op(a_tile, b_tile);
        return write_from_ttnn(out, a, result);
    }

};  // struct binary_logic

// Thin wrapper with fixed at::Tensor signatures for dispatcher; forwards to logic
template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
using binary_wrapper = binary_logic<TTNN_BINARY>;


// Alternative wrapper that directly uses TTNN ops with explicit alpha parameter (e.g., ttnn::addalpha/subalpha)

template <auto TTNN_BINARY_ALPHA>
    requires TTNNBinaryAlphaFn<TTNN_BINARY_ALPHA>
struct binary_alpha_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, alpha, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        const float alpha_value = static_cast<float>(alpha.toDouble());
        ttnn::Tensor result = TTNN_BINARY_ALPHA(a_tile, b_tile, alpha_value);
        return write_from_ttnn(out, a, result);
    }
};


}  // namespace tt_eager::ext
