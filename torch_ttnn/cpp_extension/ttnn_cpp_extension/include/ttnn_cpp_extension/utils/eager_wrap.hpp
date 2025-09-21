#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema

#include <concepts>                // std::same_as, std::convertible_to
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


//===========================
//   Concepts (C++20)
//===========================

// Non-type template parameter variant for compile-time bound TTNN ops
template <auto Op>
concept TTNNBinaryFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { Op(a, b) } -> std::convertible_to<ttnn::Tensor>;
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
//===========================
//   Invoker
//===========================

struct binary {
    // Compile-time bound operation: no runtime op parameter is passed.
    template <auto Op>
        requires TTNNBinaryFn<Op>
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out<Op>(a, b, out);
        return out;
    }

    template <auto Op>
        requires TTNNBinaryFn<Op>
    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a");
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");
        ttnn::Tensor result = Op(a_tile, b_tile);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary

// Wrappers
//===========================
//   Wrappers
//===========================

// Thin wrapper binding a compile-time TTNN op (function or stateless lambda) without storing a pointer
// Example: binary_wrapper<ttnn::add>::invoke(a, b)

template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
struct binary_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        return binary::invoke<TTNN_BINARY>(a, b);
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        return binary::invoke_out<TTNN_BINARY>(a, b, out);
    }
};

// Binary wrapper that applies scalar alpha to the second operand and then executes the binary op
// Wrapper that applies scalar alpha to the second operand before TTNN_BINARY
// Matches aten::add/sub semantics: out = a (op) (alpha * b)

template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
struct binary_with_scalar_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, alpha, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
        ttnn::Tensor a_tile = to_ttnn_tile_checked(a, "a");
        ttnn::Tensor b_tile = to_ttnn_tile_checked(b, "b");

        const double alpha_value = alpha.toDouble();
        const bool is_identity = alpha_value == 1.0;
        const ttnn::Tensor& rhs = is_identity ? b_tile : ttnn::multiply(b_tile, static_cast<float>(alpha_value));

        ttnn::Tensor result = TTNN_BINARY(a_tile, rhs);
        return write_from_ttnn(out, a, result);
    }
};


}  // namespace tt_eager::ext
