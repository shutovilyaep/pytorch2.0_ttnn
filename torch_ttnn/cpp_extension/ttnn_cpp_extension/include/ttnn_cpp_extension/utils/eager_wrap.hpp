#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema

#include <concepts>                // std::same_as, std::convertible_to
#include <type_traits>             // std::remove_cvref_t
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

// Accept either at::Tensor or ttnn::Tensor for operand adaptation
template <class T>
concept AtOrTtnnTensor =
    std::same_as<std::remove_cvref_t<T>, at::Tensor> || std::same_as<std::remove_cvref_t<T>, ttnn::Tensor>;

// Helper functions
inline ttnn::Tensor to_ttnn_tile_checked(const at::Tensor& t) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "Tensor must be on TTNN device");

    at::TtnnTensorImpl* impl = static_cast<at::TtnnTensorImpl*>(t.unsafeGetTensorImpl());
    auto tt = impl->get_ttnn_tensor();
    if (tt.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT);
        // tt = ttnn::to_layout(tt, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, tt.device());
    }

    return tt;
}

template <AtOrTtnnTensor Tens>
inline ttnn::Tensor tileify(const Tens& t) {
    if constexpr (std::same_as<std::remove_cvref_t<Tens>, at::Tensor>) {
        return to_ttnn_tile_checked(t);
    } else {
        if (t.layout() == ttnn::ROW_MAJOR_LAYOUT) {
            return ttnn::to_layout(t, ttnn::TILE_LAYOUT);
        }
        return t;
    }
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

template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_logic {
    template <AtOrTtnnTensor Tens>
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const Tens& b) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, out);
        return out;
    }

    template <AtOrTtnnTensor Tens>
    static at::Tensor& invoke_out(const at::Tensor& a, const Tens& b, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        ttnn::Tensor result = Op(a_tile, b_tile);
        return write_from_ttnn(out, a, result);
    }

    // No helpers for precomputed tiles here to keep core minimal; wrappers can adapt inputs.

    // scaling logic intentionally left out; handled in wrappers to compose with other wrappers
};  // struct binary_logic

// Wrappers
//===========================
//   Wrappers
//===========================

// Thin wrapper binding a compile-time TTNN op (function or stateless lambda) without storing a pointer
// Example: binary_wrapper<ttnn::add>::invoke(a, b)

template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
struct binary_wrapper {
    template <AtOrTtnnTensor Tens>
    static at::Tensor invoke(const at::Tensor& a, const Tens& b) {
        return binary_logic<TTNN_BINARY>::invoke(a, b);
    }

    template <AtOrTtnnTensor Tens>
    static at::Tensor& invoke_out(const at::Tensor& a, const Tens& b, at::Tensor& out) {
        return binary_logic<TTNN_BINARY>::invoke_out(a, b, out);
    }
};

// Binary wrapper that applies scalar alpha to the second operand and then executes the binary op
// Wrapper that applies scalar alpha to the second operand before TTNN_BINARY
// Matches aten::add/sub semantics: out = a (op) (alpha * b)

template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
struct binary_b_scaled_wrapper {
    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        const double alpha_value = alpha.toDouble();
        if (alpha_value == 1.0) {
            return binary_wrapper<TTNN_BINARY>::invoke(a, b);
        }
        ttnn::Tensor b_tile = tileify(b);
        return binary_wrapper<TTNN_BINARY>::invoke(a, ttnn::multiply(b_tile, static_cast<float>(alpha_value)));
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha, at::Tensor& out) {
        const double alpha_value = alpha.toDouble();
        if (alpha_value == 1.0) {
            return binary_wrapper<TTNN_BINARY>::invoke_out(a, b, out);
        }   
        ttnn::Tensor b_tile = tileify(b);
        return binary_wrapper<TTNN_BINARY>::invoke_out(a, ttnn::multiply(b_tile, static_cast<float>(alpha_value)), out);
    }
};


}  // namespace tt_eager::ext
