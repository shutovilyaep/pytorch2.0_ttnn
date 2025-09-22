#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema

#include <concepts>                // std::same_as, std::convertible_to
#include <type_traits>             // std::remove_cvref_t
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

// Non-type template parameter variant for compile-time bound TTNN ops
template <auto Op>
concept TTNNBinaryFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { Op(a, b) } -> std::convertible_to<ttnn::Tensor>;
};

// Unary op concept
template <auto Op>
concept TTNNUnaryFn = requires(const ttnn::Tensor& a) {
    { Op(a) } -> std::convertible_to<ttnn::Tensor>;
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

inline at::Tensor make_empty_like_tt(const at::Tensor& t, c10::optional<at::ScalarType> dtype_override) {
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

// Invokers
//===========================
//   Invoker
//===========================

template <AtOrTtnnTensor RightT, auto Op>
    requires TTNNBinaryFn<Op>
struct binary_logic {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const RightT& b) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, b, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, const RightT& b, at::Tensor& out) {
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

// Thin wrapper with fixed at::Tensor signatures for dispatcher; forwards to logic
template <auto TTNN_BINARY>
    requires TTNNBinaryFn<TTNN_BINARY>
using binary_wrapper = binary_logic<at::Tensor, TTNN_BINARY>;

// Alternative wrapper that directly uses TTNN ops with explicit alpha parameter (e.g., ttnn::addalpha/subalpha)
template <auto TTNN_BINARY_ALPHA>
concept TTNNBinaryAlphaFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b, float alpha) {
    { TTNN_BINARY_ALPHA(a, b, alpha) } -> std::convertible_to<ttnn::Tensor>;
};

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


//===========================
//   Unary Invoker
//===========================

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

//===========================
//   Reduction Wrapper
//===========================

template <auto TTNN_REDUCTION>
struct reduction_wrapper {
    static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out(a, out);
        return out;
    }

    static at::Tensor& invoke_out(const at::Tensor& a, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor result = TTNN_REDUCTION(
            a_tile,
            std::nullopt /*dim_arg*/,
            false /*keepdim*/, std::nullopt /*mem cfg*/, std::nullopt /*kernel cfg*/);
        return write_from_ttnn(out, a, result);
    }

    // Matches schemas like: aten::std(Tensor self, bool unbiased=True) and aten::var(Tensor self, bool unbiased=True)
    static at::Tensor invoke_unbiased(const at::Tensor& a, bool unbiased) {
        at::Tensor out = make_empty_like_tt(a);
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor result = TTNN_REDUCTION(
            a_tile,
            std::nullopt /*dim_arg*/,
            false /*keepdim*/, std::nullopt /*mem cfg*/, std::nullopt /*kernel cfg*/,
            1.0f /*scalar*/, static_cast<bool>(unbiased) /*correction*/);
        return write_from_ttnn(out, a, result);
    }

    // Matches schemas like: aten::sum(Tensor self, *, ScalarType? dtype=None)
    static at::Tensor invoke_dtype(const at::Tensor& a, c10::optional<at::ScalarType> dtype) {
        at::Tensor out = make_empty_like_tt(a, dtype);
        invoke_out(a, out);
        return out;
    }

    static at::Tensor invoke_dim_IntList(const at::Tensor& a, at::IntArrayRef dims, bool keepdim = false) {
        at::Tensor out = make_empty_like_tt(a);
        invoke_out_dim_IntList(a, dims, keepdim, out);
        return out;
    }

    static at::Tensor& invoke_out_dim_IntList(
        const at::Tensor& a, at::IntArrayRef dims, bool keepdim, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);

        ttnn::SmallVector<int> reduce_dims;
        reduce_dims.reserve(dims.size());
        for (auto d : dims) {
            reduce_dims.push_back(static_cast<int>(d));
        }

        std::optional<std::variant<int, ttnn::SmallVector<int>>> dim_arg(
            std::in_place, std::in_place_index<1>, reduce_dims);

        ttnn::Tensor result = TTNN_REDUCTION(
            a_tile, dim_arg, keepdim, std::nullopt /*mem cfg*/, std::nullopt /*kernel cfg*/);
        return write_from_ttnn(out, a, result);
    }
};

}  // namespace tt_eager::ext
