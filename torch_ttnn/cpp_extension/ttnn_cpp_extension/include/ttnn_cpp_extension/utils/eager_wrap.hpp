#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema

#include <concepts>
#include <optional>
#include <c10/util/Optional.h>
#include <optional>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Generator.h>
#include <cstdint>
#include <random>

#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"
#include "ttnn_cpp_extension/ops/creation.hpp"
// #include <fmt/format.h>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace tt_eager::ext {


// Concepts
template <auto Op>
concept TTNNUnaryFn = requires(const ttnn::Tensor& a) {
    { Op(a) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNUnaryOptIntFn = requires(const ttnn::Tensor& a, std::optional<int32_t> p) {
    { Op(a, p) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNBinaryFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { Op(a, b) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNBinaryAlphaFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b, float alpha) {
    { Op(a, b, alpha) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNBinaryScalarFn = requires(const ttnn::Tensor& a, float rhs) {
    { Op(a, rhs) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNRandomFn = requires(const ttnn::Tensor& a, uint32_t seed) {
    { Op(a, seed, std::nullopt, std::nullopt, std::nullopt, std::nullopt) } -> std::same_as<ttnn::Tensor>;
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

inline at::Tensor& write_from_ttnn(at::Tensor& out, const at::Tensor& like, const ttnn::Tensor& result) {
    auto* out_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
    out_impl->set_sizes_and_strides_as(like);
    out_impl->set_ttnn_tensor(result);
    return out;
}

// Unary Wrapper
template <auto Op>
    requires TTNNUnaryFn<Op>
struct unary_wrapper {
    static_assert(TTNNUnaryFn<Op>, "Op must be ttnn::Tensor (const&) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self) {
        return invoke_into(self, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& in, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        ttnn::Tensor result = Op(a_tile);
        return write_from_ttnn(out, in, result);
    }
};  // struct unary_wrapper


// Unary with optional integer parameter (e.g., ttnn::round)
template <auto Op>
    requires TTNNUnaryOptIntFn<Op>
struct unary_opt_int_wrapper {
    static_assert(TTNNUnaryOptIntFn<Op>, "Op must be ttnn::Tensor (const&, std::optional<int32_t>) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, c10::optional<int64_t> decimals) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, decimals, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, c10::optional<int64_t> decimals) {
        return invoke_into(self, decimals, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& in,
        c10::optional<int64_t> decimals,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        std::optional<int32_t> dec_opt = decimals.has_value()
            ? std::optional<int32_t>(static_cast<int32_t>(decimals.value()))
            : std::nullopt;
        ttnn::Tensor result = Op(a_tile, dec_opt);
        return write_from_ttnn(out, in, result);
    }
};

template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_wrapper {
    static_assert(TTNNBinaryFn<Op>, "Op must be (const ttnn::Tensor&, const ttnn::Tensor&) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, b, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, const at::Tensor& other) {
        return invoke_into(self, other, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        ttnn::Tensor result = Op(a_tile, b_tile);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary_wrapper

template <auto Op>
    requires TTNNBinaryScalarFn<Op>
struct binary_scalar_wrapper {
    static_assert(TTNNBinaryScalarFn<Op>, "Op must be (const ttnn::Tensor&, float) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const c10::Scalar& other) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, other, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, const c10::Scalar& other) {
        return invoke_into(self, other, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& a,
        const c10::Scalar& other,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        const float rhs = static_cast<float>(other.toDouble());
        ttnn::Tensor result = Op(a_tile, rhs);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary_scalar_wrapper


// Alternative binary wrapper that directly uses TTNN ops with explicit alpha parameter (e.g., ttnn::addalpha/subalpha)
template <auto Op>
    requires TTNNBinaryAlphaFn<Op>
struct binary_alpha_wrapper {
    static_assert(TTNNBinaryAlphaFn<Op>, "Op must be (const ttnn::Tensor&, const ttnn::Tensor&, float) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, b, alpha, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, const at::Tensor& other, const c10::Scalar& alpha) {
        return invoke_into(self, other, alpha, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& a,
        const at::Tensor& b,
        const c10::Scalar& alpha,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        const float alpha_value = static_cast<float>(alpha.toDouble());
        ttnn::Tensor result = Op(a_tile, b_tile, alpha_value);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary_alpha_wrapper



// Random Wrapper
template <auto Op>
    requires TTNNRandomFn<Op>
struct random_wrapper {
    static_assert(TTNNRandomFn<Op>, "Op must be (const ttnn::Tensor&, uint32_t, ...) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& input, c10::optional<at::Generator> generator = c10::nullopt) {
        at::Tensor out = make_empty_like_tt(input);
        return invoke_into(input, generator, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, c10::optional<at::Generator> generator = c10::nullopt) {
        return invoke_into(self, generator, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& input,
        c10::optional<at::Generator> generator,
        at::Tensor& out) {
        ttnn::Tensor in_tile = tileify(input);

        static thread_local std::mt19937 rng(std::random_device{}());
        uint32_t seed = generator.has_value() ? static_cast<uint32_t>(generator.value().current_seed()) : rng();

        ttnn::Tensor result = Op(
            in_tile,
            seed,
            std::nullopt,              // output
            std::nullopt,              // dtype
            std::nullopt,              // memory_config
            std::nullopt               // compute_kernel_config
        );

        return write_from_ttnn(out, input, result);
    }
};  // struct random_wrapper

}  // namespace tt_eager::ext
