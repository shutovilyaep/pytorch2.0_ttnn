#pragma once

// PR-ready: clean C++20 wrappers around TTNN binary ops for PyTorch dispatch
// - Uses concepts to validate op signatures at compile time
// - Avoids passing raw function pointers for TTNN ops; binds ops as NTTPs
// - Provides out/inplace-style invokers compatible with aten schema
//
// Guide to wrappers naming and usage (short version):
// - All wrappers expose: invoke(...), invoke_into(..., out), invoke_inplace(self, ...)
// - Common behavior: tileify inputs to TTNN TILE layout, write result back to at::Tensor
// - Quick chooser:
//   * unary_tensor<Op>                         → Op(a)
//   * unary_tensor_opt_int_none<Op>            → Op(a, nullopt)
//   * unary_tensor_opt_int<Op>                 → Op(a, optional<int>(v))
//   * unary_tensor_int<Op>                     → Op(a, int32_t)
//   * complex_unary_from_real<Op>              → Op(Complex{real=a, imag=0})
//   * unary_random_seeded<Op>                  → Op(a, seed, ...nullopts)
//   * binary_tensor_tensor<Op>                 → Op(a, b)
//   * binary_tensor_float<Op>                  → Op(a, float)
//   * binary_tensor_tensor_alpha<Op>           → Op(a, b, float alpha)
//   * binary_tensor_scalar_as_tensor<Op>       → materialize scalar to rhs tensor like a; Op(a, rhs_tt)
//   * binary_scalar_tensor_as_tensor<Op>       → materialize scalar to lhs tensor like exponent; Op(base_tt, exponent)
//   * binary_tensor_tensor_outlike<Op>         → Op(a, b, nullopt, nullopt, nullopt, nullopt)
//
// Examples of registrations (see src/open_registration_extension.cpp):
//   m.impl("abs", TORCH_FN(tt_eager::ext::unary_tensor<ttnn::abs>::invoke));
//   m.impl("mul.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor<ttnn::multiply>::invoke));
//   m.impl("eq.Scalar", TORCH_FN(tt_eager::ext::binary_tensor_scalar_as_tensor<ttnn::eq>::invoke));
//   m.impl("pow.Scalar", TORCH_FN(tt_eager::ext::binary_scalar_tensor_as_tensor<ttnn::pow>::invoke));
//   m.impl("add.Tensor", TORCH_FN(tt_eager::ext::binary_tensor_tensor_alpha<ttnn::addalpha>::invoke));
//   m.impl("dot", TORCH_FN(tt_eager::ext::binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke));

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
#include <type_traits>

#include "ttnn_cpp_extension/core/TtnnTensorImpl.hpp"
#include "ttnn_cpp_extension/ops/creation.hpp"
// #include <fmt/format.h>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/eltwise/complex/complex.hpp>
#include <ttnn/types.hpp>

namespace tt_eager::ext {


// Concepts
template <auto Op>
concept TTNNUnaryFn = requires(const ttnn::Tensor& a) {
    { Op(a) } -> std::same_as<ttnn::Tensor>;
};

// (removed misplaced unary_from_binary_zero_first_wrapper; see correct placement below)

 

template <auto Op>
concept TTNNUnaryOptIntFn = requires(const ttnn::Tensor& a, std::optional<int32_t> p) {
    { Op(a, p) } -> std::same_as<ttnn::Tensor>;
};

template <auto Op>
concept TTNNUnaryIntFn = requires(const ttnn::Tensor& a, int32_t p) {
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

template <auto Op>
concept TTNNBinaryOutLikeFn = requires(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    { Op(a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt) } -> std::same_as<ttnn::Tensor>;
};

// no concept for complex unary; we'll probe callability in-body with if constexpr

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

// Unary: expects Op(a) → ttnn::Tensor
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&)
// - Example: m.impl("abs", TORCH_FN(unary_tensor<ttnn::abs>::invoke))
// - Used by aten ops (examples): abs, neg, reciprocal, sqrt, rsqrt, square,
//   sin, cos, tan, sinh, cosh, tanh, floor, ceil, trunc, frac, bitwise_not,
//   logical_not, sign, signbit, i0, erf, erfc, erfinv, exp, log, log10,
//   log2, log1p, acos, acosh, asin, asinh, atan, atanh, deg2rad, digamma,
//   expm1, isfinite, isinf, isnan, lgamma, rad2deg, relu, sigmoid
template <auto Op>
    requires TTNNUnaryFn<Op>
struct unary_tensor {
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
};  // struct unary_tensor



// Scalar base ^ Tensor exponent adapter for any TTNN binary op (e.g., ttnn::pow)
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, const ttnn::Tensor&)
// - Behavior: materialize scalar base to tensor like exponent; call Op(base_tt, exponent)
// - Example: m.impl("pow.Scalar", TORCH_FN(binary_scalar_tensor_as_tensor<ttnn::pow>::invoke))
// - Used by aten ops (examples): pow.Scalar, pow.Scalar_out
template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_scalar_tensor_as_tensor {
    static_assert(TTNNBinaryFn<Op>, "Op must be (const ttnn::Tensor&, const ttnn::Tensor&) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const c10::Scalar& base, const at::Tensor& exponent) {
        at::Tensor out = make_empty_like_tt(exponent);
        return invoke_into(base, exponent, out);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const c10::Scalar& base,
        const at::Tensor& exponent,
        at::Tensor& out) {
        ttnn::Tensor exp_tt = tileify(exponent);
        // Build base tensor with same shape as exponent, filled with scalar base
        ttnn::Tensor zero = ttnn::multiply(exp_tt, 0.0f);
        const float base_f = static_cast<float>(base.toDouble());
        ttnn::Tensor base_tt = ttnn::add(zero, base_f);
        ttnn::Tensor result = Op(base_tt, exp_tt);
        return write_from_ttnn(out, exponent, result);
    }
};

// Complex-unary wrapper: builds ComplexTensor from real input (imag = 0) and calls a TTNN complex op
// - Expected Op signature: Complex unary (ttnn::operations::complex::ComplexTensor, ...)
// - Behavior: make Complex{real=a, imag=0}; prefer L1 memory
// - Example: m.impl("angle", TORCH_FN(complex_unary_from_real<ttnn::angle>::invoke))
// - Used by aten ops (examples): angle, angle.out, angle_, conj, conj.out, conj_
template <auto Op>
struct complex_unary_from_real {

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self) {
        return invoke_into(self, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& in, at::Tensor& out) {
        ttnn::Tensor real_tt = tileify(in);
        ttnn::Tensor zero_tt = ttnn::multiply(real_tt, 0.0f);
        ttnn::operations::complex::ComplexTensor ct({real_tt, zero_tt});
        // Prefer L1 memory for small unary outputs; fall back if needed
        auto ret = Op(ct, ttnn::L1_MEMORY_CONFIG);
        using ReturnT = decltype(ret);
        if constexpr (std::is_same_v<ReturnT, ttnn::Tensor>) {
            return write_from_ttnn(out, in, ret);
        } else {
            // For ops like ttnn::conj returning ComplexTensor, take real part for real input
            ttnn::Tensor real_tt_out = ret.real();
            return write_from_ttnn(out, in, real_tt_out);
        }
    }
};


// Tensor-Scalar adapter that materializes a scalar as a Tensor like `a` and applies a binary TTNN op
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, const ttnn::Tensor&)
// - Behavior: materialize scalar to rhs tensor shaped like `a`; call Op(a, rhs_tt)
// - Example: m.impl("eq.Scalar", TORCH_FN(binary_tensor_scalar_as_tensor<ttnn::eq>::invoke))
// - Used by aten ops (examples): eq.Scalar, eq.Scalar_out, ne.Scalar, ne.Scalar_out,
//   ge.Scalar, ge.Scalar_out, gt.Scalar, gt.Scalar_out, le.Scalar, le.Scalar_out,
//   lt.Scalar, lt.Scalar_out
template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_tensor_scalar_as_tensor {
    static_assert(TTNNBinaryFn<Op>, "Op must be (const ttnn::Tensor&, const ttnn::Tensor&) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const c10::Scalar& other) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, other, out);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& a,
        const c10::Scalar& other,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        // Build rhs tensor with same shape as `a`, filled with scalar `other`
        ttnn::Tensor zero = ttnn::multiply(a_tile, 0.0f);
        const float rhs_f = static_cast<float>(other.toDouble());
        ttnn::Tensor rhs_tt = ttnn::add(zero, rhs_f);
        ttnn::Tensor result = Op(a_tile, rhs_tt);
        return write_from_ttnn(out, a, result);
    }
};


// Binary Tensor-Scalar adapter that matches aten *Scalar signature with alpha parameter
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, float)
// - Behavior: multiply scalar by alpha; call Op(a, rhs_float)
// - Example: m.impl("add.Scalar", TORCH_FN(binary_tensor_float_with_alpha_adapter<ttnn::add>::invoke))
// - Used by aten ops (examples): add.Scalar, add_.Scalar, sub.Scalar, sub_.Scalar
template <auto Op>
    requires TTNNBinaryScalarFn<Op>
struct binary_tensor_float_with_alpha_adapter {
    static_assert(TTNNBinaryScalarFn<Op>, "Op must be (const ttnn::Tensor&, float) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const c10::Scalar& other, const c10::Scalar& alpha) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, other, alpha, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, const c10::Scalar& other, const c10::Scalar& alpha) {
        return invoke_into(self, other, alpha, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& a,
        const c10::Scalar& other,
        const c10::Scalar& alpha,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        const float rhs = static_cast<float>(other.toDouble()) * static_cast<float>(alpha.toDouble());
        ttnn::Tensor result = Op(a_tile, rhs);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary_tensor_float_with_alpha_adapter


// Unary wrappers for optional integer parameter (e.g., ttnn::round)
// No-argument variant
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, std::optional<int32_t>)
// - Behavior: pass std::nullopt for the integer parameter
// - Example: m.impl("round", TORCH_FN(unary_tensor_opt_int_none<ttnn::round>::invoke))
// - Used by aten ops (examples): round, round.out, round_
template <auto Op>
    requires TTNNUnaryOptIntFn<Op>
struct unary_tensor_opt_int_none {
    static_assert(TTNNUnaryOptIntFn<Op>, "Op must be ttnn::Tensor (const&, std::optional<int32_t>) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self) {
        return invoke_into(self, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& in,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        ttnn::Tensor result = Op(a_tile, std::nullopt);
        return write_from_ttnn(out, in, result);
    }
};

// Decimals-param variant
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, std::optional<int32_t>)
// - Behavior: pass provided decimals as optional<int32_t>
// - Example: m.impl("round.decimals", TORCH_FN(unary_tensor_opt_int<ttnn::round>::invoke_decimals))
// - Used by aten ops (examples): round.decimals, round.decimals_out
// Unary with required int parameter
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, int32_t)
// - Example: m.impl("pow.Tensor_Scalar", TORCH_FN(unary_tensor_int<ttnn::power>::invoke))
// - Used by aten ops (examples): pow.Tensor_Scalar, pow.Tensor_Scalar_out, pow_.Scalar
// Binary tensor-tensor
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, const ttnn::Tensor&)
// - Example: m.impl("mul.Tensor", TORCH_FN(binary_tensor_tensor<ttnn::multiply>::invoke))
// - Used by aten ops (examples): mul.Tensor, mul.out, mul_.Tensor, div.Tensor, div.out, div_.Tensor,
//   pow_.Tensor, nextafter, hypot, logical_and, logical_and_, logical_or, logical_or_, logical_xor,
//   logical_xor_, atan2, atan2.out, atan2_, eq.Tensor, eq.Tensor_out, ne.Tensor, ne.Tensor_out,
//   ge.Tensor, ge.Tensor_out, gt.Tensor, gt.Tensor_out, le.Tensor, le.Tensor_out, lt.Tensor,
//   lt.Tensor_out, logaddexp, logaddexp2
// Binary tensor-float
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, float)
// - Example: m.impl("div.Scalar", TORCH_FN(binary_tensor_float<ttnn::divide>::invoke))
// - Used by aten ops (examples): div.Scalar, div_.Scalar
template <auto Op>
    requires TTNNUnaryOptIntFn<Op>
struct unary_tensor_opt_int {
    static_assert(TTNNUnaryOptIntFn<Op>, "Op must be ttnn::Tensor (const&, std::optional<int32_t>) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke_decimals(const at::Tensor& a, int64_t decimals) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_decimals_into(a, decimals, out);
    }

    [[nodiscard]] static at::Tensor& invoke_decimals_inplace(at::Tensor& self, int64_t decimals) {
        return invoke_decimals_into(self, decimals, self);
    }

    [[nodiscard]] static at::Tensor& invoke_decimals_into(
        const at::Tensor& in,
        int64_t decimals,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        std::optional<int32_t> dec_opt = std::optional<int32_t>(static_cast<int32_t>(decimals));
        ttnn::Tensor result = Op(a_tile, dec_opt);
        return write_from_ttnn(out, in, result);
    }
};

template <auto Op>
    requires TTNNUnaryIntFn<Op>
struct unary_tensor_int {
    static_assert(TTNNUnaryIntFn<Op>, "Op must be ttnn::Tensor (const&, int32_t) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const c10::Scalar& value) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, value, out);
    }

    [[nodiscard]] static at::Tensor& invoke_inplace(at::Tensor& self, const c10::Scalar& value) {
        return invoke_into(self, value, self);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& in,
        const c10::Scalar& value,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        int32_t v = static_cast<int32_t>(value.toLong());
        ttnn::Tensor result = Op(a_tile, v);
        return write_from_ttnn(out, in, result);
    }
}; // struct unary_tensor_int

template <auto Op>
    requires TTNNBinaryFn<Op>
struct binary_tensor_tensor {
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
};  // struct binary_tensor_tensor

template <auto Op>
    requires TTNNBinaryScalarFn<Op>
struct binary_tensor_float {
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
};  // struct binary_tensor_float


// Alternative binary wrapper that directly uses TTNN ops with explicit alpha parameter (e.g., ttnn::addalpha/subalpha)
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, const ttnn::Tensor&, float)
// - Example: m.impl("add.Tensor", TORCH_FN(binary_tensor_tensor_alpha<ttnn::addalpha>::invoke))
// - Used by aten ops (examples): add.Tensor, add.out, add_.Tensor, sub.Tensor, sub.out, sub_.Tensor
template <auto Op>
    requires TTNNBinaryAlphaFn<Op>
struct binary_tensor_tensor_alpha {
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
};  // struct binary_tensor_tensor_alpha



// Random Wrapper
// - Expected Op signature: ttnn::Tensor (const ttnn::Tensor&, uint32_t, ...)
// - Behavior: derive seed from Generator or RNG; pass other args as nullopt
// - Example: m.impl("bernoulli", TORCH_FN(unary_random_seeded<ttnn::bernoulli>::invoke))
// - Used by aten ops (examples): bernoulli, bernoulli.out
template <auto Op>
    requires TTNNRandomFn<Op>
struct unary_random_seeded {
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
};  // struct unary_random_seeded

// Binary wrapper for TTNN ops with optional out/dtype/memory/compute params (e.g., ttnn::moreh_dot)
// - Expected Op signature: Op(a, b, opt_out, opt_dtype, opt_mem, opt_cfg)
// - Behavior: forward all optional parameters as std::nullopt
// - Example: m.impl("dot", TORCH_FN(binary_tensor_tensor_outlike<ttnn::moreh_dot>::invoke))
// - Used by aten ops (examples): dot, dot.out
template <auto Op>
    requires TTNNBinaryOutLikeFn<Op>
struct binary_tensor_tensor_outlike {
    static_assert(TTNNBinaryOutLikeFn<Op>, "Op must be (a, b, opt_out, opt_dtype, opt_mem, opt_cfg) -> ttnn::Tensor");

    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, b, out);
    }

    [[nodiscard]] static at::Tensor& invoke_into(
        const at::Tensor& a,
        const at::Tensor& b,
        at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(a);
        ttnn::Tensor b_tile = tileify(b);
        ttnn::Tensor result = Op(a_tile, b_tile, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        return write_from_ttnn(out, a, result);
    }
};  // struct binary_tensor_tensor_outlike

}  // namespace tt_eager::ext
