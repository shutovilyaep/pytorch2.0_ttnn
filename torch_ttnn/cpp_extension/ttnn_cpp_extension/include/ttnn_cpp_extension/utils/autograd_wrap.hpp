#pragma once

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <vector>
#include <optional>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>

// TODO: this is a draft of autograd-related functions

namespace tt_eager::ext {

// Common extractor for TTNN backward results that may be Tensor, vector<Tensor>, or vector<optional<Tensor>>
inline const ttnn::Tensor& pick_result(const ttnn::Tensor& t, size_t) { return t; }

inline const ttnn::Tensor& pick_result(const std::vector<ttnn::Tensor>& v, size_t idx) {
    return v.at(idx);
}

inline const ttnn::Tensor& pick_result(const std::vector<std::optional<ttnn::Tensor>>& v, size_t idx) {
    const auto& opt = v.at(idx);
    TORCH_CHECK(opt.has_value(), "TTNN backward returned empty optional at index ", idx);
    return opt.value();
}

// Backward concept: TTNN registered op callable with (grad, input, optional mem cfg)
template <auto Op>
concept TTNNUnaryBackwardFn = requires(
    const ttnn::Tensor& grad_out,
    const ttnn::Tensor& saved_in) {
    Op(grad_out, saved_in, std::nullopt);
};

// Helper to invoke TTNN unary backward entry points with ATen tensors
template <auto BackwardTTNN>
    requires TTNNUnaryBackwardFn<BackwardTTNN>
inline at::Tensor call_unary_bw(
    const at::Tensor& grad_out,
    const at::Tensor& saved_in,
    const at::Tensor& /*saved_out*/) {
    ttnn::Tensor g_tile = tileify(grad_out);
    ttnn::Tensor a_tile = tileify(saved_in);

    auto result = BackwardTTNN(g_tile, a_tile, std::nullopt);
    const ttnn::Tensor& grad_in_tt = pick_result(result, 0);

    at::Tensor grad_in = make_empty_like_tt(saved_in);
    return write_from_ttnn(grad_in, saved_in, grad_in_tt);
}


// Generic Autograd wrapper for unary functional ops: Tensor -> Tensor
template <auto ForwardTTNN, auto BackwardTTNN>
    requires TTNNUnaryFn<ForwardTTNN> && TTNNUnaryBackwardFn<BackwardTTNN>
struct autograd_unary_wrapper {
    struct Fn : public torch::autograd::Function<Fn> {
        static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const at::Tensor& a) {
            // Prevent re-entrancy into Autograd while calling the real device forward
            at::AutoDispatchBelowADInplaceOrView guard;

            // Use PrivateUse1 forward wrapper to do device math
            at::Tensor out = tt_eager::ext::unary_wrapper<ForwardTTNN>::invoke(a);

            // Save minimal state required by backward (input and output)
            ctx->save_for_backward({a, out});
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grads) {
            auto saved = ctx->get_saved_variables();
            const at::Tensor& a   = saved.at(0);
            const at::Tensor& out = saved.at(1);

            const at::Tensor& g = grads.at(0);

            at::Tensor g_in = tt_eager::ext::call_unary_bw<BackwardTTNN>(g, a, out);
            return {g_in};
        }
    };

    // Exact functional schema adaptor (Tensor self -> Tensor)
    static at::Tensor invoke(const at::Tensor& a) {
        return Fn::apply(a);
    }
};

// =============================
// Binary (Tensor, Tensor) -> Tensor
// =============================

// Some TTNN binary backward ops accept (grad, a, b, memory_config),
// others require more args: (grad, a, b, are_required_outputs, memory_config, input_grad, other_grad).
// Provide an invoker that adapts at compile time.
template <auto Op>
inline auto invoke_binary_bw_ttnn(
    const ttnn::Tensor& grad_out,
    const ttnn::Tensor& a,
    const ttnn::Tensor& b) {
    if constexpr (requires { Op(grad_out, a, b, std::nullopt); }) {
        return Op(grad_out, a, b, std::nullopt);
    } else {
        return Op(
            grad_out,
            a,
            b,
            std::vector<bool>{true, true},
            std::nullopt,
            std::nullopt,
            std::nullopt);
    }
}

template <auto BackwardTTNN>
inline std::pair<at::Tensor, at::Tensor> call_binary_bw(
    const at::Tensor& grad_out,
    const at::Tensor& a,
    const at::Tensor& b) {
    ttnn::Tensor g_tile = tileify(grad_out);
    ttnn::Tensor a_tile = tileify(a);
    ttnn::Tensor b_tile = tileify(b);

    auto result = invoke_binary_bw_ttnn<BackwardTTNN>(g_tile, a_tile, b_tile);
    const ttnn::Tensor& grad_a_tt = pick_result(result, 0);
    const ttnn::Tensor& grad_b_tt = pick_result(result, 1);

    at::Tensor grad_a = make_empty_like_tt(a);
    at::Tensor grad_b = make_empty_like_tt(b);
    return { write_from_ttnn(grad_a, a, grad_a_tt), write_from_ttnn(grad_b, b, grad_b_tt) };
}

template <auto ForwardTTNN, auto BackwardTTNN>
struct autograd_binary_wrapper {
    struct Fn : public torch::autograd::Function<Fn> {
        static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const at::Tensor& a,
            const at::Tensor& b) {
            at::AutoDispatchBelowADInplaceOrView guard;
            at::Tensor out = tt_eager::ext::binary_wrapper<ForwardTTNN>::invoke(a, b);
            ctx->save_for_backward({a, b, out});
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grads) {
            auto saved = ctx->get_saved_variables();
            const at::Tensor& a = saved.at(0);
            const at::Tensor& b = saved.at(1);
            const at::Tensor& g = grads.at(0);

            auto [ga, gb] = tt_eager::ext::call_binary_bw<BackwardTTNN>(g, a, b);
            return {ga, gb};
        }
    };

    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b) {
        return Fn::apply(a, b);
    }
};

// =============================
// Binary+alpha (Tensor, Tensor, Scalar) -> Tensor
// =============================

// Similar adaptation wrapper for addalpha/subalpha backward
template <auto Op>
inline auto invoke_binary_alpha_bw_ttnn(
    const ttnn::Tensor& grad_out,
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    float alpha) {
    if constexpr (requires { Op(grad_out, a, b, alpha, std::nullopt); }) {
        return Op(grad_out, a, b, alpha, std::nullopt);
    } else {
        return Op(
            grad_out,
            a,
            b,
            alpha,
            std::vector<bool>{true, true},
            std::nullopt,
            std::nullopt,
            std::nullopt);
    }
}

template <auto BackwardTTNN>
inline std::pair<at::Tensor, at::Tensor> call_binary_alpha_bw(
    const at::Tensor& grad_out,
    const at::Tensor& a,
    const at::Tensor& b,
    const c10::Scalar& alpha) {
    ttnn::Tensor g_tile = tileify(grad_out);
    ttnn::Tensor a_tile = tileify(a);
    ttnn::Tensor b_tile = tileify(b);
    float alpha_value = static_cast<float>(alpha.toDouble());

    auto result = invoke_binary_alpha_bw_ttnn<BackwardTTNN>(g_tile, a_tile, b_tile, alpha_value);
    const ttnn::Tensor& grad_a_tt = pick_result(result, 0);
    const ttnn::Tensor& grad_b_tt = pick_result(result, 1);

    at::Tensor grad_a = make_empty_like_tt(a);
    at::Tensor grad_b = make_empty_like_tt(b);
    return { write_from_ttnn(grad_a, a, grad_a_tt), write_from_ttnn(grad_b, b, grad_b_tt) };
}

template <auto ForwardTTNN, auto BackwardTTNN>
struct autograd_binary_alpha_wrapper {
    struct Fn : public torch::autograd::Function<Fn> {
        static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const at::Tensor& a,
            const at::Tensor& b,
            const c10::Scalar& alpha) {
            at::AutoDispatchBelowADInplaceOrView guard;
            at::Tensor out = tt_eager::ext::binary_alpha_wrapper<ForwardTTNN>::invoke(a, b, alpha);
            ctx->save_for_backward({a, b, out});
            ctx->saved_data["alpha"] = alpha;
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grads) {
            auto saved = ctx->get_saved_variables();
            const at::Tensor& a = saved.at(0);
            const at::Tensor& b = saved.at(1);
            const at::Tensor& g = grads.at(0);
            c10::Scalar alpha = ctx->saved_data.at("alpha").toScalar();

            auto [ga, gb] = tt_eager::ext::call_binary_alpha_bw<BackwardTTNN>(g, a, b, alpha);
            return {ga, gb};
        }
    };

    static at::Tensor invoke(const at::Tensor& a, const at::Tensor& b, const c10::Scalar& alpha) {
        return Fn::apply(a, b, alpha);
    }
};

} // namespace tt_eager::ext


