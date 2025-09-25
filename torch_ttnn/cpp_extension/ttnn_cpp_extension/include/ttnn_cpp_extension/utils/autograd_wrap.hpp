#pragma once

// PR-ready: generic Autograd wrappers for TTNN ops
// - Templated, concept-checked wrappers to register functional autograd kernels
// - Keeps forward math in PrivateUse1 wrappers, exposes only functional overloads to AutogradPrivateUse1

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

// #include <concepts> // not directly needed; keep minimal includes

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>

namespace tt_eager::ext {

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

    ttnn::Tensor grad_in_tt;
    if constexpr (std::is_same_v<std::decay_t<decltype(result)>, ttnn::Tensor>) {
        grad_in_tt = result;
    } else {
        grad_in_tt = result.at(0);
    }

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

} // namespace tt_eager::ext


