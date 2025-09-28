#pragma once

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <vector>
#include <optional>
#include <tuple>
#include <string>
#include <utility>
#include <type_traits>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>

// TODO: this is a draft of autograd-related functions

namespace tt_eager::ext {

// =============================
// Helpers for saving/loading args in AutogradContext (C++20)
// =============================

// Apply callable to tuple with extra prefix parameters
template <typename F, typename Tuple, typename... Prefix>
static auto apply_with_prefix(F&& f, Tuple&& t, Prefix&&... prefix)
    -> decltype(auto) {
    return std::apply(
        [&](auto&&... args) -> decltype(auto) {
            return std::forward<F>(f)(std::forward<Prefix>(prefix)..., std::forward<decltype(args)>(args)...);
        },
        std::forward<Tuple>(t));
}

template <typename... Args>
struct SavedArgs {
    static void save(torch::autograd::AutogradContext* ctx, const Args&... args) {
        constexpr size_t tensor_count = (0 + ... + (std::is_same_v<Args, at::Tensor> ? 1 : 0));
        std::vector<at::Tensor> tensors;
        tensors.reserve(tensor_count);
        save_impl(ctx, tensors, std::index_sequence_for<Args...>{}, args...);
        ctx->save_for_backward(tensors);
    }

    static std::tuple<Args...> load(torch::autograd::AutogradContext* ctx) {
        auto saved = ctx->get_saved_variables();
        size_t tensor_index = 0;
        return load_impl(ctx, saved, tensor_index, std::index_sequence_for<Args...>{});
    }

private:
    template <typename T>
    struct DependentFalse : std::false_type {};

    template <size_t I>
    static std::string arg_key() {
        return std::string("arg_") + std::to_string(I);
    }

    template <size_t I, typename T>
    struct ArgIO {
        static void save(
            torch::autograd::AutogradContext* /*ctx*/,
            std::vector<at::Tensor>& /*tensors*/,
            const T& /*value*/) {
            static_assert(DependentFalse<T>::value, "Unsupported type in SavedArgs. Provide adapter/specialization.");
        }

        static T load(
            torch::autograd::AutogradContext* /*ctx*/,
            const std::vector<at::Tensor>& /*saved_tensors*/,
            size_t& /*tensor_index*/) {
            static_assert(DependentFalse<T>::value, "Unsupported type in SavedArgs. Provide adapter/specialization.");
        }
    };

    template <size_t I>
    struct ArgIO<I, at::Tensor> {
        static void save(
            torch::autograd::AutogradContext* /*ctx*/,
            std::vector<at::Tensor>& tensors,
            const at::Tensor& value) {
            tensors.emplace_back(value);
        }

        static at::Tensor load(
            torch::autograd::AutogradContext* /*ctx*/,
            const std::vector<at::Tensor>& saved_tensors,
            size_t& tensor_index) {
            const at::Tensor& t = saved_tensors.at(tensor_index++);
            return t;
        }
    };

    template <size_t I>
    struct ArgIO<I, c10::Scalar> {
        static void save(
            torch::autograd::AutogradContext* ctx,
            std::vector<at::Tensor>& /*tensors*/,
            const c10::Scalar& value) {
            ctx->saved_data[arg_key<I>()] = value;
        }

        static c10::Scalar load(
            torch::autograd::AutogradContext* ctx,
            const std::vector<at::Tensor>& /*saved_tensors*/,
            size_t& /*tensor_index*/) {
            return ctx->saved_data.at(arg_key<I>()).toScalar();
        }
    };

    template <size_t... I>
    static std::tuple<Args...> load_impl(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& saved_tensors,
        size_t& tensor_index,
        std::index_sequence<I...>) {
        return std::tuple<Args...>{ ArgIO<I, Args>::load(ctx, saved_tensors, tensor_index)... };
    }

    template <size_t... I>
    static void save_impl(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor>& tensors,
        std::index_sequence<I...>,
        const Args&... args) {
        (ArgIO<I, Args>::save(ctx, tensors, args), ...);
    }
};

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


// General argument preparation policy
struct PrepareArg {
    static ttnn::Tensor to_backend(const at::Tensor& t) { return tileify(t); }
    static float to_backend(const c10::Scalar& s) { return static_cast<float>(s.toDouble()); }
};

// =============================
// Generic backward builders
// =============================

// Build tuple of prepared backend args in the same order as Args...
template <typename... Args>
static auto build_prepared_tuple(const Args&... args) {
    return std::make_tuple(PrepareArg::to_backend(args)...);
}

// Consume a result that yields gradients per each Tensor in Args...
// Returns tuple of at::Tensor grads in order of Tensor occurrences in Args...
template <typename Result, typename... Args>
static auto collect_tensor_grads(const Result& result, const Args&... args) {
    size_t result_index = 0;
    std::tuple<> empty;
    // Helper lambda to append a new grad tensor to tuple
    auto append = [](auto tuple, at::Tensor value) {
        return std::tuple_cat(std::move(tuple), std::make_tuple(std::move(value)));
    };

    // folder over Args...
    auto grads_tuple = std::tuple<>{};
    (void)std::initializer_list<int>{ ([&]{
        using T = std::decay_t<Args>;
        if constexpr (std::is_same_v<T, at::Tensor>) {
            const ttnn::Tensor& grad_tt = pick_result(result, result_index++);
            at::Tensor grad_like = make_empty_like_tt(args);
            at::Tensor written = write_from_ttnn(grad_like, args, grad_tt);
            grads_tuple = append(std::move(grads_tuple), std::move(written));
        }
    }(), 0)... };
    return grads_tuple;
}

// Invokers for TTNN backward calls
template <auto BackwardTTNN>
struct UnaryInvoker {
    template <typename... Prepared>
    static auto run(const ttnn::Tensor& g, const Prepared&... prepared) {
        static_assert(sizeof...(Prepared) == 1, "UnaryInvoker expects 1 prepared arg");
        return BackwardTTNN(g, prepared..., std::nullopt);
    }
};

template <auto BackwardTTNN>
struct BinaryInvoker {
    template <typename... Prepared>
    static auto run(const ttnn::Tensor& g, const Prepared&... prepared) {
        static_assert(sizeof...(Prepared) == 2, "BinaryInvoker expects 2 prepared args");
        return invoke_binary_bw_ttnn<BackwardTTNN>(g, prepared...);
    }
};

template <auto BackwardTTNN>
struct BinaryAlphaInvoker {
    template <typename... Prepared>
    static auto run(const ttnn::Tensor& g, const Prepared&... prepared) {
        static_assert(sizeof...(Prepared) == 3, "BinaryAlphaInvoker expects 3 prepared args");
        return invoke_binary_alpha_bw_ttnn<BackwardTTNN>(g, prepared...);
    }
};

// Single generic backward runner parametrized by Invoker and argument types
template <typename Invoker, typename... Args>
static auto run_generic_backward(
    const at::Tensor& grad_out,
    const Args&... args) {
    ttnn::Tensor g_tile = tileify(grad_out);
    auto prepared = build_prepared_tuple(args...);
    auto result = apply_with_prefix(
        [&](const ttnn::Tensor& g, const auto&... prepared_args) {
            return Invoker::run(g, prepared_args...);
        },
        prepared,
        g_tile);
    return collect_tensor_grads(result, args...);
}

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

// =============================
// Categories for unary/binary/binary+alpha
// =============================

template <auto ForwardTTNN, auto BackwardTTNN>
struct UnaryCategory {
    template <typename... Args>
    static at::Tensor forward(const Args&... args) {
        return tt_eager::ext::unary_wrapper<ForwardTTNN>::invoke(args...);
    }

    template <typename... Args>
    static std::tuple<at::Tensor> backward(
        const at::Tensor& grad_out,
        const Args&... args) {
        return run_generic_backward<UnaryInvoker<BackwardTTNN>>(grad_out, args...);
    }
};

template <auto ForwardTTNN, auto BackwardTTNN>
struct BinaryCategory {
    template <typename... Args>
    static at::Tensor forward(const Args&... args) {
        return tt_eager::ext::binary_wrapper<ForwardTTNN>::invoke(args...);
    }

    template <typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_out,
        const Args&... args) {
        return run_generic_backward<BinaryInvoker<BackwardTTNN>>(grad_out, args...);
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

// call_binary_alpha_bw inlined directly inside BinaryAlphaCategory::backward

// Generic utilities for tuple -> variable_list
template <typename Tuple, size_t... I>
inline torch::autograd::variable_list tuple_to_varlist_impl(const Tuple& t, std::index_sequence<I...>) {
    return { std::get<I>(t)... };
}

template <typename... T>
inline torch::autograd::variable_list tuple_to_varlist(const std::tuple<T...>& t) {
    return tuple_to_varlist_impl(t, std::index_sequence_for<T...>{});
}

// Category for Binary+alpha using existing wrappers/callers
template <auto ForwardTTNN, auto BackwardTTNN>
struct BinaryAlphaCategory {
    template <typename... Args>
    static at::Tensor forward(const Args&... args) {
        return tt_eager::ext::binary_alpha_wrapper<ForwardTTNN>::invoke(args...);
    }

    template <typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_out,
        const Args&... args) {
        return run_generic_backward<BinaryAlphaInvoker<BackwardTTNN>>(grad_out, args...);
    }
};

// Autograd wrapper generator parametrized by Category and argument types
template <typename Category, typename... Args>
struct AutogradWrapperFn {
    using SavedHelper = SavedArgs<Args...>;

    struct Fn : public torch::autograd::Function<Fn> {
        static at::Tensor forward(torch::autograd::AutogradContext* ctx, const Args&... args) {
            at::AutoDispatchBelowADInplaceOrView guard;
            at::Tensor out = Category::template forward<Args...>(args...);
            SavedHelper::save(ctx, args...);
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grads) {
            auto args = SavedHelper::load(ctx);
            const at::Tensor& g = grads.at(0);
            auto grads_tuple = apply_with_prefix(
                [](const at::Tensor& gg, const Args&... unpacked) {
                    return Category::template backward<Args...>(gg, unpacked...);
                },
                args,
                g);
            return tuple_to_varlist(grads_tuple);
        }
    };

    static at::Tensor invoke(const Args&... args) {
        return Fn::apply(args...);
    }
};

template <auto ForwardTTNN, auto BackwardTTNN>
using autograd_binary_alpha_wrapper = AutogradWrapperFn<BinaryAlphaCategory<ForwardTTNN, BackwardTTNN>, at::Tensor, at::Tensor, c10::Scalar>;

template <auto ForwardTTNN, auto BackwardTTNN>
using autograd_binary_wrapper = AutogradWrapperFn<BinaryCategory<ForwardTTNN, BackwardTTNN>, at::Tensor, at::Tensor>;

template <auto ForwardTTNN, auto BackwardTTNN>
    requires TTNNUnaryFn<ForwardTTNN> && TTNNUnaryBackwardFn<BackwardTTNN>
using autograd_unary_wrapper = AutogradWrapperFn<UnaryCategory<ForwardTTNN, BackwardTTNN>, at::Tensor>;

// previous specialized autograd_binary_alpha_wrapper replaced by alias to generic generator above

} // namespace tt_eager::ext


