#pragma once

#include <cstdint>
#include <random>

#include <ATen/core/Tensor.h>
#include <ATen/core/Generator.h>

#include <ttnn/operations/bernoulli/bernoulli.hpp>

#include "ttnn_cpp_extension/utils/eager_wrap.hpp"

namespace tt_eager::ops::random {

struct ttnn_bernoulli {
    static at::Tensor invoke(const at::Tensor& input, c10::optional<at::Generator> /*generator*/ = c10::nullopt) {
        at::Tensor out = tt_eager::ext::make_empty_like_tt(input);
        invoke_out(input, /*generator*/ c10::nullopt, out);
        return out;
    }

    static at::Tensor& invoke_out(
        const at::Tensor& input,
        c10::optional<at::Generator> /*generator*/,  // unused for now
        at::Tensor& out) {
        ttnn::Tensor in_tile = tt_eager::ext::tileify(input);

        // Produce a seed (placeholder: nondeterministic). Can be wired to torch generator later.
        static thread_local std::mt19937 rng(std::random_device{}());
        uint32_t seed = rng();

        ttnn::Tensor result = ttnn::bernoulli(
            in_tile,
            seed,
            std::nullopt /*output*/,
            std::nullopt /*dtype*/,
            std::nullopt /*memory_config*/,
            std::nullopt /*compute_kernel_config*/);

        return tt_eager::ext::write_from_ttnn(out, input, result);
    }
};

}  // namespace tt_eager::ops::random


