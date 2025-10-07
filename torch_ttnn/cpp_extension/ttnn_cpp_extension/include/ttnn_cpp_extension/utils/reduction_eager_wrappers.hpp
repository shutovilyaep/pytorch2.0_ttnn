#pragma once

#include "ttnn_cpp_extension/utils/eager_common.hpp"

#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/experimental/reduction/argmax/argmax.hpp>

namespace tt_eager::ext {

inline std::optional<std::variant<int, ttnn::SmallVector<int>>> to_ttnn_dim_variant(c10::OptionalArrayRef<int64_t> dims) {
    if (!dims.has_value()) return std::nullopt;
    return to_ttnn_dim_variant(*dims);
}

template <auto Op>
struct reduction_all {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, c10::optional<at::ScalarType> dtype = c10::nullopt) {
        at::Tensor out = make_empty_like_tt(a, dtype);
        return invoke_into(a, dtype, out);
    }
    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& in, c10::optional<at::ScalarType> dtype, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        ttnn::Tensor result = Op(a_tile, std::nullopt, /*keepdim*/ false);
        if (dtype.has_value()) {
            result = ttnn::typecast(result, to_ttnn_dtype(*dtype));
        }
        return write_from_ttnn(out, in, result);
    }
};

template <auto Op>
struct reduction_all_nodtype {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a) {
        at::Tensor out = make_empty_like_tt(a);
        return invoke_into(a, out);
    }
    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& in, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        ttnn::Tensor result = Op(a_tile, std::nullopt, /*keepdim*/ false);
        return write_from_ttnn(out, in, result);
    }
};

template <auto Op>
struct reduction_dimlist {
    [[nodiscard]] static at::Tensor invoke(const at::Tensor& a, c10::OptionalArrayRef<int64_t> dim, bool keepdim, c10::optional<at::ScalarType> dtype = c10::nullopt) {
        at::Tensor out = make_empty_like_tt(a, dtype);
        return invoke_into(a, dim, keepdim, dtype, out);
    }
    [[nodiscard]] static at::Tensor& invoke_into(const at::Tensor& in, c10::OptionalArrayRef<int64_t> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out) {
        ttnn::Tensor a_tile = tileify(in);
        std::optional<std::variant<int, ttnn::SmallVector<int>>> dim_variant = to_ttnn_dim_variant(dim);
        ttnn::Tensor result = Op(a_tile, dim_variant, keepdim);
        if (dtype.has_value()) {
            result = ttnn::typecast(result, to_ttnn_dtype(*dtype));
        }
        return write_from_ttnn(out, in, result);
    }
};

template <auto ReduceOp, auto ArgOp>
struct reduction_dim_pair {
    [[nodiscard]] static std::tuple<at::Tensor, at::Tensor> invoke(const at::Tensor& a, int64_t dim, bool keepdim) {
        ttnn::Tensor a_tile = tileify(a);
        auto dim_variant = std::optional<std::variant<int, ttnn::SmallVector<int>>>(static_cast<int>(dim));
        ttnn::Tensor v_tt = ReduceOp(a_tile, dim_variant, keepdim);
        ttnn::Tensor i_tt = ArgOp(a_tile, static_cast<int64_t>(dim), /*all=*/false);
        at::Tensor v_out = make_empty_like_tt(a);
        at::Tensor i_out = make_empty_like_tt(a, at::kInt);
        write_from_ttnn(v_out, a, v_tt);
        write_from_ttnn(i_out, a, i_tt);
        return {v_out, i_out};
    }
    [[nodiscard]] static std::tuple<at::Tensor&, at::Tensor&> invoke_into(const at::Tensor& a, int64_t dim, bool keepdim, at::Tensor& values_out, at::Tensor& indices_out) {
        ttnn::Tensor a_tile = tileify(a);
        auto dim_variant = std::optional<std::variant<int, ttnn::SmallVector<int>>>(static_cast<int>(dim));
        ttnn::Tensor v_tt = ReduceOp(a_tile, dim_variant, keepdim);
        ttnn::Tensor i_tt = ArgOp(a_tile, static_cast<int64_t>(dim), /*all=*/false);
        write_from_ttnn(values_out, a, v_tt);
        write_from_ttnn(indices_out, a, i_tt);
        return {values_out, indices_out};
    }
};

} // namespace tt_eager::ext



