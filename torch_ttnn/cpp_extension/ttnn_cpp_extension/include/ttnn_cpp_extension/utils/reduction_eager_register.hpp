#pragma once

#include "ttnn_cpp_extension/utils/reduction_eager_wrappers.hpp"
#include <torch/library.h>

namespace tt_eager::ext {

static inline void register_reductions(torch::Library& m) {
    // =========================
    // Reductions
    // =========================
    // Sum
    // schema: sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum", TORCH_FN(tt_eager::ext::reduction_all<ttnn::sum>::invoke));
    // schema: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum.dim_IntList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke));
    // schema: sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out)
    // -> Tensor(a!)
    m.impl("sum.IntList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_into));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("sum.dim_DimnameList", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("sum.DimnameList_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::sum>::invoke_dimnames_into));
#endif  // TTNN_TORCH_HAS_DIMNAME

    // Mean
    // schema: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean", TORCH_FN(tt_eager::ext::reduction_all<ttnn::mean>::invoke));
    // schema: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke));
    // schema: mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("mean.out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_into));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    m.impl("mean.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!)
    // out) -> Tensor(a!)
    m.impl("mean.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist<ttnn::mean>::invoke_dimnames_into));
#endif  // TTNN_TORCH_HAS_DIMNAME

    // Max / Min (value-only reductions; aten::max/min no dtype)
    // schema: max(Tensor self) -> Tensor
    m.impl("max", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::max>::invoke));
    // schema: min(Tensor self) -> Tensor
    m.impl("min", TORCH_FN(tt_eager::ext::reduction_all_nodtype<ttnn::min>::invoke));

    // max with indices along dim (return (values, indices))
    // Pin 8dfb324: argmax lives in ttnn::argmax (not ttnn::experimental::argmax). No argmin op.
    using MaxPair = tt_eager::ext::reduction_dim_pair<ttnn::max, ttnn::argmax>;
    // schema: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("max.dim", TORCH_FN(MaxPair::invoke));
    // schema: max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) ->
    // (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.dim_max", TORCH_FN(MaxPair::invoke_into));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    m.impl("max.names_dim", TORCH_FN(MaxPair::invoke_dimname));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values)
    // -> (Tensor(a!) values, Tensor(b!) indices)
    m.impl("max.names_dim_max", TORCH_FN(MaxPair::invoke_dimname_into));
#endif  // TTNN_TORCH_HAS_DIMNAME

    // min.dim* with indices omitted: pin has no ttnn::argmin.

    // Std / Var
    // Base (all-elements) with unbiased flag default (correction)
    // schema: var(Tensor self, bool unbiased=True) -> Tensor
    m.impl("var", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::var>::invoke));
    // schema: std(Tensor self, bool unbiased=True) -> Tensor
    m.impl("std", TORCH_FN(tt_eager::ext::reduction_all_unbiased<ttnn::std>::invoke));

    // schema: var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("var.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::var>::invoke_into));
    // schema: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("var.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: var.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) ->
    // Tensor
    m.impl("var.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: var.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl(
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
        "var.correction_names_out",
        TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_dimnames_into));
    // schema: var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("var.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::var>::invoke_into));
#endif  // TTNN_TORCH_HAS_DIMNAME
    // schema: var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("var.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("var.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::var>::invoke_dimnames_into));
#endif  // TTNN_TORCH_HAS_DIMNAME

    // schema: std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("std.out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased_out<ttnn::std>::invoke_into));
    // schema: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    m.impl("std.correction", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: std.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) ->
    // Tensor
    m.impl("std.correction_names", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: std.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl(
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
        "std.correction_names_out",
        TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_dimnames_into));
    // schema: std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False,
    // Tensor(a!) out) -> Tensor(a!)
    m.impl("std.correction_out", TORCH_FN(tt_eager::ext::reduction_dimlist_correction<ttnn::std>::invoke_into));
#endif  // TTNN_TORCH_HAS_DIMNAME
    // schema: std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke));
#if TTNN_TORCH_HAS_DIMNAME
    // schema: std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
    m.impl("std.names_dim", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames));
#endif  // TTNN_TORCH_HAS_DIMNAME
#if TTNN_TORCH_HAS_DIMNAME
    // schema: std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) ->
    // Tensor(a!)
    m.impl("std.names_out", TORCH_FN(tt_eager::ext::reduction_dimlist_unbiased<ttnn::std>::invoke_dimnames_into));
#endif  // TTNN_TORCH_HAS_DIMNAME
}

}  // namespace tt_eager::ext
