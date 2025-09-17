## Eager Ops Prioritization Status (based on #1215)

- **Model coverage**: 156 models
- **Unique operations**: 141
- **Source**: Tracking Prioritization of eager operations (#1215)

### 🥇 TIER 1: Critical (1 operation)
- **aten.convolution.default** — 128 models (82.1%): core convolution operation

### 🥈 TIER 2: High-priority foundation (3 operations)
- **aten.view.default** — 119 models (76.3%): tensor reshaping
- **aten.add.Tensor** — 114 models (73.1%): element-wise addition
- **aten.addmm.default** — 96 models (61.5%): matrix multiply + bias

### 🏗️ TIER 3: Foundational operations (5 operations)
- **aten.relu.default** — 87 models (55.8%): ReLU activation
- **aten.mean.dim** — 85 models (54.5%): mean along dimensions
- **aten.cat.default** — 68 models (43.6%): tensor concatenation
- **aten._native_batch_norm_legit_no_training.default** — 65 models (41.7%): batch norm (inference)
- **aten.mul.Tensor** — 64 models (41.0%): element-wise multiplication

### 🚀 Quick Wins Strategy (16 operations)
- Target the 10 simplest models (2–6 ops each):
  - Autoencoder (linear) — 2 ops
  - Autoencoder (conv) — 3 ops
  - VGG (11/13/16/19) — 5 ops each
  - U-Net (variants) — 5–6 ops

- Phase 1 operations needed for these models:
  - **aten.addmm.default**
  - **aten.relu.default**
  - **aten.convolution.default**
  - **aten.max_pool2d_with_indices.default**
  - **aten.view.default**
  - **aten._native_batch_norm_legit_no_training.default**
  - + ~10 more specialized operations (see per-model documentation)

### 🎯 Recommended Implementation Path
1. Implement/optimize **aten.convolution.default** (up to 82% model coverage)
2. Add **TIER 2** (view, add, addmm) for broad tensor manipulation support
3. Complete **Phase 1 (16 ops)** to unlock 10 simple models end-to-end
4. Finish **TIER 3** to reach 40–60% model coverage
5. Proceed incrementally by operation-count/complexity levels

---

### PR plan: split OPs registration across multiple PRs
- PR 1: Base infrastructure
  - Scope: allocator, device utilities, core creation/copy ops (`aten::empty_*`, `_copy_from`), registration skeleton and `TORCH_LIBRARY_IMPL` module.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/src/open_registration_extension.cpp`, `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/device.hpp`, `torch_ttnn/cpp_extension/ttnn_cpp_extension/core/*`
  - Refer: tracking `#1215`, PR: <add link to PR>
- PR 2: Unary
  - Scope: register unary operations (base/out/_) including `relu` and standard math functions.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/unary_eager_register.hpp`, `.../unary_eager_wrappers.hpp`
  - PR: <add link>
- PR 3: Binary
  - Scope: `add`/`_add_relu`, `sub`, `mul`, `div`, comparisons and logical operations, `matmul`.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/binary_eager_register.hpp`, `.../binary_eager_wrappers.hpp`
  - PR: <add link>
- PR 4: Convolution & Pooling
  - Scope: `aten.convolution.default`/`convolution_overrideable` (dispatcher) + `conv1d/2d/3d`, `conv_transpose2d`; `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/conv_pool_eager_register.hpp`, `.../conv_pool_eager_wrappers.hpp`
  - PR: <add link>
- PR 5: Reductions
  - Scope: `sum`, `mean`, `max/min` (including dim variants and indices), `std/var`.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/reduction_eager_register.hpp`, `.../reduction_eager_wrappers.hpp`
  - PR: <add link>
- PR 6: Random
  - Scope: `bernoulli`, `random_` (and variants), `uniform_`.
  - Key files: `torch_ttnn/cpp_extension/ttnn_cpp_extension/include/ttnn_cpp_extension/utils/random_eager_register.hpp`, `.../random_eager_wrappers.hpp`
  - PR: <add link>
- PR 7: Phase 1 extras (for simple models)
  - Scope: `aten.addmm.default`, `aten.view.default`, `aten.cat.default`, `aten._native_batch_norm_legit_no_training.default`.
  - Key files: corresponding registration files (add/extend as implemented).
  - PR: <add link>

> How to update this section: when opening each PR, add the link next to its item and reflect progress in the checklist below. In the PR description, reference `#1215` and this file.

### Execution Checklist (tracking)
- [x] TIER 1
  - [x] aten.convolution.default (dispatch via convolution/convolution_overrideable; conv1d/2d/3d + conv_transpose2d wired) — PR: Convolution & Pooling (<add link>)
- [ ] TIER 2
  - [ ] aten.view.default — PR: Phase 1 extras (<add link>)
  - [x] aten.add.Tensor — PR: Binary (<add link>)
  - [ ] aten.addmm.default — PR: Phase 1 extras (<add link>)
- [ ] Phase 1 (simple models plan)
  - [ ] aten.max_pool2d_with_indices.default — PR: Convolution & Pooling (<add link>)
  - [x] aten.max_pool2d (no indices) — PR: Convolution & Pooling (<add link>)
  - [x] aten.avg_pool2d — PR: Convolution & Pooling (<add link>)
  - [x] aten.adaptive_avg_pool2d (constraint: output [1,1]) — PR: Convolution & Pooling (<add link>)
- [ ] TIER 3
  - [x] aten.relu.default — PR: Unary (<add link>)
  - [x] aten.mean.dim — PR: Reductions (<add link>)
  - [ ] aten.cat.default — PR: Phase 1 extras (<add link>)
  - [ ] aten._native_batch_norm_legit_no_training.default — PR: Phase 1 extras (<add link>)
  - [x] aten.mul.Tensor — PR: Binary (<add link>)

> Note: Priorities and coverage reflect document #1215; details for specialized ops are in per-model docs.

### Status summary
- TIER 1: 1/1 complete
- TIER 2: 1/3 complete (done: add.Tensor)
- Phase 1 (core ops for simple models): 1/6 complete (done: relu; pooling partially covered)
- TIER 3: 3/5 complete (relu, mean.dim, mul.Tensor)

### TIER 1 — short implementation plan for `aten.convolution.default`
- Register `convolution` (and if needed `convolution_overrideable`) in `open_registration_extension.cpp` and dispatch by dimensionality and `transposed` flag:
  - 1D → `tt_eager::ext::conv1d_aten::invoke`
  - 2D (transposed=false) → `tt_eager::ext::conv2d_aten::invoke`
  - 2D (transposed=true) → `tt_eager::ext::conv_transpose2d_aten::invoke`
  - 3D → `tt_eager::ext::conv3d_aten::invoke`
- Normalize parameters: `stride`, `padding`, `dilation`, `output_padding`; compute output shape via existing helpers (`conv_out_dim`, `conv_transpose_out_dim`).
- Support `groups` and optional `bias`; validate TTNN device; use `tilize`/`write_from_ttnn`.
- Phase 1 scope: NCHW, float/bfloat16; no string padding modes; expand coverage afterwards.
- Add tests for 1D/2D/3D and transposed, checking output shapes and accuracy.
