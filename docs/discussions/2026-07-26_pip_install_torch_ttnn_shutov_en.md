# Why `pip install torch-ttnn[pypi]` does not work (and what `torch-ttnn-shutov[pypi]` fixes)

This note documents the packaging gap between Tenstorrent’s public instructions and a reproducible PyPI install path for the PyTorch 2.0 eager backend work that was ready for merge in late 2025.

## Executive summary

- Upstream `tenstorrent/pytorch2.0_ttnn` **main** does not document `pip install torch-ttnn[pypi]`; it points at Bitbucket and editable installs.
- `pip install torch-ttnn` from public PyPI currently resolves to **0.5.6**, which does **not** depend on `ttnn` and has no `[pypi]` extra.
- The $500 packaging bounty ([#1036](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036) → [#1095](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095)) made `ttnn` optional and split CI pins from publish metadata; it did not deliver an end-to-end public install.
- The first `ttnn-shutov` wheel repack (0.62) dropped auditwheel-mangled runtime libraries while `libtt_metal.so` still NEEDED them, so `import ttnn` failed on a clean machine.
- **`torch-ttnn-shutov[pypi]`** pairs `torch-ttnn-shutov` with a from-source `ttnn-shutov` wheel built at the tt-metal merge pin, bundling OpenMPI ULFM without `patchelf` (which SIGSEGVs on these ELFs) and using an RTLD_GLOBAL pre-loader in `ttnn/__init__.py`.

## Upstream install story

On upstream **main**, `README.md` recommends:

- `pip install git+https://bitbucket.org/tenstorrent/pytorch2.0_ttnn`
- `pip install -e .` for development

`requirements.txt` pins `ttnn` via a **direct URL** to Tenstorrent’s internal index (`pypi.eng.aws.tenstorrent.com`), which is not reachable from the public internet.

## Public PyPI `torch-ttnn`

Installing `torch-ttnn` from https://pypi.org/project/torch-ttnn/ yields **0.5.6** without pulling `ttnn`. There is no `[pypi]` extra on that release line, so following a hypothetical `pip install torch-ttnn[pypi]` would not install a runtime even if documented.

## Bounty #1036 / PR #1095 (July 2025)

| Item | Detail |
|------|--------|
| Issue | [#1036](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036) — “Fix packaging workflow (PyPI)” |
| PR | [#1095](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095) by `aybanda` |
| Merge | 2025-07-09 by `jmalone-tt` |
| Payment thread | `cguerreroTT` on the issue |

The merged change made `ttnn` an optional dependency (`torch-ttnn` vs `torch-ttnn[ttnn]`) and removed `requirements.txt` from published package metadata while CI kept a direct-URL `ttnn` pin. That is incompatible with uploading a single PyPI distribution that both declares and installs `ttnn` from a public index.

## What broke in `ttnn-shutov` 0.62

The repack workflow produced wheels where `libtt_metal.so` still listed auditwheel-mangled NEEDED entries (`libmpi-0057c1fb…`, `libhwloc-936ce3d6…`, etc.) but the `ttnn.libs` directory was not shipped. Additional blockers in that line included `version("ttnn")` (distribution name is `ttnn_shutov`) and a hard SFPI system-package gate.

## Our approach (`shutovilyaep`)

1. Build `ttnn` from the tt-metal merge pin in manylinux CI (no `auditwheel` / no `patchelf` on project ELFs).
2. `tools/bundle_ttnn_runtime_libs.py`: copy-only closure for OpenMPI ULFM + `libtracy`, plus RTLD_GLOBAL `dlopen` pre-load in `ttnn/__init__.py`.
3. Repack as `ttnn-shutov` with PEP 440 public version (e.g. `0.65.0.dev20251205`) and pin from `torch-ttnn-shutov[pypi]`.
4. `torch_ttnn/__init__.py` shims removed APIs (`format_output_tensor` → `to_layout`) for pin 8dfb324.

## Supported install (no Tenstorrent hardware required for import)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install 'torch-ttnn-shutov[pypi]'
python -c "import ttnn; import torch_ttnn; print('ok')"
```

Platform: **Linux x86_64**, **CPython 3.10**. Import succeeds without `LD_LIBRARY_PATH`, `LD_PRELOAD`, or `TT_METAL_HOME`. Running models on device still requires Tenstorrent hardware and their system packages.

## Evidence discipline

Facts about merge readiness and reviews are scoped per [evidence manifest](https://github.com/shutovilyaep/pytorch2.0_ttnn/blob/main/docs/discussions/evidence_manifest.md) in the discussion archive: “no approving review” and a conflict-free window are recorded facts; “only Approve remained” is not asserted as fact.
