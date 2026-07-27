# Why `pip install torch-ttnn[pypi]` does not work (and what `torch-ttnn-shutov[pypi]` fixes)

This note documents the packaging gap between Tenstorrent's public instructions and a reproducible PyPI install path for the PyTorch 2.0 eager backend work that was ready for merge in late 2025.

## Executive summary

- Upstream `tenstorrent/pytorch2.0_ttnn` **main** does not document `pip install torch-ttnn[pypi]`; it points at a Bitbucket URL (HTTP 404) and editable installs.
- `pip install torch-ttnn` from public PyPI currently resolves to **0.5.6**, which does **not** depend on `ttnn` and has no `[pypi]` extra (only `dev`).
- The $500 packaging bounty ([#1036](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036) -> [#1095](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095)) added optional extra `ttnn` pinning `ttnn==0.59.0`, which does not exist on public PyPI; it did not deliver an end-to-end public install.
- Upstream `requirements.txt` pins `ttnn` via a direct URL to `pypi.eng.aws.tenstorrent.com`. That wheel **is** publicly downloadable (HTTP 200, sha256 matches the pin), but it is built from tt-metal `627c4eed5b` while the repo submodule is `33cbd50ba33` (3336 commits apart) - a cross-ABI runtime mismatch for eager C++.
- On **2025-11-20** (when the eager stack was ready), the newest public `ttnn` on PyPI was **0.64.0**, which still misses **90** of **145** `ttnn` symbols required by `torch-ttnn-shutov` 0.2.0. Public `ttnn==0.65.0` appeared on **2025-12-15**, after that window.
- Early `ttnn-shutov` 0.62 wheels failed import for other reasons (`version("ttnn")` lookup and an SFPI gate in `library_tweaks.py`), not because `ttnn.libs` was missing from the wheel.
- **`torch-ttnn-shutov[pypi]`** pairs `torch-ttnn-shutov` with a from-source `ttnn-shutov` wheel built at tt-metal pin `8dfb324`, bundling OpenMPI ULFM without `patchelf` (which SIGSEGVs on these ELFs) and using an RTLD_GLOBAL pre-loader in `ttnn/__init__.py`.

## Upstream install story

On upstream **main**, `README.md` recommends:

- `pip install git+https://bitbucket.org/tenstorrent/pytorch2.0_ttnn` (returns **404** as of 2026-07-26)
- `pip install -e .` for development

`requirements.txt` pins `ttnn` with a **direct URL** to Tenstorrent's index host `pypi.eng.aws.tenstorrent.com`:

- Wheel: `ttnn-0.62.0rc36.dev247+g627c4eed5b-cp310-cp310-manylinux_2_34_x86_64.whl`
- Recorded sha256: `c219ab8b8179d160a5262a802706710c0c4ae135bfbd5d203d8107fa3b6d0d07`
- Verified download: HTTP **200**, file size 27,981,239 bytes, sha256 **matches** (no auth required)

So the blocker is **not** "index unreachable from the public internet". The blockers are:

1. Public PyPI/TestPyPI reject uploaded wheels whose metadata contains direct URL dependencies (`ttnn @ https://...`).
2. The pinned wheel is built from tt-metal **`627c4eed5b`** (2025-09-16), while upstream's `tt-metal` submodule on main is **`33cbd50ba33`** (2025-05-04). Eager C++ is compiled against the submodule, not against that wheel revision.

## How upstream CI actually resolves `tt-metal` (four versions in one commit)

Verified on upstream `main` at `7fb5bbaaff` (2026-02-24). A single checkout encodes **four** divergent `ttnn` / `tt-metal` identities:

| Source | Value |
| --- | --- |
| `.gitmodules` `branch =` | `v0.58.0-rc25` |
| Submodule gitlink (what `git submodule update` checks out) | `33cbd50ba338b2ea784582d8cad7a3e19884b9db` (2025-05-04) |
| `requirements.txt` direct-URL wheel | `ttnn-0.62.0rc36.dev247+g627c4eed5b-...` = tt-metal `627c4eed5b` (2025-09-17) |
| `setup.py` `install_requires` | `ttnn==0.60.1` (and `torch==2.2.1+cpu`, while `requirements.txt` asks for `torch==2.7.1+cpu`) |

Distance pin -> wheel: **3336** commits linearly (`git rev-list --left-right --count 33cbd50ba33...627c4eed5b` = `0 3336`; pin is a direct ancestor).

### Decorative `.gitmodules` rewrite

Two workflows rewrite `branch =` in `.gitmodules` by scraping the internal PyPI HTML or the GitHub releases API:

- [`update-ttnn-wheel.yaml`](https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/.github/workflows/update-ttnn-wheel.yaml) job `build_cache_cpp_extension`, step `Update .gitsubmodules`
- [`run-cpp-native-tests.yaml`](https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/.github/workflows/run-cpp-native-tests.yaml), step `Update .gitsubmodules`

Both then run `git submodule sync && git submodule update --init --recursive`. The flag `--remote` does **not** appear anywhere in the repository. Without `--remote`, `git submodule update` checks out the recorded gitlink and **ignores** the `branch` field. The log line looks like "Updated .gitmodules with the latest version: v0.62.0rc36-247-g627c4eed5b", but the tree that is actually built remains `33cbd50ba33`. The substituted string is also a `git describe`-style label, not a published ref: tag `v0.62.0rc36*` is absent from public `tt-metal`, so even `--remote` would fail.

### Editable overlay on top of the wheel

[`build_cpp_extension.sh`](https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/torch_ttnn/cpp_extension/build_cpp_extension.sh) ends with `pip3 install -e $CUR_DIR/third-party/tt-metal/`. By that point `requirements-dev.txt` (which includes `-r requirements.txt`) has already installed the `627c4eed5b` wheel; the editable submodule install overlays it. In CI, tests therefore run against an editable build of the submodule pin, not against the wheel a `pip install` user receives.

**Consequence:** upstream CI never exercises the configuration a public `pip install` user gets. CI tests editable submodule `33cbd50ba33`; a user who follows `requirements.txt` gets wheel `627c4eed5b` (3336 commits apart). The C++ extension is compiled against submodule headers, so the mismatch surfaces as a cross-ABI failure on the eager path - and is invisible inside CI.

Public `torch-ttnn` on PyPI remains **0.5.6** with no `ttnn` dependency and no `[pypi]` extra. `ttnn==0.60.1` from `setup.py` does exist on public PyPI (66 releases total) - a third version, distinct from both the submodule pin and the requirements wheel. Internal index host returns HTTP 200.

## Public PyPI `torch-ttnn` and `ttnn`

Installing `torch-ttnn` from https://pypi.org/project/torch-ttnn/ yields **0.5.6** without pulling `ttnn`. Metadata lists `provides_extra = ['dev']` only - no `[pypi]` and no `[ttnn]` extra on that release line.

Public `ttnn` **does** exist on https://pypi.org/project/ttnn/ (many versions from 0.59.x through 0.74.0). That does not help the November 2025 eager handoff timeline:

| Runtime wheel | tt-metal / build identity | Missing `ttnn` symbols (of 145 needed by `torch-ttnn-shutov` 0.2.0) |
| --- | --- | ---: |
| `ttnn-shutov==0.65.0.dev20251205` (our pin `8dfb324`) | pin `8dfb324099a` | **0** |
| Public PyPI `ttnn==0.65.0` (uploaded 2025-12-15) | upstream release line | **0** |
| Public PyPI `ttnn==0.64.0` (newest on 2025-11-20) | upstream release line | **90** |
| Upstream `requirements.txt` wheel (`627c4eed5b`) | `627c4eed5b` | **102** |

Method: `nm -D -u` on the published `torch-ttnn-shutov` 0.2.0 extension, undefined symbols in `ttnn::` / `tt::tt_metal::` / `ttsl::` namespaces only, compared against each wheel's `_ttnncpp.so` (and bundled libs).

Example ABI surface change (`ExecuteUnary` op 34): upstream requirements wheel uses `invoke(QueueId, Tensor const&, ...)`; public `ttnn` 0.64.0+ and our `8dfb324` build use `invoke(Tensor const&, ..., optional<CoreRangeSet> const&)`.

**Post-hoc check (2026-07-26):** `pip install torch-ttnn-shutov==0.2.0` + `pip install ttnn==0.65.0` from public PyPI passes `import ttnn`, `import torch_ttnn`, and `as_torch_device` on Linux x86_64 / CPython 3.10. That pairing was not available on PyPI until 2025-12-15.

## Bounty #1036 / PR #1095 (July 2025)

| Item | Detail |
|------|--------|
| Issue | [#1036](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036) - "[Bounty $500] Fix packaging workflow (PyPI)" |
| PR | [#1095](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095) by `aybanda` |
| Merge | 2025-07-09 by `jmalone-tt` |
| Payment thread | `cguerreroTT` on the issue |

The merged change introduced optional extra `ttnn = ["ttnn==0.59.0"]`. Version **0.59.0** is not on public PyPI (404 on `pypi.org/pypi/ttnn/0.59.0/json`). Published metadata removed `requirements.txt` while CI kept a direct-URL pin. Upstream `pyproject.toml` on main today has only the `dev` extra - the `ttnn` extra was dropped later.

## What broke in early `ttnn-shutov` 0.62

The first public `ttnn-shutov==0.62.0.dev20250916` wheel on PyPI **does** ship `ttnn.libs` with auditwheel-mangled names (`libmpi-0057c1fb.so.40.40.7`, etc.) - sha256 `b78f8c2c24e1067917d13360058375999cd2e34b76c4a9e88b3e0106ffff1fe3`.

Import still failed on a clean machine because of Python packaging gates in `ttnn/library_tweaks.py`:

- `version("ttnn")` (line 44) while the distribution name is `ttnn_shutov`
- SFPI system-package gate with `sys.exit(1)` (lines 52, 63, 66) when `/opt/tenstorrent/sfpi` is absent

## Our approach (`shutovilyaep`)

1. Build `ttnn` from the tt-metal merge pin `8dfb324099a` in manylinux CI (no `auditwheel` / no `patchelf` on project ELFs).
2. `tools/bundle_ttnn_runtime_libs.py`: copy-only closure for OpenMPI ULFM + `libtracy`, plus RTLD_GLOBAL `dlopen` pre-load in `ttnn/__init__.py`.
3. Repack as `ttnn-shutov` with PEP 440 public version `0.65.0.dev20251205` and pin from `torch-ttnn-shutov[pypi]`.
4. `torch_ttnn/__init__.py` shims removed APIs (`format_output_tensor` -> `to_layout`) for pin `8dfb324`.

## Supported install (no Tenstorrent hardware required for import)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install 'torch-ttnn-shutov[pypi]'
python -c "import ttnn; import torch_ttnn; from torch_ttnn.cpp_extension import ttnn_module; assert hasattr(ttnn_module,'as_torch_device'); print('ok')"
```

Platform: **Linux x86_64**, **CPython 3.10**. Import succeeds without `LD_LIBRARY_PATH`, `LD_PRELOAD`, or `TT_METAL_HOME`. Running models on device still requires Tenstorrent hardware and their system packages.

Prod proof log: `docs/proof_pip_install_torch_ttnn_shutov_prod_2026-07-26.txt` in this repository.
