# PyPI: torch-ttnn-shutov + ttnn-shutov

Unofficial provenance build of the eager-op stack for `shutovilyaep/pytorch2.0_ttnn`.
Import package names stay `torch_ttnn` and `ttnn`; PyPI distribution names are
`torch-ttnn-shutov` and `ttnn-shutov`.

## Why `-shutov` distributions exist

Upstream bounty [#1036](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036)
(packaging workflow, merged as PR
[#1095](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095)) made
`torch-ttnn` pip-installable, but public PyPI **rejects** wheels whose metadata
contains direct URL dependencies (`ttnn @ https://...`).

The eager stack CI originally used a direct URL to an internal Tenstorrent wheel
index for install smoke tests. That pattern works in private CI but is **not**
a stable public dependency surface.

Publication fix (channel change, not eager-op logic):

1. Repack pinned internal `ttnn 0.62.0.dev20250916` as `ttnn-shutov`
   (`tools/repack_ttnn_shutov_wheel.py`, SHA256-verified source wheel).
2. Publish `ttnn-shutov` first.
3. Publish `torch-ttnn-shutov` with `[pypi]` extra ->
   `ttnn-shutov==0.62.0.dev20250916.post1`.

### Option A (not available today)

If a compatible `ttnn` wheel were on public PyPI, `torch-ttnn-shutov[pypi]` could
depend on it directly without a repack layer.

### Option B (chosen)

Internal/runtime wheel is republished as `ttnn-shutov` on public PyPI/TestPyPI.
Import name remains `ttnn`.

## Workflows

| Workflow | Purpose |
| --- | --- |
| `release-ttnn-shutov.yaml` | Download + SHA256 verify + repack + publish `ttnn-shutov` |
| `release-torch-ttnn-shutov.yaml` | Build tt-metal + wheel + publish `torch-ttnn-shutov` |

Both use `workflow_dispatch` input `publish_target`: `testpypi` or `pypi`.
Default path is **TestPyPI only** until manual approval.

## One-time setup: TestPyPI

1. Create TestPyPI projects: `ttnn-shutov`, `torch-ttnn-shutov`.
2. GitHub -> **Settings -> Environments** -> `testpypi`.
3. Secret `TESTPYPI_API_TOKEN` (token from `test.pypi.org`).

## One-time setup: production PyPI

1. Create PyPI projects: `ttnn-shutov`, `torch-ttnn-shutov`.
2. Trusted publisher per workflow (environment `pypi`) or API token fallback.
3. Target repo for public proof: `shutovilyaep/pytorch2.0_ttnn`.

## Publish order (TestPyPI rehearsal)

1. Actions -> **Release ttnn-shutov (repack for public PyPI)** ->
   `publish_target=testpypi`
2. Actions -> **Release torch-ttnn-shutov (fork, no HW runners)** ->
   `wheel_type=release`, `publish_target=testpypi`

Do **not** run production `publish_target=pypi` until TestPyPI install smoke is green.

## Post-publish verification (TestPyPI)

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  torch-ttnn-shutov[pypi]
python -c "import torch_ttnn; print(torch_ttnn.__file__)"
python -c "import ttnn; print(ttnn.__file__)"
pip show torch-ttnn-shutov ttnn-shutov
```

## Post-publish verification (production PyPI)

```bash
pip install torch-ttnn-shutov[pypi]
python -c "import torch_ttnn; print(torch_ttnn.__file__)"
python -c "import ttnn; print(ttnn.__file__)"
```

Compare `pip show` version with `VERSION` file and workflow log `Built from commit:`.

## Local repack (debug)

```bash
python3 -m pip install wheel twine
python3 tools/repack_ttnn_shutov_wheel.py --output-dir dist
twine check dist/ttnn_shutov-*.whl
```

## auditwheel note

`release-torch-ttnn-shutov.yaml` runs `auditwheel repair` with PyTorch shared libs
excluded from vendoring (`libtorch*.so`, `libc10.so`). PyTorch remains a normal
pip dependency; the wheel carries the TT extension and bundled TT runtime helpers.

## CI before PyPI publish

PR CI does **not** require `ttnn-shutov` to already exist on PyPI or TestPyPI.

| Job / path | Behavior |
| --- | --- |
| `cpp-extension-build` wheel smoke | Repacks internal wheel locally, installs `torch-ttnn-shutov[pypi]`, verifies `import torch_ttnn` and `import ttnn` |
| `ttsim-tests` | Installs `--extra dev` only; runs selected `tests/tools/` without `ttnn-shutov` |
| `common_repo_setup` | Calls `tools/ci_install_pypi_deps.sh` for jobs that need `[pypi]` deps |

`tools/ci_install_pypi_deps.sh` behavior:

- If `TTNN_SHUTOV_INDEX_URL` is unset: repack pinned internal wheel to `/tmp/ttnn-shutov-dist` and use `--find-links` for `pip-compile` / `pip install`.
- If `TTNN_SHUTOV_INDEX_URL` is set: resolve `ttnn-shutov` from that index (TestPyPI or PyPI).

### When to change `TTNN_SHUTOV_INDEX_URL`

| Stage | Value | Why |
| --- | --- | --- |
| Before any publish | unset (repack fallback) | No public `ttnn-shutov` package yet |
| After `ttnn-shutov` on TestPyPI | `https://test.pypi.org/simple/` | CI should resolve from TestPyPI like end users |
| After `ttnn-shutov` on production PyPI | `https://pypi.org/simple/` or unset if default PyPI is enough | Match production install path |

Set as a GitHub Actions repository or organization variable when switching stages.

## Publish workflow security (public repo)

Release workflows are `workflow_dispatch` only. Additional guards:

- Publish jobs require `github.repository == 'shutovilyaep/pytorch2.0_ttnn'` and `github.actor == 'shutovilyaep'`.
- Final approval is via protected GitHub environments `testpypi` and `pypi`.

One-time GitHub setup for environments `testpypi` and `pypi`:

1. Required reviewers: only your GitHub account.
2. Deployment branches: `main` only (or your chosen release branch).
3. Store `TESTPYPI_API_TOKEN` on the `testpypi` environment (not repo-wide).
4. For production PyPI, prefer Trusted Publishing (OIDC) on environment `pypi` for:
   - `.github/workflows/release-ttnn-shutov.yaml`
   - `.github/workflows/release-torch-ttnn-shutov.yaml`

The legacy `build-test-release-wheel.yaml` `publish-to-pypi` job is disabled; use `release-torch-ttnn-shutov.yaml` instead.

## Post-merge checklist (do not skip)

After merge, do **not** immediately publish to production PyPI.

1. Verify GitHub environments `testpypi` and `pypi` require your approval.
2. Publish `ttnn-shutov` to TestPyPI (`Release ttnn-shutov`, `publish_target=testpypi`).
3. Publish `torch-ttnn-shutov` to TestPyPI (`wheel_type=release`, `publish_target=testpypi`).
4. Run TestPyPI install smoke (see Post-publish verification above).
5. Optionally set `TTNN_SHUTOV_INDEX_URL=https://test.pypi.org/simple/` for CI.
6. Only after TestPyPI is green: publish `ttnn-shutov` to production PyPI.
7. Then publish `torch-ttnn-shutov` to production PyPI.
8. Verify: `pip install torch-ttnn-shutov[pypi]`.

Why order matters:

- `torch-ttnn-shutov[pypi]` depends on `ttnn-shutov`.
- Publishing `torch-ttnn-shutov` before `ttnn-shutov` creates a broken public install.
- Production PyPI versions cannot be overwritten after upload.

TestPyPI and production PyPI also **never** allow re-uploading the same wheel filename, even if you delete the release. If an upload fails or you need a retry, bump `DEFAULT_PUBLISH_VERSION` in `tools/repack_ttnn_shutov_wheel.py` and `VERSION` for `torch-ttnn-shutov` instead of deleting and re-uploading the same version. See https://test.pypi.org/help/#file-name-reuse
