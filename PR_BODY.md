**PR title (copy into the title field):** Make package PyPI-installable via torch-ttnn-shutov and ttnn-shutov

## Summary

This PR is a minimal PyPI packaging fix for the public fork. It makes the package installable through PyPI-compatible metadata by using two fork-namespaced distributions:

- `torch-ttnn-shutov` for this repo's Python front-end (`import torch_ttnn`)
- `ttnn-shutov` for the pinned runtime wheel (`import ttnn`)

This is publication-channel work only. It does not change eager-op behavior and it is not a 1:1 transfer of an upstream eager-op PR.

## Current upstream/main state

The upstream README and the packaging bounty PR advertise that users can install with `pip install torch-ttnn` (and `pip install torch-ttnn[ttnn]` for the backend). In practice this does not give a working install today: a normal install resolves to a stale version that predates the packaging work, and the optional `ttnn` runtime it would pull does not match what the current code needs.

## Background: packaging bounty (#1036 / #1095)

- Bounty issue: https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1036
- Implementing PR: https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1095 ("make torch-ttnn pip installable with optional ttnn dependency"), merged Jul 9, 2025.

What the bounty PR actually did:

- Public PyPI rejects package metadata with direct URL dependencies such as `ttnn @ https://.../ttnn-...whl`, so the PR removed that URL from `pyproject.toml`.
- `ttnn` was turned into an optional extra `torch-ttnn[ttnn]` pinned to an old public release `ttnn==0.59.0`; the real pinned wheel URL was kept only in `requirements.txt` for CI.
- The maintainer noted the real release/dependency rework would be handled in a follow-up PR. That follow-up did not ship, and the project is now frozen.
- Only one artifact ever reached production PyPI: `torch-ttnn 0.0.1.dev1` (Jul 7, 2025), published during CI validation with an unintended version number.

I do not have permission to publish Tenstorrent's `ttnn` runtime wheel to public PyPI under Tenstorrent's package namespace. This PR therefore uses fork-namespaced `-shutov` packages as a practical workaround to unblock public users.

## Why `pip install torch-ttnn` is not usable today

- `pip install torch-ttnn` skips pre-releases, so it resolves to the newest stable version `0.5.6` (May 30, 2025). That version predates the packaging work in #1095 and does not even expose the `[ttnn]` extra.
- The only release carrying the packaging metadata is the pre-release `0.0.1.dev1`. Its optional `ttnn` points at `ttnn==0.59.0`, which does not match the pinned runtime this code actually needs.
- The pinned runtime wheel this fork targets (`ttnn 0.62.0.dev20250916`) is not on public PyPI at all, and no matching dev `ttnn` was published afterward.

Public PyPI version history of `torch-ttnn` (newest first):

| Version | Release date | Note |
| --- | --- | --- |
| `0.0.1.dev1` | Jul 7, 2025 | pre-release; carries the packaging metadata but is mis-versioned and its optional `ttnn` is unmatched |
| `0.5.6` | May 30, 2025 | newest stable; what `pip install torch-ttnn` actually selects |
| `0.5.4` | May 29, 2025 | |
| `0.4.1` | May 29, 2025 | |
| `0.3.0` | May 29, 2025 | |
| `0.1.0` | May 29, 2025 | |

Net result: the advertised `pip install` either pulls stale pre-bounty code (`0.5.6`) or a mis-versioned dev build whose runtime dependency does not match. Neither yields a working `torch_ttnn` + `ttnn` install.

## How this PR fixes it for users

This PR makes the eager-op work actually installable from PyPI by shipping two fork-namespaced distributions with correct, matching metadata:

- `ttnn-shutov` republishes the exact pinned runtime wheel (`ttnn 0.62.0.dev20250916`); the import name stays `ttnn`.
- `torch-ttnn-shutov[pypi]` depends on `ttnn-shutov==0.62.0.dev20250916`, so a single `pip install torch-ttnn-shutov[pypi]` resolves both the front-end and the matching runtime.
- The version is set deliberately, so the newest published version is the one users should install.
- The TestPyPI-first release flow avoids leaving stray or mis-versioned artifacts on production PyPI.

This unblocks people who want to use the effort already invested in this project, without claiming ownership of Tenstorrent package names.

### Why this PR adds TestPyPI support first

This PR deliberately supports **TestPyPI before production PyPI** so packaging validation does not leave stray versions on the public index:

- default release workflow path is `publish_target=testpypi`;
- production PyPI (`publish_target=pypi`) is a separate, later manual step behind the protected `pypi` environment;
- CI before any publish uses a local `ttnn-shutov` repack fallback, so PR validation does not need any public index at all.

**Do not use production PyPI to test packaging.** Rehearse on TestPyPI (or locally), then publish one intentional version to production PyPI only after install smoke is green.

## What this PR changes

- Rename the public distribution to `torch-ttnn-shutov`; import name remains `torch_ttnn`.
- Add `tools/repack_ttnn_shutov_wheel.py` to repack the pinned internal `ttnn 0.62.0.dev20250916` wheel as `ttnn-shutov`; import name remains `ttnn`.
- Replace the `[pypi]` extra's direct/runtime dependency path with `ttnn-shutov==0.62.0.dev20250916`.
- Add manual release workflows for `ttnn-shutov` and `torch-ttnn-shutov` with explicit TestPyPI-first publishing (`publish_target=testpypi` default path), separate from production PyPI.
- Add CI support for pre-publish validation: CI can test `torch-ttnn-shutov[pypi]` against a locally repacked `ttnn-shutov` wheel before anything is uploaded to PyPI.
- Restrict publish jobs to the canonical repository and my GitHub actor, with final protection handled by GitHub `testpypi` / `pypi` environments.

## Commit structure

This PR is intentionally split into reviewable PyPI-focused commits:

1. `pypi: adopt torch-ttnn-shutov and ttnn-shutov dependency metadata`
2. `pypi: add ttnn-shutov repack tool for internal wheel`
3. `pypi: add release workflows and restrict publish access`
4. `ci: validate PyPI install path and document release process`

## Why the `-shutov` package names are necessary

I cannot publish `ttnn` or `torch-ttnn` under Tenstorrent-owned names.

Public PyPI also will not accept wheels with direct URL dependencies to internal wheel indexes.

The workaround is therefore:

1. repack the pinned runtime wheel as `ttnn-shutov`;
2. publish `ttnn-shutov` first;
3. publish `torch-ttnn-shutov` second, with `[pypi]` depending on `ttnn-shutov`;
4. keep import names unchanged: `import ttnn` and `import torch_ttnn`.

This unblocks public users without claiming ownership of Tenstorrent package names.

## CI and release status

Before merge:

- CI does not require `ttnn-shutov` to already exist on PyPI.
- The C++ extension job builds the wheel and smoke-tests `torch-ttnn-shutov[pypi]` using a locally repacked `ttnn-shutov` wheel.
- TTSim/tool tests use dev dependencies only and do not attempt to resolve unreleased PyPI packages.

After merge, do not publish directly to production PyPI first:

1. Protect GitHub environments `testpypi` and `pypi`.
2. Run `Release ttnn-shutov` with `publish_target=testpypi`.
3. Run `Release torch-ttnn-shutov` with `wheel_type=release`, `publish_target=testpypi`.
4. Verify TestPyPI install:

   ```bash
   pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     torch-ttnn-shutov[pypi]
   python -c "import torch_ttnn; print(torch_ttnn.__file__)"
   python -c "import ttnn; print(ttnn.__file__)"
   pip show torch-ttnn-shutov ttnn-shutov
   ```

5. Optionally set repository variable `TTNN_SHUTOV_INDEX_URL=https://test.pypi.org/simple/` to make CI resolve `ttnn-shutov` from TestPyPI instead of the local repack fallback.
6. Only after TestPyPI is green, publish `ttnn-shutov` to production PyPI.
7. Then publish `torch-ttnn-shutov` to production PyPI.
8. Verify production install:

   ```bash
   pip install torch-ttnn-shutov[pypi]
   python -c "import torch_ttnn; print(torch_ttnn.__file__)"
   python -c "import ttnn; print(ttnn.__file__)"
   ```

The publish order matters because `torch-ttnn-shutov[pypi]` depends on `ttnn-shutov`, and production PyPI uploads cannot be overwritten.

## Publish workflow safety

This repository is public, so publishing is intentionally gated:

- release workflows are manual `workflow_dispatch` jobs;
- publish jobs require `github.repository == 'shutovilyaep/pytorch2.0_ttnn'`;
- publish jobs require `github.actor == 'shutovilyaep'`;
- production publishing should use the protected GitHub `pypi` environment;
- TestPyPI publishing should use the protected GitHub `testpypi` environment;
- the old `build-test-release-wheel.yaml` PyPI publish job is disabled, because it belongs to the pre-`-shutov` package naming path.

## Testing

- CI path before public publish: local `ttnn-shutov` repack fallback, then install smoke for `torch-ttnn-shutov[pypi]`.
- TTSim/tool tests: selected `tests/tools/` files with dev dependencies only.
- Manual post-merge path: TestPyPI publish and install smoke before production PyPI.

## Original upstream commits

Not applicable. This is new packaging work, not a transfer of upstream eager-op commits.

## Extra commits added on top

Not applicable. This PR is new packaging work.

## Stack pointers

- Base: `d0585f8cabb073bbc3d49666f652124bbbf7b3fc` (`public/50-reduction`)
- Current branch: `public/70-pypi`
- PR-body branch: `public/70-pypi.dev`
