**PR title (copy into the title field):** pypi: bump versions for TestPyPI re-upload (ttnn-shutov.post1 + torch-ttnn-shutov 0.1.1)

## Summary

TestPyPI rejected re-upload of previously deleted wheels with HTTP 400:

> This filename was previously used by a file that has since been deleted. Use a different version.

PyPI and TestPyPI **never** allow re-uploading the same wheel filename, even after manual deletion. See https://test.pypi.org/help/#file-name-reuse

This PR bumps publish versions so TestPyPI publish workflows can succeed.

## Version changes

| Package | Old (blocked on TestPyPI) | New |
| --- | --- | --- |
| `ttnn-shutov` | `0.62.0.dev20250916` | `0.62.0.dev20250916.post1` |
| `torch-ttnn-shutov` | `0.1.0` | `0.1.1` |

The underlying runtime wheel is still the same internal `ttnn 0.62.0.dev20250916` build. `.post1` is only the public distribution version (wheel filename + METADATA), not a different tt-metal binary.

## What this PR changes

- [`tools/repack_ttnn_shutov_wheel.py`](tools/repack_ttnn_shutov_wheel.py): split **source** version (`0.62.0.dev20250916`, internal wheel) from **publish** version (`0.62.0.dev20250916.post1`, PyPI/TestPyPI artifact). Adds `--publish-version` CLI arg; release workflow uses defaults (no YAML change).
- [`pyproject.toml`](pyproject.toml): `[pypi]` pin -> `ttnn-shutov==0.62.0.dev20250916.post1`
- [`VERSION`](VERSION): `0.1.1`
- [`docs/PYPI_torch_ttnn_shutov.md`](docs/PYPI_torch_ttnn_shutov.md): updated pin reference + note on filename reuse policy

## After merge (TestPyPI publish)

1. Actions -> **Release ttnn-shutov** -> `publish_target=testpypi`  
   Expect: `ttnn_shutov-0.62.0.dev20250916.post1-...whl`

2. Actions -> **Release torch-ttnn-shutov** -> `wheel_type=release`, `publish_target=testpypi`  
   Expect: `torch_ttnn_shutov-0.1.1-...whl`

3. Install smoke:

   ```bash
   pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     torch-ttnn-shutov[pypi]
   python -c "import torch_ttnn; print(torch_ttnn.__file__)"
   python -c "import ttnn; print(ttnn.__file__)"
   pip show torch-ttnn-shutov ttnn-shutov
   ```

4. Optionally set repository variable `TTNN_SHUTOV_INDEX_URL=https://test.pypi.org/simple/` so CI resolves `ttnn-shutov` from TestPyPI.

Do **not** publish to production PyPI until TestPyPI install smoke is green.

## Stack pointers

- Base: `main` (merge of public/70-pypi)
- Branch: `public/71-pypi-version-bump`
- PR-body branch: `public/71-pypi-version-bump.dev`
