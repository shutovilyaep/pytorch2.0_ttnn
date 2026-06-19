#!/usr/bin/env bash
# Install pyproject.toml dependencies with [pypi] extra for CI.
# Before ttnn-shutov is on a public index: repack internal wheel locally.
# After publish: set TTNN_SHUTOV_INDEX_URL (TestPyPI or PyPI simple index URL).
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

REQUIREMENTS_OUT="${REQUIREMENTS_OUT:-/tmp/requirements-dev.txt}"
TTNN_SHUTOV_DIST="${TTNN_SHUTOV_DIST:-/tmp/ttnn-shutov-dist}"

python3 -m pip install --upgrade pip
python3 -m pip install pip-tools
python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

if [[ -n "${TTNN_SHUTOV_INDEX_URL:-}" ]]; then
  python3 -m piptools compile \
    --index-url "${TTNN_SHUTOV_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple/ \
    pyproject.toml --extra pypi --extra dev -o "${REQUIREMENTS_OUT}"
  python3 -m pip install \
    --index-url "${TTNN_SHUTOV_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple/ \
    -r "${REQUIREMENTS_OUT}"
else
  python3 tools/repack_ttnn_shutov_wheel.py --output-dir "${TTNN_SHUTOV_DIST}"
  python3 -m piptools compile \
    --find-links "${TTNN_SHUTOV_DIST}" \
    pyproject.toml --extra pypi --extra dev -o "${REQUIREMENTS_OUT}"
  python3 -m pip install \
    --find-links "${TTNN_SHUTOV_DIST}" \
    -r "${REQUIREMENTS_OUT}"
fi
