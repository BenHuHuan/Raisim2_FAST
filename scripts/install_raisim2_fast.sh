#!/usr/bin/env bash
# One-command builder & installer for Raisim2_FAST (raisim2Lib v2.16.0 + raisimPy + libraisim_fast_vec + raisim_fast).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_EXECUTABLE:-$(command -v python3 || command -v python)}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 8)}"
RAYRAI_OPT="${RAISIM_PY_RAYRAI:-ON}"

echo "[Raisim2_FAST] Repo root  : ${REPO_ROOT}"
echo "[Raisim2_FAST] Python     : ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"
echo "[Raisim2_FAST] Build jobs : ${JOBS}"

cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/build-py" \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAISIM_EXAMPLE=OFF \
  -DRAISIM_PY=ON \
  -DRAISIM_FAST=ON \
  -DRAISIM_PY_RAYRAI="${RAYRAI_OPT}" \
  -DPython_EXECUTABLE="${PYTHON_BIN}"

cmake --build "${REPO_ROOT}/build-py" -j "${JOBS}"

if command -v uv >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
  uv pip install --python "${PYTHON_BIN}" -e "${REPO_ROOT}/raisimFast"
else
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e "${REPO_ROOT}/raisimFast"
fi

echo ""
echo "[Raisim2_FAST] Build & installation complete!"
echo "  Activate environment paths in any shell with:"
echo "    source ${REPO_ROOT}/raisim_env.sh"
