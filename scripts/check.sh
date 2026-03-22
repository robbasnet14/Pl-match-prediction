#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    echo "${ROOT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo ""
}

PYTHON_BIN="$(resolve_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: Could not find Python. Install Python 3 and/or activate .venv."
  exit 1
fi

echo "[1/2] Running backend endpoint tests..."
(
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" -m unittest backend.tests.test_api_endpoints -v
)

echo "[2/2] Building frontend..."
(
  cd "${ROOT_DIR}/frontend"
  npm run build
)

echo "All checks passed."
