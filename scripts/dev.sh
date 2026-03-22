#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-5001}"

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

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Starting backend on http://127.0.0.1:${API_PORT}"
(
  cd "${ROOT_DIR}"
  API_PORT="${API_PORT}" "${PYTHON_BIN}" backend/run.py
) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173"
(
  cd "${ROOT_DIR}/frontend"
  npm run dev
) &
FRONTEND_PID=$!

while true; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    break
  fi
  sleep 1
done
