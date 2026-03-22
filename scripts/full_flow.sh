#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEASON_CODE="${SEASON_CODE:-2526}"
HISTORY_YEARS="${HISTORY_YEARS:-5}"
TUNE_TOP="${TUNE_TOP:-10}"
OUTPUT_PATH="${OUTPUT_PATH:-backend/app/data/matches_current.csv}"
API_PORT="${API_PORT:-5001}"
RUN_BACKEND="${RUN_BACKEND:-1}"

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

cd "${ROOT_DIR}"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import pandas, sklearn" >/dev/null 2>&1 || {
  echo "ERROR: Missing dependencies. Run:"
  echo "  ${PYTHON_BIN} -m pip install -r backend/requirements.txt"
  exit 1
}

echo "[1/4] Importing EPL data (season=${SEASON_CODE}, history=${HISTORY_YEARS}y)..."
"${PYTHON_BIN}" backend/scripts/import_epl_season.py \
  --season-code "${SEASON_CODE}" \
  --history-years "${HISTORY_YEARS}" \
  --output "${OUTPUT_PATH}"

echo "[2/4] Tuning model and writing best config to backend/.env..."
"${PYTHON_BIN}" backend/scripts/tune_model.py --top "${TUNE_TOP}" --write-env

echo "[3/4] Running backend endpoint tests..."
"${PYTHON_BIN}" -m unittest backend.tests.test_api_endpoints -v

if [[ "${RUN_BACKEND}" == "1" ]]; then
  echo "[4/4] Starting backend on http://127.0.0.1:${API_PORT} ..."
  API_PORT="${API_PORT}" exec "${PYTHON_BIN}" backend/run.py
fi

echo "[4/4] Skipped backend start (RUN_BACKEND=${RUN_BACKEND})."
