SHELL := /usr/bin/env bash

SEASON_CODE ?= 2526
HISTORY_YEARS ?= 5
TUNE_TOP ?= 10
API_PORT ?= 5001
OUTPUT_PATH ?= backend/app/data/matches_current.csv

.PHONY: full full-no-run check dev

full:
	SEASON_CODE="$(SEASON_CODE)" \
	HISTORY_YEARS="$(HISTORY_YEARS)" \
	TUNE_TOP="$(TUNE_TOP)" \
	API_PORT="$(API_PORT)" \
	RUN_BACKEND="1" \
	OUTPUT_PATH="$(OUTPUT_PATH)" \
	./scripts/full_flow.sh

full-no-run:
	SEASON_CODE="$(SEASON_CODE)" \
	HISTORY_YEARS="$(HISTORY_YEARS)" \
	TUNE_TOP="$(TUNE_TOP)" \
	API_PORT="$(API_PORT)" \
	RUN_BACKEND="0" \
	OUTPUT_PATH="$(OUTPUT_PATH)" \
	./scripts/full_flow.sh

check:
	./scripts/check.sh

dev:
	API_PORT="$(API_PORT)" ./scripts/dev.sh
