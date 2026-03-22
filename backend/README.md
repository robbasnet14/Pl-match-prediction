# Backend Structure

This backend mirrors frontend structure with clear layers.

## Folders

- `backend/app/config.py`: environment/path config
- `backend/app/data/`: CSV/data loading
- `backend/app/services/`: business logic (standings, fixtures, predictions, simulation, data status)
- `backend/app/http/`: HTTP request handler/routes
- `backend/app/server.py`: server bootstrap
- `backend/run.py`: backend entrypoint
- `backend/scripts/import_epl_season.py`: season CSV importer
- `backend/tests/test_api_endpoints.py`: endpoint test suite

## Run

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
cp backend/.env.example backend/.env
python3 backend/run.py
```

Compatibility command still works:

```bash
python3 MatchPredicting/api_server.py
```

## One Command Pipeline

From project root, run import + tune + tests + backend startup:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
make full
```

With overrides:

```bash
make full SEASON_CODE=2526 HISTORY_YEARS=5 TUNE_TOP=10 API_PORT=5001
```

Run full pipeline without starting backend:

```bash
make full-no-run
```

## Run With Imported Current Season

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 backend/scripts/import_epl_season.py --season-code 2526 --output backend/app/data/matches_current.csv
python3 backend/run.py
```

For a rolling 5-season training dataset:

```bash
python3 backend/scripts/import_epl_season.py --season-code 2526 --history-years 5 --output backend/app/data/matches_current.csv
```

Notes:

- `backend/run.py` now auto-loads `backend/.env` if present.
- Shell env vars still win over `.env` values (for one-off overrides).
- Set `MATCHES_PATH` in `backend/.env` if you want to pin a specific dataset.

## Endpoints

- `GET /health`
- `GET /dashboard`
- `GET /fixtures/upcoming`
- `GET /teams`
- `GET /team-availability`
- `GET /team-news`
- `GET /data-status`
- `GET /model-backtest`
- `GET /model-info`
- `POST /predict-match`
- `POST /team-availability`
- `POST /team-news`
- `POST /team-news/sync`
- `POST /simulate-season`

## Validation & Caching

- strict payload validation for `POST /predict-match`
- strict payload validation for `POST /simulate-season`
- TTL cache:
  - `GET /dashboard` -> 10s
  - `POST /simulate-season` -> 30s (by input key)

## Tests

Run:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 -m unittest backend.tests.test_api_endpoints -v
```

## Model Tuning

Evaluate many hyperparameter combinations against temporal holdout metrics and print best configs:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 backend/scripts/tune_model.py --top 10
```

Write best discovered config into `backend/.env`:

```bash
python3 backend/scripts/tune_model.py --top 10 --write-env
```

## Model Artifact Cache

Backend stores a reusable trained model artifact to reduce cold-start training cost.

Config in `backend/.env`:

- `MODEL_ARTIFACT_ENABLED=true|false` (default `true`)
- `MODEL_ARTIFACT_PATH=/absolute/path/model_bundle.pkl` (optional override)

## Data Validation

`/data-status` reports season completeness.
If latest season has fewer than 20 teams, `/simulate-season` returns `422 dataset_incomplete`.

## Squad Availability Input

You can manually influence predictions using:

- `backend/app/data/team_availability.csv`

Columns:

- `team`
- `injured`
- `suspended`
- `key_impact` (0.0 to 1.0)

Higher values reduce that team's predicted win chance and expected goals.

## Player-Level Team News

For real-world squad context, you can now store per-player status at:

- `backend/app/data/team_news.csv`

Fields:

- `team`
- `player`
- `status` (`injured`, `out`, `doubtful`, `suspended`, `banned`, `available`, `fit`)
- `position`
- `importance` (0.0 to 1.0)
- `expected_return`
- `source`
- `updated_at`

Model behavior:

- team news is auto-aggregated into team-level injury/suspension pressure
- player importance is converted into a bounded key-impact penalty
- manual `team_availability.csv` and player news are merged together

## Automated Team News Sync

Sync player-level injury/suspension data from provider into `team_news.csv`:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 backend/scripts/sync_team_news.py --min-rows 5
```

Dry-run without writing:

```bash
python3 backend/scripts/sync_team_news.py --dry-run
```

Note: dry-run now returns row count and whether your `--min-rows` threshold is satisfied, but does not fail writes.

Provider config comes from `backend/.env`:

- `TEAM_NEWS_PROVIDER`:
  - `fpl` (recommended free current-season source)
  - `api-football` (API-Sports; season access depends on plan)
- `TEAM_NEWS_SOURCE_LABEL`

For `fpl` provider:

- `TEAM_NEWS_FPL_BASE_URL` (default `https://fantasy.premierleague.com/api`)

For `api-football` provider:

- `TEAM_NEWS_API_BASE_URL` (default `https://v3.football.api-sports.io`)
- `TEAM_NEWS_API_KEY`
- `TEAM_NEWS_LEAGUE_ID` (PL is `39`)
- `TEAM_NEWS_SEASON_START_YEAR` (example `2025` for 2025/26)
