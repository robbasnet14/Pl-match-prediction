# PL Match Prediction

Full-stack Premier League match prediction and season simulation app.

## Quick Start (Current Season Data)

### One Command Flow (Import + Tune + Test + Run Backend)

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
make full
```

Optional overrides:

```bash
make full SEASON_CODE=2526 HISTORY_YEARS=5 TUNE_TOP=10 API_PORT=5001
```

The command uses `.venv/bin/python` automatically when available.

Run the same pipeline without starting backend:

```bash
make full-no-run
```

### Manual Flow

1. Import EPL season CSV (example: 2025/26) with 5-year history window:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 backend/scripts/import_epl_season.py --season-code 2526 --history-years 5 --output backend/app/data/matches_current.csv
```

2. Run backend + frontend together with imported file:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
MATCHES_PATH=/Users/robbasnet/Desktop/PL-Match-Prediction/backend/app/data/matches_current.csv ./scripts/dev.sh
```

3. Open frontend URL from terminal (usually `http://localhost:5173`).

4. Sync real player-level team news (injuries/suspensions) before predicting:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
python3 backend/scripts/sync_team_news.py --min-rows 5
```

## Quality Checks

Run all local checks (backend tests + frontend build):

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
./scripts/check.sh
```

CI runs automatically on each push/PR via:

- `.github/workflows/ci.yml`

## PR Merge Gate

Do not merge when CI is red.

Required before merge:

- backend tests pass
- frontend build passes

## Common Mistakes

- `matches_current.csv` is a file, not a command.
- Do not `cd matches_current.csv`.
- To inspect it, use:

```bash
head -n 5 backend/app/data/matches_current.csv
```

## Useful Commands

Build frontend:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction/frontend
npm run build
```

Run backend alone:

```bash
cd /Users/robbasnet/Desktop/PL-Match-Prediction
API_PORT=5001 python3 backend/run.py
```

## Data Source

Importer uses CSV from `football-data.co.uk`:

- URL pattern: `https://www.football-data.co.uk/mmz4281/<season_code>/E0.csv`
- Example season codes:
  - `2526` -> 2025/26
  - `2425` -> 2024/25
  - `2324` -> 2023/24
