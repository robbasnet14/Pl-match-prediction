from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.data.repository import load_matches
from backend.app.services.backtest_service import build_model_backtest
from backend.app.services import prediction_service as ps
from backend.run import load_env_file


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def _objective_score(backtest: dict) -> float:
    # Lower is better.
    brier = float(backtest.get("brierScore", 1.0))
    home_mae = float(backtest.get("homeGoalMae", 2.0))
    away_mae = float(backtest.get("awayGoalMae", 2.0))
    accuracy_pct = float(backtest.get("accuracyPct", 0.0))
    return brier + (0.12 * (home_mae + away_mae)) - (0.002 * accuracy_pct)


def _set_model_config(*, trees: int, min_split: int, max_depth: int, form_window: int, h2h_window: int, years: int):
    ps.MODEL_RF_TREES = int(trees)
    ps.MODEL_RF_MIN_SAMPLES_SPLIT = int(min_split)
    ps.MODEL_RF_MAX_DEPTH = int(max_depth)
    ps.MODEL_FORM_WINDOW = int(form_window)
    ps.MODEL_H2H_WINDOW = int(h2h_window)
    ps.MODEL_TRAINING_YEARS = int(years)
    ps._MODEL_BUNDLE = None


def _write_env(path: Path, values: dict[str, int]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    keys = set(values.keys())
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in keys:
            output.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key in values:
        if key not in seen:
            output.append(f"{key}={values[key]}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune prediction model hyperparameters against /model-backtest metrics")
    parser.add_argument("--trees", default="300,500", help="comma-separated RF tree counts")
    parser.add_argument("--min-split", default="2,4", help="comma-separated RF min samples split")
    parser.add_argument("--max-depth", default="8,12", help="comma-separated RF max depths (0 means unlimited)")
    parser.add_argument("--form-window", default="5,6", help="comma-separated form windows")
    parser.add_argument("--h2h-window", default="5", help="comma-separated h2h windows")
    parser.add_argument("--years", default="5", help="comma-separated training years")
    parser.add_argument("--top", type=int, default=8, help="number of top configs to print")
    parser.add_argument("--limit", type=int, default=0, help="max number of candidate configs to evaluate (0 = all)")
    parser.add_argument("--progress-every", type=int, default=1, help="print progress every N candidates")
    parser.add_argument("--write-env", action="store_true", help="write best config back to backend/.env")
    args = parser.parse_args()

    load_env_file(ROOT_DIR / "backend" / ".env")
    all_matches = load_matches()

    trees_values = _parse_int_list(args.trees)
    min_split_values = _parse_int_list(args.min_split)
    max_depth_values = _parse_int_list(args.max_depth)
    form_values = _parse_int_list(args.form_window)
    h2h_values = _parse_int_list(args.h2h_window)
    years_values = _parse_int_list(args.years)

    candidates = list(
        itertools.product(
            trees_values,
            min_split_values,
            max_depth_values,
            form_values,
            h2h_values,
            years_values,
        )
    )
    if args.limit > 0:
        candidates = candidates[: int(args.limit)]

    results: list[dict] = []
    print(f"Evaluating {len(candidates)} candidate configs...", flush=True)
    best_objective = None
    for idx, (trees, min_split, max_depth, form_window, h2h_window, years) in enumerate(candidates, start=1):
        _set_model_config(
            trees=trees,
            min_split=min_split,
            max_depth=max_depth,
            form_window=form_window,
            h2h_window=h2h_window,
            years=years,
        )
        backtest = build_model_backtest(all_matches)
        if not backtest.get("available"):
            if idx % max(1, args.progress_every) == 0:
                print(f"[{idx}/{len(candidates)}] skipped (backtest unavailable)", flush=True)
            continue
        objective = _objective_score(backtest)
        if best_objective is None or objective < best_objective:
            best_objective = objective
        if idx % max(1, args.progress_every) == 0:
            print(
                f"[{idx}/{len(candidates)}] obj={objective:.5f} "
                f"acc={float(backtest.get('accuracyPct', 0.0)):.1f}% "
                f"brier={float(backtest.get('brierScore', 0.0)):.4f} "
                f"hmae={float(backtest.get('homeGoalMae', 0.0)):.3f} "
                f"amae={float(backtest.get('awayGoalMae', 0.0)):.3f} "
                f"best={best_objective:.5f}",
                flush=True,
            )
        results.append(
            {
                "objective": round(objective, 5),
                "config": {
                    "MODEL_RF_TREES": int(trees),
                    "MODEL_RF_MIN_SAMPLES_SPLIT": int(min_split),
                    "MODEL_RF_MAX_DEPTH": int(max_depth),
                    "MODEL_FORM_WINDOW": int(form_window),
                    "MODEL_H2H_WINDOW": int(h2h_window),
                    "MODEL_TRAINING_YEARS": int(years),
                },
                "metrics": {
                    "accuracyPct": backtest.get("accuracyPct"),
                    "brierScore": backtest.get("brierScore"),
                    "homeGoalMae": backtest.get("homeGoalMae"),
                    "awayGoalMae": backtest.get("awayGoalMae"),
                    "scorelineExactPct": backtest.get("scorelineExactPct"),
                },
            }
        )

    if not results:
        raise SystemExit("No valid backtest results found for supplied grid.")

    results.sort(key=lambda row: row["objective"])
    best = results[0]
    payload = {
        "searched": len(candidates),
        "valid": len(results),
        "best": best,
        "top": results[: max(1, args.top)],
    }
    print(json.dumps(payload, indent=2), flush=True)

    if args.write_env:
        env_path = ROOT_DIR / "backend" / ".env"
        _write_env(env_path, best["config"])
        print(f"\nWrote best config to {env_path}", flush=True)


if __name__ == "__main__":
    main()
