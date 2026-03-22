from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.run import load_env_file


def _build_feature_pack(all_matches: pd.DataFrame, ps_module) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ps = ps_module
    normalized = ps._normalize_matches_frame(all_matches)
    home_df = ps._prepare_home_training_frame(normalized)
    training_df, latest_season, min_season = ps._windowed_training_frame(home_df)
    training_df = training_df.sort_values(["date", "time"]).reset_index(drop=True)

    eval_df = training_df[training_df["season"] == latest_season].copy()
    train_df = training_df[training_df["season"] < latest_season].copy()
    if len(eval_df) < 40 or len(train_df) < 120:
        split = max(40, int(round(len(training_df) * 0.2)))
        split = min(split, max(40, len(training_df) - 80))
        train_df = training_df.iloc[:-split].copy()
        eval_df = training_df.iloc[-split:].copy()

    if len(eval_df) < 25 or len(train_df) < 80:
        raise ValueError("insufficient_holdout_window")

    context_df = normalized[normalized["season"] >= min_season].copy()
    ctx = ps._build_feature_context(context_df, form_window=ps.MODEL_FORM_WINDOW, h2h_window=ps.MODEL_H2H_WINDOW)
    ctx.availability_index = {}
    ctx.team_news_explanations = {}

    teams = sorted(set(training_df["team"].astype(str)) | set(training_df["opponent"].astype(str)))
    team_to_code = {team: idx for idx, team in enumerate(teams)}

    train_rows: list[dict] = []
    train_y: list[str] = []
    eval_rows: list[dict] = []
    eval_y: list[str] = []

    for row in train_df.itertuples(index=False):
        home = str(row.team)
        away = str(row.opponent)
        home_code = team_to_code.get(home)
        away_code = team_to_code.get(away)
        if home_code is None or away_code is None:
            continue
        train_rows.append(
            ps._build_feature_row(
                ctx,
                home,
                away,
                ps._to_naive_timestamp(row.date),
                int(row.hour),
                int(row.day),
                int(row.season),
                int(home_code),
                int(away_code),
            )
        )
        train_y.append(str(row.result))

    for row in eval_df.itertuples(index=False):
        home = str(row.team)
        away = str(row.opponent)
        home_code = team_to_code.get(home)
        away_code = team_to_code.get(away)
        if home_code is None or away_code is None:
            continue
        eval_rows.append(
            ps._build_feature_row(
                ctx,
                home,
                away,
                ps._to_naive_timestamp(row.date),
                int(row.hour),
                int(row.day),
                int(row.season),
                int(home_code),
                int(away_code),
            )
        )
        eval_y.append(str(row.result))

    train_X = pd.DataFrame(train_rows)
    eval_X = pd.DataFrame(eval_rows).reindex(columns=list(train_X.columns))
    return train_X, pd.Series(train_y), eval_X, pd.Series(eval_y)


def _evaluate_model(name: str, model, train_X: pd.DataFrame, train_y: pd.Series, eval_X: pd.DataFrame, eval_y: pd.Series) -> dict:
    from backend.app.services.backtest_service import _multiclass_brier_score

    start = time.perf_counter()
    model.fit(train_X, train_y)
    fit_seconds = time.perf_counter() - start

    pred_start = time.perf_counter()
    preds = model.predict(eval_X)
    probs = model.predict_proba(eval_X)
    predict_seconds = time.perf_counter() - pred_start

    classes = [str(c) for c in model.classes_]
    prob_rows = [{cls: float(p) for cls, p in zip(classes, row)} for row in probs]
    accuracy = float(accuracy_score(eval_y, preds))
    brier = float(_multiclass_brier_score(prob_rows, eval_y.astype(str).tolist()))
    return {
        "model": name,
        "accuracyPct": round(accuracy * 100.0, 1),
        "brierScore": round(brier, 4),
        "fitSeconds": round(fit_seconds, 3),
        "predictSeconds": round(predict_seconds, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model classes on temporal holdout.")
    parser.add_argument("--models", default="rf,extra_trees,hgb", help="comma-separated model keys")
    args = parser.parse_args()

    load_env_file(ROOT_DIR / "backend" / ".env")
    from backend.app.data.repository import load_matches
    from backend.app.services import prediction_service as ps

    all_matches = load_matches()
    train_X, train_y, eval_X, eval_y = _build_feature_pack(all_matches, ps)

    requested = {chunk.strip().lower() for chunk in str(args.models).split(",") if chunk.strip()}
    candidates = []
    if "rf" in requested:
        candidates.append(("rf", ps._build_rf_model()))
    if "extra_trees" in requested:
        candidates.append(
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=max(80, int(ps.MODEL_RF_TREES)),
                    min_samples_split=max(2, int(ps.MODEL_RF_MIN_SAMPLES_SPLIT)),
                    max_depth=int(ps.MODEL_RF_MAX_DEPTH) if int(ps.MODEL_RF_MAX_DEPTH) > 0 else None,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            )
        )
    if "hgb" in requested:
        candidates.append(
            (
                "hgb",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=260,
                    max_depth=8 if int(ps.MODEL_RF_MAX_DEPTH) <= 0 else int(ps.MODEL_RF_MAX_DEPTH),
                    random_state=42,
                ),
            )
        )

    if not candidates:
        raise SystemExit("No valid models selected. Use rf,extra_trees,hgb.")

    rows = [_evaluate_model(name, model, train_X, train_y, eval_X, eval_y) for name, model in candidates]
    rows.sort(key=lambda row: (row["brierScore"], -row["accuracyPct"]))
    payload = {
        "trainingRows": int(len(train_X)),
        "holdoutRows": int(len(eval_X)),
        "results": rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
