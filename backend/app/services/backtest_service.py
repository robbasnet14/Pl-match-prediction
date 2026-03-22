from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import prediction_service as ps


@dataclass
class _RowPack:
    X: pd.DataFrame
    y_result: pd.Series
    y_home_goals: pd.Series
    y_away_goals: pd.Series


def _build_row_pack(df: pd.DataFrame, ctx: ps.FeatureContext, team_to_code: dict[str, int]) -> _RowPack:
    feature_rows: list[dict] = []
    y_result: list[str] = []
    y_home_goals: list[float] = []
    y_away_goals: list[float] = []

    for row in df.itertuples(index=False):
        home = str(row.team)
        away = str(row.opponent)
        home_code = team_to_code.get(home)
        away_code = team_to_code.get(away)
        if home_code is None or away_code is None:
            continue

        match_date = ps._to_naive_timestamp(row.date)
        feature_rows.append(
            ps._build_feature_row(
                ctx,
                home,
                away,
                match_date,
                int(row.hour),
                int(row.day),
                int(row.season),
                int(home_code),
                int(away_code),
            )
        )
        y_result.append(str(row.result))
        y_home_goals.append(float(row.gf))
        y_away_goals.append(float(row.ga))

    if not feature_rows:
        return _RowPack(
            X=pd.DataFrame(),
            y_result=pd.Series(dtype=str),
            y_home_goals=pd.Series(dtype=float),
            y_away_goals=pd.Series(dtype=float),
        )

    return _RowPack(
        X=pd.DataFrame(feature_rows),
        y_result=pd.Series(y_result),
        y_home_goals=pd.Series(y_home_goals),
        y_away_goals=pd.Series(y_away_goals),
    )


def _multiclass_brier_score(prob_rows: list[dict[str, float]], actual: list[str]) -> float:
    classes = ["W", "D", "L"]
    total = 0.0
    n = max(len(actual), 1)
    for probs, y in zip(prob_rows, actual):
        row_score = 0.0
        for c in classes:
            yk = 1.0 if y == c else 0.0
            pk = float(probs.get(c, 0.0))
            row_score += (pk - yk) ** 2
        total += row_score / len(classes)
    return total / n


def _confidence_bucket(value: float) -> str:
    if value < 45:
        return "<45%"
    if value < 55:
        return "45-55%"
    if value < 65:
        return "55-65%"
    return "65%+"


def build_model_backtest(all_matches: pd.DataFrame) -> dict:
    normalized = ps._normalize_matches_frame(all_matches)
    home_df = ps._prepare_home_training_frame(normalized)
    if len(home_df) < 180:
        return {
            "available": False,
            "reason": "insufficient_training_data",
            "sampleCount": int(len(home_df)),
        }

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
        return {
            "available": False,
            "reason": "insufficient_holdout_window",
            "sampleCount": int(len(training_df)),
        }

    ctx_df = normalized[normalized["season"] >= min_season].copy()
    ctx = ps._build_feature_context(ctx_df, form_window=ps.MODEL_FORM_WINDOW, h2h_window=ps.MODEL_H2H_WINDOW)
    # Backtest intentionally excludes live availability overrides to keep evaluation stable and historical.
    ctx.availability_index = {}
    ctx.team_news_explanations = {}

    teams = sorted(set(training_df["team"].astype(str)) | set(training_df["opponent"].astype(str)))
    team_to_code = {team: idx for idx, team in enumerate(teams)}

    train_pack = _build_row_pack(train_df, ctx, team_to_code)
    eval_pack = _build_row_pack(eval_df, ctx, team_to_code)
    if train_pack.X.empty or eval_pack.X.empty:
        return {
            "available": False,
            "reason": "failed_feature_build",
            "sampleCount": int(len(training_df)),
        }

    # Align eval columns with training columns.
    train_cols = list(train_pack.X.columns)
    eval_X = eval_pack.X.reindex(columns=train_cols)

    clf = ps._build_rf_model()
    clf.fit(train_pack.X, train_pack.y_result)
    reg_home = ps._build_goal_regressor()
    reg_away = ps._build_goal_regressor()
    reg_home.fit(train_pack.X, train_pack.y_home_goals)
    reg_away.fit(train_pack.X, train_pack.y_away_goals)

    pred_result = clf.predict(eval_X)
    pred_prob = clf.predict_proba(eval_X)
    classes = list(clf.classes_)

    prob_rows: list[dict[str, float]] = []
    confidence_rows: list[float] = []
    for row in pred_prob:
        mapping = {c: float(p) for c, p in zip(classes, row)}
        prob_rows.append(mapping)
        confidence_rows.append(max(mapping.values()) * 100.0 if mapping else 0.0)

    eval_true = eval_pack.y_result.astype(str).tolist()
    correct = sum(1 for y, p in zip(eval_true, pred_result) if str(y) == str(p))
    accuracy = correct / max(len(eval_true), 1)
    brier = _multiclass_brier_score(prob_rows, eval_true)

    pred_home_goals = reg_home.predict(eval_X)
    pred_away_goals = reg_away.predict(eval_X)
    home_mae = float((abs(eval_pack.y_home_goals - pred_home_goals)).mean())
    away_mae = float((abs(eval_pack.y_away_goals - pred_away_goals)).mean())

    rounded_home = [int(max(0, min(6, round(x)))) for x in pred_home_goals]
    rounded_away = [int(max(0, min(6, round(x)))) for x in pred_away_goals]
    true_home = [int(max(0, min(6, round(x)))) for x in eval_pack.y_home_goals.tolist()]
    true_away = [int(max(0, min(6, round(x)))) for x in eval_pack.y_away_goals.tolist()]
    exact_hits = sum(1 for ph, pa, th, ta in zip(rounded_home, rounded_away, true_home, true_away) if ph == th and pa == ta)
    exact_rate = exact_hits / max(len(true_home), 1)

    # Confidence calibration summary.
    buckets: dict[str, dict[str, float]] = {}
    for conf, y, p in zip(confidence_rows, eval_true, pred_result):
        key = _confidence_bucket(float(conf))
        row = buckets.setdefault(key, {"count": 0.0, "correct": 0.0})
        row["count"] += 1.0
        if str(y) == str(p):
            row["correct"] += 1.0

    calibration = []
    for key in ["<45%", "45-55%", "55-65%", "65%+"]:
        row = buckets.get(key, {"count": 0.0, "correct": 0.0})
        count = int(row["count"])
        acc = float(row["correct"] / row["count"]) if row["count"] > 0 else 0.0
        calibration.append({"bucket": key, "count": count, "accuracyPct": round(acc * 100, 1)})

    return {
        "available": True,
        "evaluationScope": f"temporal_holdout_latest_or_tail",
        "latestSeason": int(latest_season),
        "trainingSampleCount": int(len(train_pack.X)),
        "holdoutSampleCount": int(len(eval_pack.X)),
        "accuracyPct": round(float(accuracy * 100), 1),
        "brierScore": round(float(brier), 4),
        "homeGoalMae": round(home_mae, 3),
        "awayGoalMae": round(away_mae, 3),
        "scorelineExactPct": round(float(exact_rate * 100), 1),
        "confidenceCalibration": calibration,
    }
