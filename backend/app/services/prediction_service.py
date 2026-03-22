from __future__ import annotations

import hashlib
import json
import pickle
import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

from ..data.repository import load_team_availability_frame, load_team_news_frame
from ..config import (
    MODEL_FORM_WINDOW,
    MODEL_H2H_WINDOW,
    MODEL_ODDS_BLEND_WEIGHT,
    MODEL_RF_MAX_DEPTH,
    MODEL_RF_MIN_SAMPLES_SPLIT,
    MODEL_RF_TREES,
    MODEL_SCORE_DISPERSION,
    MODEL_ARTIFACT_ENABLED,
    MODEL_ARTIFACT_PATH,
    MODEL_TRAINING_YEARS,
)

TEAM_ALIASES = {
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "manchester utd": "Manchester United",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "man city": "Manchester City",
    "tottenham": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "afc bournemouth": "Bournemouth",
    "nottingham": "Nottingham Forest",
}

RESULT_POINTS = {"W": 3, "D": 1, "L": 0}
TEAM_NEWS_STATUS_WEIGHTS = {
    "injured": (1.0, 0.0, 1.0),
    "out": (1.0, 0.0, 1.0),
    "doubtful": (0.5, 0.0, 0.45),
    "suspended": (0.0, 1.0, 1.1),
    "banned": (0.0, 1.0, 1.1),
}
TEAM_NEWS_AVAILABLE_STATUSES = {"available", "fit"}


@dataclass
class FeatureContext:
    team_history: dict[str, list[dict]]
    team_home_history: dict[str, list[dict]]
    team_away_history: dict[str, list[dict]]
    h2h_index: dict[tuple[str, str], list[dict]]
    team_rating_history: dict[str, list[dict]]
    availability_index: dict[str, dict[str, float]]
    team_news_explanations: dict[str, list[dict]]
    form_window: int
    h2h_window: int
    default_rating: float
    default_ppg: float
    default_gf: float
    default_ga: float


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    home_goals_model: RandomForestRegressor
    away_goals_model: RandomForestRegressor
    probability_calibrators: dict[str, IsotonicRegression]
    over25_model: RandomForestClassifier | None
    over35_model: RandomForestClassifier | None
    btts_model: RandomForestClassifier | None
    over25_calibrator: IsotonicRegression | None
    over35_calibrator: IsotonicRegression | None
    btts_calibrator: IsotonicRegression | None
    draw_temperature: float
    team_to_code: dict[str, int]
    metrics: dict
    feature_columns: list[str]
    feature_context: FeatureContext
    training_latest_season: int


_MODEL_BUNDLE: ModelBundle | None = None
_MODEL_META: dict[str, object] = {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _availability_signature(availability_index: dict[str, dict[str, float]]) -> str:
    rows: list[dict[str, float | str]] = []
    for team in sorted(availability_index.keys()):
        row = availability_index.get(team, {})
        rows.append(
            {
                "team": str(team),
                "injured": round(float(row.get("injured", 0.0)), 6),
                "suspended": round(float(row.get("suspended", 0.0)), 6),
                "key_impact": round(float(row.get("key_impact", 0.0)), 6),
            }
        )
    return _sha256_text(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def _training_data_signature(training_df: pd.DataFrame) -> str:
    if training_df.empty:
        return "empty"
    sig_cols = ["date", "time", "team", "opponent", "season", "result", "gf", "ga", "xg", "xga", "sh", "sot"]
    frame = training_df.reindex(columns=sig_cols).fillna("")
    hash_values = pd.util.hash_pandas_object(frame.astype(str), index=False).values.tobytes()
    return hashlib.sha256(hash_values).hexdigest()


def _build_model_cache_key(
    training_df: pd.DataFrame,
    latest_season: int,
    min_season: int,
    availability_index: dict[str, dict[str, float]],
) -> str:
    key_payload = {
        "v": 2,
        "latestSeason": int(latest_season),
        "minSeason": int(min_season),
        "rows": int(len(training_df)),
        "dataSig": _training_data_signature(training_df),
        "availabilitySig": _availability_signature(availability_index),
        "cfg": {
            "years": int(max(1, MODEL_TRAINING_YEARS)),
            "formWindow": int(max(1, MODEL_FORM_WINDOW)),
            "h2hWindow": int(max(1, MODEL_H2H_WINDOW)),
            "rfTrees": int(max(50, MODEL_RF_TREES)),
            "rfMinSplit": int(max(2, MODEL_RF_MIN_SAMPLES_SPLIT)),
            "rfMaxDepth": int(MODEL_RF_MAX_DEPTH),
            "oddsBlendWeight": round(float(max(0.0, min(0.45, MODEL_ODDS_BLEND_WEIGHT))), 6),
            "scoreDispersion": round(float(max(0.05, min(0.5, MODEL_SCORE_DISPERSION))), 6),
        },
    }
    return _sha256_text(json.dumps(key_payload, sort_keys=True, separators=(",", ":")))


def _apply_model_source_metadata(bundle: ModelBundle, source: str, cache_key: str, artifact_saved_at: str | None = None) -> None:
    bundle.metrics["modelSource"] = str(source)
    bundle.metrics["modelCacheKey"] = str(cache_key)
    bundle.metrics["modelArtifactPath"] = str(MODEL_ARTIFACT_PATH)
    if artifact_saved_at:
        bundle.metrics["modelArtifactSavedAt"] = str(artifact_saved_at)


def _load_model_bundle_from_artifact(cache_key: str) -> ModelBundle | None:
    global _MODEL_META
    if not MODEL_ARTIFACT_ENABLED:
        return None
    path = MODEL_ARTIFACT_PATH
    if not path.exists():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("cacheKey", "")) != str(cache_key):
        return None
    bundle = payload.get("bundle")
    if not isinstance(bundle, ModelBundle):
        return None

    saved_at = str(payload.get("savedAt", ""))
    _apply_model_source_metadata(bundle, source="artifact", cache_key=cache_key, artifact_saved_at=saved_at or None)
    _MODEL_META = {
        "source": "artifact",
        "cacheKey": cache_key,
        "artifactPath": str(path),
        "savedAt": saved_at,
    }
    return bundle


def _save_model_bundle_to_artifact(cache_key: str, bundle: ModelBundle) -> None:
    global _MODEL_META
    if not MODEL_ARTIFACT_ENABLED:
        return
    path = MODEL_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "cacheKey": cache_key,
        "savedAt": saved_at,
        "bundle": bundle,
    }
    try:
        path.write_bytes(pickle.dumps(payload))
    except Exception:
        return
    _MODEL_META = {
        "source": "trained",
        "cacheKey": cache_key,
        "artifactPath": str(path),
        "savedAt": saved_at,
    }
    _apply_model_source_metadata(bundle, source="trained", cache_key=cache_key, artifact_saved_at=saved_at)


def _build_rf_model() -> RandomForestClassifier:
    max_depth = MODEL_RF_MAX_DEPTH if MODEL_RF_MAX_DEPTH > 0 else None
    return RandomForestClassifier(
        n_estimators=max(50, MODEL_RF_TREES),
        min_samples_split=max(2, MODEL_RF_MIN_SAMPLES_SPLIT),
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def _build_goal_regressor() -> RandomForestRegressor:
    max_depth = MODEL_RF_MAX_DEPTH if MODEL_RF_MAX_DEPTH > 0 else None
    return RandomForestRegressor(
        n_estimators=max(80, MODEL_RF_TREES),
        min_samples_split=max(2, MODEL_RF_MIN_SAMPLES_SPLIT),
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )


def _build_binary_market_model() -> RandomForestClassifier:
    max_depth = MODEL_RF_MAX_DEPTH if MODEL_RF_MAX_DEPTH > 0 else None
    return RandomForestClassifier(
        n_estimators=max(120, MODEL_RF_TREES),
        min_samples_split=max(2, MODEL_RF_MIN_SAMPLES_SPLIT),
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def _normalize_three_way(home: int, draw: int, away: int) -> tuple[int, int, int]:
    total = max(home + draw + away, 1)
    home_pct = round((home / total) * 100)
    draw_pct = round((draw / total) * 100)
    away_pct = 100 - home_pct - draw_pct
    return max(home_pct, 1), max(draw_pct, 1), max(away_pct, 1)


def _normalize_three_way_probs(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = max(float(home) + float(draw) + float(away), 1e-8)
    return (
        max(1e-6, float(home) / total),
        max(1e-6, float(draw) / total),
        max(1e-6, float(away) / total),
    )


def _canonical_team_name(raw_name: str) -> str:
    value = re.sub(r"\s+", " ", str(raw_name).strip())
    value = re.sub(r"\s+FC$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+AFC$", "", value, flags=re.IGNORECASE)
    key = value.lower()
    return TEAM_ALIASES.get(key, value)


def _to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.utcnow().tz_localize(None)
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert(None)
    return ts


def _normalize_matches_frame(all_matches: pd.DataFrame) -> pd.DataFrame:
    df = all_matches.copy()
    df["team"] = df["team"].astype(str).map(_canonical_team_name)
    df["opponent"] = df["opponent"].astype(str).map(_canonical_team_name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    df["gf"] = pd.to_numeric(df["gf"], errors="coerce").fillna(0)
    df["ga"] = pd.to_numeric(df["ga"], errors="coerce").fillna(0)
    for metric in ["xg", "xga", "sh", "sot", "odds_home", "odds_draw", "odds_away"]:
        if metric not in df.columns:
            df[metric] = 0.0
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
    if "referee" not in df.columns:
        df["referee"] = ""
    df["referee"] = df["referee"].astype(str).fillna("").str.strip()
    return df.dropna(subset=["date", "team", "opponent", "result"])


def _prepare_home_training_frame(all_matches: pd.DataFrame) -> pd.DataFrame:
    home_df = all_matches[all_matches["venue"].astype(str).str.lower() == "home"].copy()
    home_df = home_df[home_df["result"].isin(["W", "D", "L"])].copy()
    home_df["hour"] = (
        home_df["time"]
        .astype(str)
        .str.extract(r"^(\d{1,2})", expand=False)
        .fillna("15")
        .astype(int)
    )
    home_df["day"] = pd.to_datetime(home_df["date"], errors="coerce").dt.dayofweek.fillna(0).astype(int)
    return home_df.dropna(subset=["team", "opponent", "result"])


def _build_feature_context(matches_df: pd.DataFrame, form_window: int, h2h_window: int) -> FeatureContext:
    ordered = matches_df.sort_values(["date", "time"]).reset_index(drop=True)

    team_history: dict[str, list[dict]] = {}
    team_home_history: dict[str, list[dict]] = {}
    team_away_history: dict[str, list[dict]] = {}
    for row in ordered.itertuples(index=False):
        points = RESULT_POINTS.get(str(row.result), 0)
        entry = {
            "date": row.date,
            "points": points,
            "gf": float(row.gf),
            "ga": float(row.ga),
            "xg": float(getattr(row, "xg", 0.0)),
            "xga": float(getattr(row, "xga", 0.0)),
            "sh": float(getattr(row, "sh", 0.0)),
            "sot": float(getattr(row, "sot", 0.0)),
            "referee": str(getattr(row, "referee", "")).strip(),
            "odds_for": float(getattr(row, "odds_home", 0.0)),
            "odds_draw": float(getattr(row, "odds_draw", 0.0)),
            "odds_against": float(getattr(row, "odds_away", 0.0)),
        }
        team = str(row.team)
        venue = str(getattr(row, "venue", "")).strip().lower()
        team_history.setdefault(team, []).append(entry)
        if venue == "home":
            team_home_history.setdefault(team, []).append(entry)
        elif venue == "away":
            team_away_history.setdefault(team, []).append(entry)

    h2h_index: dict[tuple[str, str], list[dict]] = {}
    home_only = ordered[ordered["venue"].astype(str).str.lower() == "home"]
    for row in home_only.itertuples(index=False):
        key = tuple(sorted((row.team, row.opponent)))
        h2h_index.setdefault(key, []).append(
            {
                "date": row.date,
                "home": row.team,
                "away": row.opponent,
                "result": row.result,
            }
        )

    # Build pre-match Elo snapshot history for each team.
    team_rating_history: dict[str, list[dict]] = {}
    ratings: dict[str, float] = {}
    base_rating = 1500.0
    k_factor = 18.0
    home_adv = 55.0
    for row in home_only.sort_values(["date", "time"]).itertuples(index=False):
        home_team = str(row.team)
        away_team = str(row.opponent)
        home_rating = float(ratings.get(home_team, base_rating))
        away_rating = float(ratings.get(away_team, base_rating))
        team_rating_history.setdefault(home_team, []).append({"date": row.date, "rating": home_rating})
        team_rating_history.setdefault(away_team, []).append({"date": row.date, "rating": away_rating})

        expected_home = 1.0 / (1.0 + (10.0 ** (((away_rating - (home_rating + home_adv)) / 400.0))))
        result = str(row.result)
        if result == "W":
            actual_home = 1.0
        elif result == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0

        ratings[home_team] = home_rating + (k_factor * (actual_home - expected_home))
        ratings[away_team] = away_rating + (k_factor * ((1.0 - actual_home) - (1.0 - expected_home)))

    default_ppg = float(ordered["result"].map(RESULT_POINTS).fillna(0).mean())
    default_gf = float(ordered["gf"].mean()) if not ordered.empty else 1.3
    default_ga = float(ordered["ga"].mean()) if not ordered.empty else 1.3

    return FeatureContext(
        team_history=team_history,
        team_home_history=team_home_history,
        team_away_history=team_away_history,
        h2h_index=h2h_index,
        team_rating_history=team_rating_history,
        availability_index={},
        team_news_explanations={},
        form_window=max(1, form_window),
        h2h_window=max(1, h2h_window),
        default_rating=base_rating,
        default_ppg=default_ppg,
        default_gf=default_gf,
        default_ga=default_ga,
    )


def _build_manual_availability_index() -> dict[str, dict[str, float]]:
    frame = load_team_availability_frame()
    if frame is None:
        return {}

    index: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        team = _canonical_team_name(str(row.team))
        index[team] = {
            "injured": max(0.0, float(row.injured)),
            "suspended": max(0.0, float(row.suspended)),
            "key_impact": max(0.0, min(1.0, float(row.key_impact))),
        }
    return index


def _build_team_news_availability_index() -> tuple[dict[str, dict[str, float]], dict[str, list[dict]]]:
    frame = load_team_news_frame()
    if frame is None:
        return {}, {}

    def _player_strength(row) -> float:
        importance = max(0.0, min(1.0, float(getattr(row, "importance", 0.0))))
        minutes = float(getattr(row, "minutes", 0.0))
        starts = float(getattr(row, "starts", 0.0))
        influence = float(getattr(row, "influence", 0.0))
        creativity = float(getattr(row, "creativity", 0.0))
        threat = float(getattr(row, "threat", 0.0))
        expected_goals = float(getattr(row, "expected_goals", 0.0))
        expected_assists = float(getattr(row, "expected_assists", 0.0))

        score = (
            (importance * 0.9)
            + (min(1.0, minutes / 1800.0) * 0.26)
            + (min(1.0, starts / 20.0) * 0.18)
            + (min(1.0, influence / 120.0) * 0.2)
            + (min(1.0, creativity / 120.0) * 0.14)
            + (min(1.0, threat / 120.0) * 0.18)
            + (min(1.0, expected_goals / 4.0) * 0.1)
            + (min(1.0, expected_assists / 3.0) * 0.08)
        )
        return max(0.12, min(2.2, score))

    normalized = frame.copy()
    normalized["team"] = normalized["team"].astype(str).map(_canonical_team_name)
    normalized["status"] = normalized["status"].astype(str).str.strip().str.lower()
    normalized["position"] = normalized["position"].astype(str).str.strip().str.upper()

    team_total_strength: dict[str, float] = {}
    available_by_position: dict[tuple[str, str], float] = {}
    for row in normalized.itertuples(index=False):
        team = str(row.team)
        position = str(row.position)
        status = str(row.status)
        strength = _player_strength(row)
        team_total_strength[team] = team_total_strength.get(team, 0.0) + strength
        if status in TEAM_NEWS_AVAILABLE_STATUSES:
            key = (team, position)
            available_by_position[key] = available_by_position.get(key, 0.0) + strength

    rolled_up: dict[str, dict[str, float]] = {}
    explanations: dict[str, list[dict]] = {}
    per_team_contributions: dict[str, list[dict]] = {}
    for row in normalized.itertuples(index=False):
        team = str(row.team)
        status = str(row.status).strip().lower()
        if status not in TEAM_NEWS_STATUS_WEIGHTS:
            continue

        injured_units, suspended_units, impact_multiplier = TEAM_NEWS_STATUS_WEIGHTS[status]
        importance = max(0.0, min(1.0, float(row.importance)))
        position = str(row.position).strip().upper()
        strength = _player_strength(row)
        total_strength = max(1.0, team_total_strength.get(team, 0.0))
        dependency_share = min(1.0, (strength / total_strength) * 6.0)
        replacement_strength = float(available_by_position.get((team, position), 0.0))
        replacement_ratio = replacement_strength / max(replacement_strength + strength, 0.01)
        depth_adjustment = 1.0 - (0.5 * replacement_ratio)

        count_weight = 0.7 + (0.7 * importance) + (0.3 * dependency_share)
        impact_weight = (0.45 * importance) + (0.35 * dependency_share) + (0.2 * min(1.0, strength / 1.5))
        impact_weight *= depth_adjustment
        impact_contribution = impact_weight * impact_multiplier

        per_team_contributions.setdefault(team, []).append(
            {
                "player": str(getattr(row, "player", "")),
                "status": status,
                "position": position,
                "importance": float(importance),
                "dependencyShare": float(dependency_share),
                "depthAdjustment": float(depth_adjustment),
                "impactContribution": float(impact_contribution),
                "injuredCount": float(injured_units * count_weight),
                "suspendedCount": float(suspended_units * count_weight),
            }
        )

    for team, items in per_team_contributions.items():
        items.sort(key=lambda item: float(item.get("impactContribution", 0.0)), reverse=True)
        focused = items[:6]
        rolled_up[team] = {
            "injured": sum(float(item.get("injuredCount", 0.0)) for item in focused),
            "suspended": sum(float(item.get("suspendedCount", 0.0)) for item in focused),
            "impact_raw": sum(float(item.get("impactContribution", 0.0)) for item in focused),
        }
        explanations[team] = [
            {
                "player": str(item.get("player", "")),
                "status": str(item.get("status", "")),
                "position": str(item.get("position", "")),
                "importance": round(float(item.get("importance", 0.0)), 3),
                "dependencyShare": round(float(item.get("dependencyShare", 0.0)), 3),
                "depthAdjustment": round(float(item.get("depthAdjustment", 0.0)), 3),
                "impactContribution": round(float(item.get("impactContribution", 0.0)), 3),
            }
            for item in focused[:8]
        ]

    index: dict[str, dict[str, float]] = {}
    for team, row in rolled_up.items():
        key_impact = 1.0 - math.exp(-float(row["impact_raw"]))
        index[team] = {
            "injured": float(row["injured"]),
            "suspended": float(row["suspended"]),
            "key_impact": max(0.0, min(1.0, key_impact)),
        }

    return index, explanations


def _build_availability_index() -> tuple[dict[str, dict[str, float]], dict[str, list[dict]]]:
    manual = _build_manual_availability_index()
    team_news, explanations = _build_team_news_availability_index()
    if not team_news:
        return manual, explanations

    merged = {team: values.copy() for team, values in manual.items()}
    for team, news_row in team_news.items():
        existing = merged.setdefault(team, {"injured": 0.0, "suspended": 0.0, "key_impact": 0.0})
        existing["injured"] = max(float(existing.get("injured", 0.0)), float(news_row.get("injured", 0.0)))
        existing["suspended"] = max(float(existing.get("suspended", 0.0)), float(news_row.get("suspended", 0.0)))
        manual_impact = max(0.0, min(1.0, float(existing.get("key_impact", 0.0))))
        news_impact = max(0.0, min(1.0, float(news_row.get("key_impact", 0.0))))
        existing["key_impact"] = 1.0 - ((1.0 - manual_impact) * (1.0 - news_impact))

    return merged, explanations


def _recent_team_form(context: FeatureContext, team: str, match_date: pd.Timestamp) -> dict[str, float]:
    history = context.team_history.get(team, [])
    if not history:
        return {
            "ppg": context.default_ppg,
            "gf": context.default_gf,
            "ga": context.default_ga,
            "xg": context.default_gf,
            "xga": context.default_ga,
            "sh": 12.0,
            "sot": 4.0,
            "odds_for": 2.6,
            "odds_draw": 3.4,
            "odds_against": 2.9,
        }

    samples: list[dict] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue
        samples.append(item)
        if len(samples) >= context.form_window:
            break

    if not samples:
        return {
            "ppg": context.default_ppg,
            "gf": context.default_gf,
            "ga": context.default_ga,
            "xg": context.default_gf,
            "xga": context.default_ga,
            "sh": 12.0,
            "sot": 4.0,
            "odds_for": 2.6,
            "odds_draw": 3.4,
            "odds_against": 2.9,
        }

    return {
        "ppg": float(sum(x["points"] for x in samples) / len(samples)),
        "gf": float(sum(x["gf"] for x in samples) / len(samples)),
        "ga": float(sum(x["ga"] for x in samples) / len(samples)),
        "xg": float(sum(x.get("xg", x["gf"]) for x in samples) / len(samples)),
        "xga": float(sum(x.get("xga", x["ga"]) for x in samples) / len(samples)),
        "sh": float(sum(x.get("sh", 12.0) for x in samples) / len(samples)),
        "sot": float(sum(x.get("sot", 4.0) for x in samples) / len(samples)),
        "odds_for": float(sum(x.get("odds_for", 2.6) for x in samples) / len(samples)),
        "odds_draw": float(sum(x.get("odds_draw", 3.4) for x in samples) / len(samples)),
        "odds_against": float(sum(x.get("odds_against", 2.9) for x in samples) / len(samples)),
    }


def _rolling_team_trends(
    context: FeatureContext,
    team: str,
    match_date: pd.Timestamp,
    windows: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    history = context.team_history.get(team, [])
    if not history:
        output: dict[str, float] = {}
        for window in windows:
            output[f"ppg_{window}"] = float(context.default_ppg)
            output[f"gf_{window}"] = float(context.default_gf)
            output[f"ga_{window}"] = float(context.default_ga)
            output[f"xg_{window}"] = float(context.default_gf)
            output[f"xga_{window}"] = float(context.default_ga)
            output[f"sh_{window}"] = 12.0
            output[f"sot_{window}"] = 4.0
            output[f"odds_for_{window}"] = 2.6
            output[f"odds_against_{window}"] = 2.9
        return output

    prior_matches: list[dict] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue
        prior_matches.append(item)
        if len(prior_matches) >= max(windows):
            break

    output = {}
    for window in windows:
        sample = prior_matches[:window]
        if not sample:
            output[f"ppg_{window}"] = float(context.default_ppg)
            output[f"gf_{window}"] = float(context.default_gf)
            output[f"ga_{window}"] = float(context.default_ga)
            output[f"xg_{window}"] = float(context.default_gf)
            output[f"xga_{window}"] = float(context.default_ga)
            output[f"sh_{window}"] = 12.0
            output[f"sot_{window}"] = 4.0
            output[f"odds_for_{window}"] = 2.6
            output[f"odds_against_{window}"] = 2.9
            continue
        output[f"ppg_{window}"] = float(sum(x["points"] for x in sample) / len(sample))
        output[f"gf_{window}"] = float(sum(x["gf"] for x in sample) / len(sample))
        output[f"ga_{window}"] = float(sum(x["ga"] for x in sample) / len(sample))
        output[f"xg_{window}"] = float(sum(x.get("xg", x["gf"]) for x in sample) / len(sample))
        output[f"xga_{window}"] = float(sum(x.get("xga", x["ga"]) for x in sample) / len(sample))
        output[f"sh_{window}"] = float(sum(x.get("sh", 12.0) for x in sample) / len(sample))
        output[f"sot_{window}"] = float(sum(x.get("sot", 4.0) for x in sample) / len(sample))
        output[f"odds_for_{window}"] = float(sum(x.get("odds_for", 2.6) for x in sample) / len(sample))
        output[f"odds_against_{window}"] = float(sum(x.get("odds_against", 2.9) for x in sample) / len(sample))

    return output


def _rolling_venue_trends(
    context: FeatureContext,
    team: str,
    match_date: pd.Timestamp,
    venue: str,
    windows: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    venue_key = str(venue).strip().lower()
    if venue_key == "home":
        history = context.team_home_history.get(team, [])
    elif venue_key == "away":
        history = context.team_away_history.get(team, [])
    else:
        history = []
    if not history:
        output: dict[str, float] = {}
        for window in windows:
            output[f"ppg_{window}"] = float(context.default_ppg)
            output[f"gf_{window}"] = float(context.default_gf)
            output[f"ga_{window}"] = float(context.default_ga)
            output[f"xg_{window}"] = float(context.default_gf)
            output[f"xga_{window}"] = float(context.default_ga)
            output[f"sh_{window}"] = 12.0
            output[f"sot_{window}"] = 4.0
        return output

    prior_matches: list[dict] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue
        prior_matches.append(item)
        if len(prior_matches) >= max(windows):
            break

    output = {}
    for window in windows:
        sample = prior_matches[:window]
        if not sample:
            output[f"ppg_{window}"] = float(context.default_ppg)
            output[f"gf_{window}"] = float(context.default_gf)
            output[f"ga_{window}"] = float(context.default_ga)
            output[f"xg_{window}"] = float(context.default_gf)
            output[f"xga_{window}"] = float(context.default_ga)
            output[f"sh_{window}"] = 12.0
            output[f"sot_{window}"] = 4.0
            continue
        output[f"ppg_{window}"] = float(sum(x["points"] for x in sample) / len(sample))
        output[f"gf_{window}"] = float(sum(x["gf"] for x in sample) / len(sample))
        output[f"ga_{window}"] = float(sum(x["ga"] for x in sample) / len(sample))
        output[f"xg_{window}"] = float(sum(x.get("xg", x["gf"]) for x in sample) / len(sample))
        output[f"xga_{window}"] = float(sum(x.get("xga", x["ga"]) for x in sample) / len(sample))
        output[f"sh_{window}"] = float(sum(x.get("sh", 12.0) for x in sample) / len(sample))
        output[f"sot_{window}"] = float(sum(x.get("sot", 4.0) for x in sample) / len(sample))

    return output


def _recent_h2h(context: FeatureContext, home: str, away: str, match_date: pd.Timestamp) -> dict[str, float]:
    key = tuple(sorted((home, away)))
    history = context.h2h_index.get(key, [])

    if not history:
        return {
            "home_win_rate": 0.4,
            "draw_rate": 0.24,
            "away_win_rate": 0.36,
            "sample": 0,
        }

    outcomes: list[str] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue

        result = str(item["result"])
        if item["home"] == home:
            perspective = result
        else:
            perspective = "L" if result == "W" else "W" if result == "L" else "D"

        outcomes.append(perspective)
        if len(outcomes) >= context.h2h_window:
            break

    if not outcomes:
        return {
            "home_win_rate": 0.4,
            "draw_rate": 0.24,
            "away_win_rate": 0.36,
            "sample": 0,
        }

    sample = len(outcomes)
    home_wins = sum(1 for x in outcomes if x == "W") / sample
    draws = sum(1 for x in outcomes if x == "D") / sample
    away_wins = sum(1 for x in outcomes if x == "L") / sample

    return {
        "home_win_rate": float(home_wins),
        "draw_rate": float(draws),
        "away_win_rate": float(away_wins),
        "sample": sample,
    }


def _pre_match_rating(context: FeatureContext, team: str, match_date: pd.Timestamp) -> float:
    history = context.team_rating_history.get(team, [])
    if not history:
        return float(context.default_rating)

    for item in reversed(history):
        if item["date"] < match_date:
            return float(item["rating"])
    return float(context.default_rating)


def _rest_and_congestion(context: FeatureContext, team: str, match_date: pd.Timestamp) -> dict[str, float]:
    history = context.team_history.get(team, [])
    if not history:
        return {
            "rest_days": 6.0,
            "matches_last_7d": 0.0,
            "matches_last_14d": 0.0,
        }

    last_match_date: pd.Timestamp | None = None
    matches_7d = 0
    matches_14d = 0
    for item in reversed(history):
        item_date = item["date"]
        if item_date >= match_date:
            continue

        delta_days = int((match_date - item_date).days)
        if delta_days <= 14:
            matches_14d += 1
            if delta_days <= 7:
                matches_7d += 1
        if last_match_date is None:
            last_match_date = item_date
        if delta_days > 14 and last_match_date is not None:
            break

    if last_match_date is None:
        rest_days = 6.0
    else:
        rest_days = float(max(1, min(14, int((match_date - last_match_date).days))))

    return {
        "rest_days": rest_days,
        "matches_last_7d": float(max(0, matches_7d)),
        "matches_last_14d": float(max(0, matches_14d)),
    }


def _team_referee_points_rate(context: FeatureContext, team: str, referee: str, match_date: pd.Timestamp) -> float:
    referee_name = str(referee).strip().lower()
    if not referee_name:
        return float(context.default_ppg)

    history = context.team_history.get(team, [])
    if not history:
        return float(context.default_ppg)

    samples: list[float] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue
        if str(item.get("referee", "")).strip().lower() != referee_name:
            continue
        samples.append(float(item.get("points", 0.0)))
        if len(samples) >= 8:
            break

    if not samples:
        return float(context.default_ppg)
    return float(sum(samples) / len(samples))


def _build_feature_row(
    context: FeatureContext,
    home: str,
    away: str,
    match_date: pd.Timestamp,
    hour: int,
    day: int,
    season: int,
    home_code: int,
    away_code: int,
) -> dict:
    home_form = _recent_team_form(context, home, match_date)
    away_form = _recent_team_form(context, away, match_date)
    h2h = _recent_h2h(context, home, away, match_date)
    home_rating = _pre_match_rating(context, home, match_date)
    away_rating = _pre_match_rating(context, away, match_date)
    home_workload = _rest_and_congestion(context, home, match_date)
    away_workload = _rest_and_congestion(context, away, match_date)
    home_trends = _rolling_team_trends(context, home, match_date, windows=(3, 5, 10))
    away_trends = _rolling_team_trends(context, away, match_date, windows=(3, 5, 10))
    home_venue_trends = _rolling_venue_trends(context, home, match_date, venue="home", windows=(3, 5, 10))
    away_venue_trends = _rolling_venue_trends(context, away, match_date, venue="away", windows=(3, 5, 10))
    referee_name = ""
    home_history = context.team_history.get(home, [])
    for item in reversed(home_history):
        if item["date"] >= match_date:
            continue
        referee_name = str(item.get("referee", "")).strip()
        if referee_name:
            break
    home_ref_ppg = _team_referee_points_rate(context, home, referee_name, match_date)
    away_ref_ppg = _team_referee_points_rate(context, away, referee_name, match_date)

    interaction_attack_defense = float(home_form["gf"] * away_form["ga"])
    interaction_away_attack_defense = float(away_form["gf"] * home_form["ga"])
    interaction_xg_edge = float((home_form["xg"] - away_form["xg"]) - (home_form["xga"] - away_form["xga"]))
    interaction_shot_quality = float(
        (home_form["sot"] / max(home_form["sh"], 1.0)) - (away_form["sot"] / max(away_form["sh"], 1.0))
    )
    interaction_market_edge = float((1.0 / max(home_form["odds_for"], 1.01)) - (1.0 / max(away_form["odds_for"], 1.01)))
    interaction_form_elo = float((home_form["ppg"] - away_form["ppg"]) * ((home_rating - away_rating) / 100.0))
    interaction_congestion_edge = float(
        (home_workload["matches_last_7d"] - away_workload["matches_last_7d"]) * (home_form["ppg"] - away_form["ppg"])
    )
    interaction_h2h_form = float((h2h["home_win_rate"] - h2h["away_win_rate"]) * (home_form["ppg"] - away_form["ppg"]))

    return {
        "home_code": home_code,
        "away_code": away_code,
        "hour": int(hour),
        "day": int(day),
        "season": int(season),
        "home_form_ppg": home_form["ppg"],
        "away_form_ppg": away_form["ppg"],
        "form_ppg_diff": home_form["ppg"] - away_form["ppg"],
        "home_form_gf": home_form["gf"],
        "away_form_gf": away_form["gf"],
        "home_form_ga": home_form["ga"],
        "away_form_ga": away_form["ga"],
        "home_form_xg": home_form["xg"],
        "away_form_xg": away_form["xg"],
        "home_form_xga": home_form["xga"],
        "away_form_xga": away_form["xga"],
        "form_xg_diff": float(home_form["xg"] - away_form["xg"]),
        "form_xga_diff": float(home_form["xga"] - away_form["xga"]),
        "home_form_sh": home_form["sh"],
        "away_form_sh": away_form["sh"],
        "home_form_sot": home_form["sot"],
        "away_form_sot": away_form["sot"],
        "form_sh_diff": float(home_form["sh"] - away_form["sh"]),
        "form_sot_diff": float(home_form["sot"] - away_form["sot"]),
        "home_form_odds_for": home_form["odds_for"],
        "away_form_odds_for": away_form["odds_for"],
        "form_odds_for_diff": float(home_form["odds_for"] - away_form["odds_for"]),
        "h2h_home_win_rate": h2h["home_win_rate"],
        "h2h_draw_rate": h2h["draw_rate"],
        "h2h_away_win_rate": h2h["away_win_rate"],
        "h2h_sample": float(h2h["sample"]),
        "home_elo": float(home_rating),
        "away_elo": float(away_rating),
        "elo_diff": float(home_rating - away_rating),
        "home_referee_ppg": float(home_ref_ppg),
        "away_referee_ppg": float(away_ref_ppg),
        "referee_ppg_diff": float(home_ref_ppg - away_ref_ppg),
        "home_rest_days": float(home_workload["rest_days"]),
        "away_rest_days": float(away_workload["rest_days"]),
        "rest_days_diff": float(home_workload["rest_days"] - away_workload["rest_days"]),
        "home_matches_7d": float(home_workload["matches_last_7d"]),
        "away_matches_7d": float(away_workload["matches_last_7d"]),
        "congestion_7d_diff": float(home_workload["matches_last_7d"] - away_workload["matches_last_7d"]),
        "home_matches_14d": float(home_workload["matches_last_14d"]),
        "away_matches_14d": float(away_workload["matches_last_14d"]),
        "congestion_14d_diff": float(home_workload["matches_last_14d"] - away_workload["matches_last_14d"]),
        "home_ppg_3": float(home_trends["ppg_3"]),
        "away_ppg_3": float(away_trends["ppg_3"]),
        "ppg_3_diff": float(home_trends["ppg_3"] - away_trends["ppg_3"]),
        "home_ppg_5": float(home_trends["ppg_5"]),
        "away_ppg_5": float(away_trends["ppg_5"]),
        "ppg_5_diff": float(home_trends["ppg_5"] - away_trends["ppg_5"]),
        "home_ppg_10": float(home_trends["ppg_10"]),
        "away_ppg_10": float(away_trends["ppg_10"]),
        "ppg_10_diff": float(home_trends["ppg_10"] - away_trends["ppg_10"]),
        "home_gf_3": float(home_trends["gf_3"]),
        "away_gf_3": float(away_trends["gf_3"]),
        "gf_3_diff": float(home_trends["gf_3"] - away_trends["gf_3"]),
        "home_gf_5": float(home_trends["gf_5"]),
        "away_gf_5": float(away_trends["gf_5"]),
        "gf_5_diff": float(home_trends["gf_5"] - away_trends["gf_5"]),
        "home_xg_5": float(home_trends["xg_5"]),
        "away_xg_5": float(away_trends["xg_5"]),
        "xg_5_diff": float(home_trends["xg_5"] - away_trends["xg_5"]),
        "home_xga_5": float(home_trends["xga_5"]),
        "away_xga_5": float(away_trends["xga_5"]),
        "xga_5_diff": float(home_trends["xga_5"] - away_trends["xga_5"]),
        "home_sh_5": float(home_trends["sh_5"]),
        "away_sh_5": float(away_trends["sh_5"]),
        "sh_5_diff": float(home_trends["sh_5"] - away_trends["sh_5"]),
        "home_sot_5": float(home_trends["sot_5"]),
        "away_sot_5": float(away_trends["sot_5"]),
        "sot_5_diff": float(home_trends["sot_5"] - away_trends["sot_5"]),
        "home_market_odds_for_5": float(home_trends["odds_for_5"]),
        "away_market_odds_for_5": float(away_trends["odds_for_5"]),
        "market_odds_for_5_diff": float(home_trends["odds_for_5"] - away_trends["odds_for_5"]),
        "home_gf_10": float(home_trends["gf_10"]),
        "away_gf_10": float(away_trends["gf_10"]),
        "gf_10_diff": float(home_trends["gf_10"] - away_trends["gf_10"]),
        "home_ga_3": float(home_trends["ga_3"]),
        "away_ga_3": float(away_trends["ga_3"]),
        "ga_3_diff": float(home_trends["ga_3"] - away_trends["ga_3"]),
        "home_ga_5": float(home_trends["ga_5"]),
        "away_ga_5": float(away_trends["ga_5"]),
        "ga_5_diff": float(home_trends["ga_5"] - away_trends["ga_5"]),
        "home_ga_10": float(home_trends["ga_10"]),
        "away_ga_10": float(away_trends["ga_10"]),
        "ga_10_diff": float(home_trends["ga_10"] - away_trends["ga_10"]),
        "home_home_ppg_3": float(home_venue_trends["ppg_3"]),
        "away_away_ppg_3": float(away_venue_trends["ppg_3"]),
        "venue_ppg_3_diff": float(home_venue_trends["ppg_3"] - away_venue_trends["ppg_3"]),
        "home_home_ppg_5": float(home_venue_trends["ppg_5"]),
        "away_away_ppg_5": float(away_venue_trends["ppg_5"]),
        "venue_ppg_5_diff": float(home_venue_trends["ppg_5"] - away_venue_trends["ppg_5"]),
        "home_home_ppg_10": float(home_venue_trends["ppg_10"]),
        "away_away_ppg_10": float(away_venue_trends["ppg_10"]),
        "venue_ppg_10_diff": float(home_venue_trends["ppg_10"] - away_venue_trends["ppg_10"]),
        "home_home_gf_5": float(home_venue_trends["gf_5"]),
        "away_away_gf_5": float(away_venue_trends["gf_5"]),
        "venue_gf_5_diff": float(home_venue_trends["gf_5"] - away_venue_trends["gf_5"]),
        "home_home_xg_5": float(home_venue_trends["xg_5"]),
        "away_away_xg_5": float(away_venue_trends["xg_5"]),
        "venue_xg_5_diff": float(home_venue_trends["xg_5"] - away_venue_trends["xg_5"]),
        "home_home_ga_5": float(home_venue_trends["ga_5"]),
        "away_away_ga_5": float(away_venue_trends["ga_5"]),
        "venue_ga_5_diff": float(home_venue_trends["ga_5"] - away_venue_trends["ga_5"]),
        "interaction_attack_defense": interaction_attack_defense,
        "interaction_away_attack_defense": interaction_away_attack_defense,
        "interaction_xg_edge": interaction_xg_edge,
        "interaction_shot_quality": interaction_shot_quality,
        "interaction_market_edge": interaction_market_edge,
        "interaction_form_elo": interaction_form_elo,
        "interaction_congestion_edge": interaction_congestion_edge,
        "interaction_h2h_form": interaction_h2h_form,
    }


def _windowed_training_frame(home_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    latest_season = int(home_df["season"].max())
    years = max(1, MODEL_TRAINING_YEARS)
    min_season = latest_season - years + 1
    filtered = home_df[home_df["season"] >= min_season].copy()

    if len(filtered) < 120:
        filtered = home_df.copy()
        min_season = int(filtered["season"].min())

    return filtered, latest_season, min_season


def _evaluate_accuracy(features: pd.DataFrame, target: pd.Series, training_df: pd.DataFrame, latest_season: int) -> tuple[float, str]:
    holdout_mask = training_df["season"] == latest_season
    if int(holdout_mask.sum()) > 0 and int((~holdout_mask).sum()) > 0:
        holdout_features = features[holdout_mask.values]
        holdout_target = target[holdout_mask.values]
        base_features = features[(~holdout_mask).values]
        base_target = target[(~holdout_mask).values]

        eval_model = _build_rf_model()
        eval_model.fit(base_features, base_target)
        holdout_preds = eval_model.predict(holdout_features)
        return float(accuracy_score(holdout_target, holdout_preds)), f"season_{latest_season}_holdout"

    # Strict fallback: use chronological holdout from the end of the training window.
    sample_count = len(features)
    if sample_count >= 60:
        test_size = max(20, int(round(sample_count * 0.2)))
        test_size = min(test_size, sample_count - 30)
        split_idx = sample_count - test_size
        if split_idx >= 30 and test_size >= 10:
            train_features = features.iloc[:split_idx]
            train_target = target.iloc[:split_idx]
            test_features = features.iloc[split_idx:]
            test_target = target.iloc[split_idx:]

            eval_model = _build_rf_model()
            eval_model.fit(train_features, train_target)
            preds = eval_model.predict(test_features)
            return float(accuracy_score(test_target, preds)), "time_split_holdout"

    # Last resort only when there is not enough data to form a meaningful holdout.
    fallback_model = _build_rf_model()
    fallback_model.fit(features, target)
    train_preds = fallback_model.predict(features)
    return float(accuracy_score(target, train_preds)), "insufficient_holdout_training_set"


def _build_probability_calibrators(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
) -> dict[str, IsotonicRegression]:
    split = _calibration_split(features, target, training_df, latest_season)
    if split is None:
        return {}
    calib_train_X, calib_train_y, calib_eval_X, calib_eval_y = split

    calib_model = _build_rf_model()
    calib_model.fit(calib_train_X, calib_train_y)
    eval_probs = calib_model.predict_proba(calib_eval_X)
    classes = [str(c) for c in calib_model.classes_]
    y_eval = calib_eval_y.astype(str).tolist()

    calibrators: dict[str, IsotonicRegression] = {}
    for class_idx, class_name in enumerate(classes):
        class_scores = [float(row[class_idx]) for row in eval_probs]
        y_binary = [1.0 if y == class_name else 0.0 for y in y_eval]
        positives = sum(y_binary)
        negatives = len(y_binary) - positives
        if positives < 12 or negatives < 12:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(class_scores, y_binary)
        calibrators[class_name] = iso

    return calibrators


def _calibration_split(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    holdout_mask = training_df["season"] == latest_season
    if int(holdout_mask.sum()) > 30 and int((~holdout_mask).sum()) > 60:
        return (
            features[(~holdout_mask).values],
            target[(~holdout_mask).values],
            features[holdout_mask.values],
            target[holdout_mask.values],
        )

    sample_count = len(features)
    test_size = max(30, int(round(sample_count * 0.2)))
    test_size = min(test_size, max(0, sample_count - 80))
    split_idx = sample_count - test_size
    if split_idx < 80 or test_size < 30:
        return None
    return (
        features.iloc[:split_idx],
        target.iloc[:split_idx],
        features.iloc[split_idx:],
        target.iloc[split_idx:],
    )


def _build_binary_calibrator(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
) -> IsotonicRegression | None:
    split = _calibration_split(features, target, training_df, latest_season)
    if split is None:
        return None
    calib_train_X, calib_train_y, calib_eval_X, calib_eval_y = split

    if int(pd.Series(calib_train_y).nunique()) < 2 or int(pd.Series(calib_eval_y).nunique()) < 2:
        return None

    calib_model = _build_binary_market_model()
    calib_model.fit(calib_train_X, calib_train_y)
    probs = calib_model.predict_proba(calib_eval_X)
    class_list = [int(c) for c in calib_model.classes_]
    if 1 not in class_list:
        return None
    one_idx = class_list.index(1)
    positive_scores = [float(row[one_idx]) for row in probs]
    y_eval = [float(int(v)) for v in calib_eval_y.tolist()]
    positives = sum(y_eval)
    negatives = len(y_eval) - positives
    if positives < 12 or negatives < 12:
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(positive_scores, y_eval)
    return iso


def _fit_binary_market_with_calibration(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
) -> tuple[RandomForestClassifier | None, IsotonicRegression | None]:
    y = pd.Series(target).astype(int)
    if int(y.nunique()) < 2:
        return None, None
    model = _build_binary_market_model()
    model.fit(features, y)
    calibrator = _build_binary_calibrator(features, y, training_df, latest_season)
    return model, calibrator


def _calibrate_class_probabilities(
    class_probs: dict[str, float],
    calibrators: dict[str, IsotonicRegression],
) -> dict[str, float]:
    if not class_probs:
        return class_probs

    calibrated: dict[str, float] = {}
    for class_name, raw_prob in class_probs.items():
        value = float(raw_prob)
        calibrator = calibrators.get(str(class_name))
        if calibrator is not None:
            value = float(calibrator.predict([value])[0])
        calibrated[str(class_name)] = max(1e-6, min(1.0, value))

    total = sum(calibrated.values())
    if total <= 0:
        return {k: 1.0 / max(len(calibrated), 1) for k in calibrated}
    return {k: float(v / total) for k, v in calibrated.items()}


def _apply_draw_temperature(class_probs: dict[str, float], draw_temperature: float) -> dict[str, float]:
    if not class_probs:
        return class_probs
    adjusted = {str(k): float(v) for k, v in class_probs.items()}
    if "D" in adjusted:
        adjusted["D"] = max(1e-6, float(adjusted["D"]) * float(draw_temperature))
    total = sum(adjusted.values())
    if total <= 0:
        return {k: 1.0 / max(len(adjusted), 1) for k in adjusted}
    return {k: float(v / total) for k, v in adjusted.items()}


def _multiclass_brier_from_probs(prob_rows: list[dict[str, float]], actual: list[str]) -> float:
    classes = ["W", "D", "L"]
    total = 0.0
    n = max(1, len(actual))
    for probs, y in zip(prob_rows, actual):
        row_score = 0.0
        for c in classes:
            yk = 1.0 if str(y) == c else 0.0
            pk = float(probs.get(c, 0.0))
            row_score += (pk - yk) ** 2
        total += row_score / len(classes)
    return float(total / n)


def _fit_draw_temperature(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
    calibrators: dict[str, IsotonicRegression],
) -> float:
    split = _calibration_split(features, target, training_df, latest_season)
    if split is None:
        return 1.0
    train_X, train_y, eval_X, eval_y = split
    if len(eval_X) < 30:
        return 1.0

    model = _build_rf_model()
    model.fit(train_X, train_y)
    raw_probs = model.predict_proba(eval_X)
    classes = [str(c) for c in model.classes_]
    actual = eval_y.astype(str).tolist()

    base_rows: list[dict[str, float]] = []
    for row in raw_probs:
        class_probs_raw = {cls: float(prob) for cls, prob in zip(classes, row)}
        class_probs = _calibrate_class_probabilities(class_probs_raw, calibrators)
        base_rows.append(class_probs)

    best_temp = 1.0
    best_brier = _multiclass_brier_from_probs(base_rows, actual)
    for step in range(70, 131, 2):
        temp = float(step) / 100.0
        adjusted_rows = [_apply_draw_temperature(probs, temp) for probs in base_rows]
        brier = _multiclass_brier_from_probs(adjusted_rows, actual)
        if brier < best_brier:
            best_brier = brier
            best_temp = temp
    return float(best_temp)


def _classwise_calibration_summary(
    features: pd.DataFrame,
    target: pd.Series,
    training_df: pd.DataFrame,
    latest_season: int,
    calibrators: dict[str, IsotonicRegression],
    draw_temperature: float,
) -> list[dict[str, float | str]]:
    split = _calibration_split(features, target, training_df, latest_season)
    if split is None:
        return []
    train_X, train_y, eval_X, eval_y = split
    if len(eval_X) < 30:
        return []

    model = _build_rf_model()
    model.fit(train_X, train_y)
    raw_probs = model.predict_proba(eval_X)
    classes = [str(c) for c in model.classes_]
    actual = eval_y.astype(str).tolist()

    prob_rows: list[dict[str, float]] = []
    for row in raw_probs:
        class_probs_raw = {cls: float(prob) for cls, prob in zip(classes, row)}
        class_probs = _calibrate_class_probabilities(class_probs_raw, calibrators)
        class_probs = _apply_draw_temperature(class_probs, draw_temperature)
        prob_rows.append(class_probs)

    rows: list[dict[str, float | str]] = []
    for c in ["W", "D", "L"]:
        pred_mean = float(sum(float(r.get(c, 0.0)) for r in prob_rows) / max(1, len(prob_rows)))
        actual_rate = float(sum(1.0 for y in actual if str(y) == c) / max(1, len(actual)))
        rows.append(
            {
                "class": c,
                "predictedPct": round(pred_mean * 100.0, 1),
                "actualPct": round(actual_rate * 100.0, 1),
                "deltaPct": round((pred_mean - actual_rate) * 100.0, 1),
            }
        )
    return rows


def _predict_binary_probability(
    model: RandomForestClassifier | None,
    calibrator: IsotonicRegression | None,
    features: pd.DataFrame | None,
    fallback_prob: float,
) -> float:
    if model is None or features is None:
        return float(max(0.0, min(1.0, fallback_prob)))
    probs = model.predict_proba(features)[0]
    class_list = [int(c) for c in model.classes_]
    if 1 in class_list:
        positive_prob = float(probs[class_list.index(1)])
    else:
        positive_prob = float(fallback_prob)
    if calibrator is not None:
        positive_prob = float(calibrator.predict([positive_prob])[0])
    return float(max(0.0, min(1.0, positive_prob)))


def _get_model_bundle(all_matches: pd.DataFrame) -> ModelBundle:
    global _MODEL_BUNDLE, _MODEL_META
    if _MODEL_BUNDLE is not None:
        if "modelSource" not in _MODEL_BUNDLE.metrics:
            _apply_model_source_metadata(_MODEL_BUNDLE, source="memory", cache_key=str(_MODEL_META.get("cacheKey", "")))
        return _MODEL_BUNDLE

    normalized = _normalize_matches_frame(all_matches)
    home_df = _prepare_home_training_frame(normalized)
    if home_df.empty:
        raise ValueError("insufficient_training_data")

    training_df, latest_season, min_season = _windowed_training_frame(home_df)
    training_df = training_df.sort_values(["date", "time"]).reset_index(drop=True)
    context_df = normalized[normalized["season"] >= min_season].copy()
    feature_context = _build_feature_context(
        context_df,
        form_window=MODEL_FORM_WINDOW,
        h2h_window=MODEL_H2H_WINDOW,
    )
    availability_index, team_news_explanations = _build_availability_index()
    feature_context.availability_index = availability_index
    feature_context.team_news_explanations = team_news_explanations
    model_cache_key = _build_model_cache_key(
        training_df=training_df,
        latest_season=latest_season,
        min_season=min_season,
        availability_index=availability_index,
    )
    artifact_bundle = _load_model_bundle_from_artifact(model_cache_key)
    if artifact_bundle is not None:
        _MODEL_BUNDLE = artifact_bundle
        return _MODEL_BUNDLE

    teams = sorted(set(training_df["team"].astype(str)) | set(training_df["opponent"].astype(str)))
    team_to_code = {team: idx for idx, team in enumerate(teams)}

    feature_rows: list[dict] = []
    targets: list[str] = []
    home_goal_targets: list[float] = []
    away_goal_targets: list[float] = []
    for row in training_df.itertuples(index=False):
        home_code = team_to_code.get(row.team)
        away_code = team_to_code.get(row.opponent)
        if home_code is None or away_code is None:
            continue

        match_date = _to_naive_timestamp(row.date)
        feature_rows.append(
            _build_feature_row(
                feature_context,
                row.team,
                row.opponent,
                match_date,
                int(row.hour),
                int(row.day),
                int(row.season),
                int(home_code),
                int(away_code),
            )
        )
        targets.append(str(row.result))
        home_goal_targets.append(float(getattr(row, "gf", 0.0)))
        away_goal_targets.append(float(getattr(row, "ga", 0.0)))

    if not feature_rows:
        raise ValueError("insufficient_training_features")

    features = pd.DataFrame(feature_rows)
    target = pd.Series(targets)
    feature_columns = list(features.columns)

    model = _build_rf_model()
    model.fit(features, target)
    home_goals_model = _build_goal_regressor()
    away_goals_model = _build_goal_regressor()
    home_goals_model.fit(features, pd.Series(home_goal_targets))
    away_goals_model.fit(features, pd.Series(away_goal_targets))

    total_goals = pd.Series(home_goal_targets) + pd.Series(away_goal_targets)
    y_over25 = (total_goals >= 3).astype(int)
    y_over35 = (total_goals >= 4).astype(int)
    y_btts = ((pd.Series(home_goal_targets) >= 1) & (pd.Series(away_goal_targets) >= 1)).astype(int)
    over25_model, over25_calibrator = _fit_binary_market_with_calibration(features, y_over25, training_df, latest_season)
    over35_model, over35_calibrator = _fit_binary_market_with_calibration(features, y_over35, training_df, latest_season)
    btts_model, btts_calibrator = _fit_binary_market_with_calibration(features, y_btts, training_df, latest_season)

    eval_accuracy, evaluation_scope = _evaluate_accuracy(features, target, training_df, latest_season)
    probability_calibrators = _build_probability_calibrators(features, target, training_df, latest_season)
    draw_temperature = _fit_draw_temperature(features, target, training_df, latest_season, probability_calibrators)
    classwise_calibration = _classwise_calibration_summary(
        features,
        target,
        training_df,
        latest_season,
        probability_calibrators,
        draw_temperature,
    )
    train_home_mae = float(mean_absolute_error(pd.Series(home_goal_targets), home_goals_model.predict(features)))
    train_away_mae = float(mean_absolute_error(pd.Series(away_goal_targets), away_goals_model.predict(features)))

    metrics = {
        "accuracyPct": int(round(eval_accuracy * 100)),
        "evaluationScope": evaluation_scope,
        "sampleCount": int(len(features)),
        "lastTrainedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "trainingYears": int(max(1, MODEL_TRAINING_YEARS)),
        "formWindow": int(max(1, MODEL_FORM_WINDOW)),
        "h2hWindow": int(max(1, MODEL_H2H_WINDOW)),
        "rfTrees": int(max(50, MODEL_RF_TREES)),
        "rfMinSamplesSplit": int(max(2, MODEL_RF_MIN_SAMPLES_SPLIT)),
        "rfMaxDepth": int(MODEL_RF_MAX_DEPTH),
        "scoreDispersion": round(float(max(0.05, min(0.5, MODEL_SCORE_DISPERSION))), 3),
        "oddsBlendWeight": round(float(max(0.0, min(0.45, MODEL_ODDS_BLEND_WEIGHT))), 3),
        "availabilityTeams": int(len(feature_context.availability_index)),
        "probabilityCalibrators": int(len(probability_calibrators)),
        "drawTemperature": round(float(draw_temperature), 3),
        "classwiseCalibration": classwise_calibration,
        "goalMarketModels": int(sum(1 for m in [over25_model, over35_model, btts_model] if m is not None)),
        "goalMarketCalibrators": int(
            sum(1 for c in [over25_calibrator, over35_calibrator, btts_calibrator] if c is not None)
        ),
        "homeGoalMae": round(train_home_mae, 3),
        "awayGoalMae": round(train_away_mae, 3),
    }

    _MODEL_BUNDLE = ModelBundle(
        model=model,
        home_goals_model=home_goals_model,
        away_goals_model=away_goals_model,
        probability_calibrators=probability_calibrators,
        over25_model=over25_model,
        over35_model=over35_model,
        btts_model=btts_model,
        over25_calibrator=over25_calibrator,
        over35_calibrator=over35_calibrator,
        btts_calibrator=btts_calibrator,
        draw_temperature=float(draw_temperature),
        team_to_code=team_to_code,
        metrics=metrics,
        feature_columns=feature_columns,
        feature_context=feature_context,
        training_latest_season=latest_season,
    )
    _save_model_bundle_to_artifact(model_cache_key, _MODEL_BUNDLE)
    if "modelSource" not in _MODEL_BUNDLE.metrics:
        _apply_model_source_metadata(_MODEL_BUNDLE, source="trained", cache_key=model_cache_key)
    return _MODEL_BUNDLE


def _team_goal_rates(all_matches: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    normalized = _normalize_matches_frame(all_matches)
    home_rows = normalized[normalized["venue"].astype(str).str.lower() == "home"]
    away_rows = normalized[normalized["venue"].astype(str).str.lower() == "away"]

    home_attack = home_rows.groupby("team")["gf"].mean().to_dict()
    home_defense = home_rows.groupby("team")["ga"].mean().to_dict()
    away_attack = away_rows.groupby("team")["gf"].mean().to_dict()
    away_defense = away_rows.groupby("team")["ga"].mean().to_dict()

    return (
        {
            "home_attack": home_attack,
            "home_defense": home_defense,
            "away_attack": away_attack,
            "away_defense": away_defense,
        },
        {
            "home_attack_default": float(home_rows["gf"].mean()) if not home_rows.empty else 1.3,
            "home_defense_default": float(home_rows["ga"].mean()) if not home_rows.empty else 1.1,
            "away_attack_default": float(away_rows["gf"].mean()) if not away_rows.empty else 1.1,
            "away_defense_default": float(away_rows["ga"].mean()) if not away_rows.empty else 1.3,
        },
    )


def _fallback_probabilities(home: str, away: str, standings: list[dict]) -> tuple[int, int, int]:
    points_by_team = {row["name"]: row["points"] for row in standings}
    played_by_team = {row["name"]: max(row["played"], 1) for row in standings}

    home_ppm = points_by_team.get(home, 1) / played_by_team.get(home, 1)
    away_ppm = points_by_team.get(away, 1) / played_by_team.get(away, 1)

    raw_home = int(44 + (home_ppm - away_ppm) * 18 + 6)
    raw_away = int(34 + (away_ppm - home_ppm) * 18)
    raw_draw = 22
    return _normalize_three_way(max(raw_home, 8), raw_draw, max(raw_away, 8))


def _availability_penalty(context: FeatureContext, team: str) -> float:
    return _availability_penalty_with_resilience(
        context=context,
        team=team,
        match_date=pd.Timestamp.utcnow().tz_localize(None),
    )


def _team_resilience_multiplier(context: FeatureContext, team: str, match_date: pd.Timestamp) -> float:
    history = context.team_history.get(team, [])
    if not history:
        return 1.0

    recent_points: list[float] = []
    baseline_points: list[float] = []
    for item in reversed(history):
        if item["date"] >= match_date:
            continue
        points = float(item["points"])
        if len(recent_points) < max(3, context.form_window):
            recent_points.append(points)
        elif len(baseline_points) < 20:
            baseline_points.append(points)
        if len(recent_points) >= max(3, context.form_window) and len(baseline_points) >= 8:
            break

    if len(recent_points) < 3:
        return 1.0

    recent_ppg = sum(recent_points) / max(len(recent_points), 1)
    baseline_ppg = sum(baseline_points) / max(len(baseline_points), 1) if baseline_points else context.default_ppg
    baseline_ppg = max(0.25, baseline_ppg)

    # Teams outperforming baseline are treated as more resilient to absences.
    ratio = recent_ppg / baseline_ppg
    if ratio >= 1.12:
        return 0.9
    if ratio <= 0.88:
        return 1.12
    return 1.0


def _availability_penalty_with_resilience(context: FeatureContext, team: str, match_date: pd.Timestamp) -> float:
    entry = context.availability_index.get(team, {})
    injured = float(entry.get("injured", 0.0))
    suspended = float(entry.get("suspended", 0.0))
    key_impact = float(entry.get("key_impact", 0.0))

    # Non-linear weighted penalty for missing squad strength with resilience scaling.
    raw_penalty = (injured * 0.02) + (suspended * 0.042) + ((key_impact**1.2) * 0.24)
    scaled_penalty = raw_penalty * _team_resilience_multiplier(context, team, match_date)
    return max(0.0, min(0.3, scaled_penalty))


def _apply_availability_adjustment(
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    context: FeatureContext,
    home: str,
    away: str,
) -> tuple[int, int, int]:
    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    home_penalty = _availability_penalty_with_resilience(context, home, now_ts)
    away_penalty = _availability_penalty_with_resilience(context, away, now_ts)
    penalty_delta = away_penalty - home_penalty

    shift = int(round(penalty_delta * 46))
    shift = max(-12, min(12, shift))

    adj_home = max(1, home_prob + shift)
    adj_away = max(1, away_prob - shift)
    adj_draw = max(1, draw_prob)
    return _normalize_three_way(adj_home, adj_draw, adj_away)


def _blend_low_coverage_with_fallback(
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    fallback_home: int,
    fallback_draw: int,
    fallback_away: int,
    home_history_count: int,
    away_history_count: int,
) -> tuple[int, int, int]:
    # Newly promoted / sparse-history teams are noisy in the model feature space.
    # Blend toward standings fallback when minimum coverage is low.
    min_history = max(0, min(int(home_history_count), int(away_history_count)))
    if min_history >= 60:
        return home_prob, draw_prob, away_prob

    model_weight = max(0.25, min(1.0, min_history / 60.0))
    fallback_weight = 1.0 - model_weight
    blended_home = int(round((home_prob * model_weight) + (fallback_home * fallback_weight)))
    blended_draw = int(round((draw_prob * model_weight) + (fallback_draw * fallback_weight)))
    blended_away = int(round((away_prob * model_weight) + (fallback_away * fallback_weight)))
    return _normalize_three_way(blended_home, blended_draw, blended_away)


def _blend_with_odds_priors(
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    odds_priors: tuple[float, float, float] | None,
    blend_weight: float,
) -> tuple[int, int, int]:
    if odds_priors is None:
        return home_prob, draw_prob, away_prob
    w = max(0.0, min(0.45, float(blend_weight)))
    if w <= 0:
        return home_prob, draw_prob, away_prob

    mh, md, ma = _normalize_three_way_probs(home_prob / 100.0, draw_prob / 100.0, away_prob / 100.0)
    oh, od, oa = odds_priors
    bh = ((1.0 - w) * mh) + (w * oh)
    bd = ((1.0 - w) * md) + (w * od)
    ba = ((1.0 - w) * ma) + (w * oa)
    return _normalize_three_way(int(round(bh * 100)), int(round(bd * 100)), int(round(ba * 100)))


def _fixture_kickoff(fixture_time: str) -> pd.Timestamp:
    if not fixture_time:
        return pd.Timestamp.utcnow().tz_localize(None)

    ts = pd.to_datetime(str(fixture_time), errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.utcnow().tz_localize(None)

    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert(None)
    return ts


def _implied_prob_from_decimal_odds(odds_value: object) -> float | None:
    try:
        odds = float(odds_value)
    except (TypeError, ValueError):
        return None
    if odds <= 1.01:
        return None
    return 1.0 / odds


def _extract_fixture_odds_priors(fixture: dict | None) -> tuple[float, float, float] | None:
    if not fixture:
        return None
    home_raw = fixture.get("oddsHome")
    draw_raw = fixture.get("oddsDraw")
    away_raw = fixture.get("oddsAway")
    if home_raw is None or draw_raw is None or away_raw is None:
        home_raw = fixture.get("homeOdds")
        draw_raw = fixture.get("drawOdds")
        away_raw = fixture.get("awayOdds")

    home_p = _implied_prob_from_decimal_odds(home_raw)
    draw_p = _implied_prob_from_decimal_odds(draw_raw)
    away_p = _implied_prob_from_decimal_odds(away_raw)
    if home_p is None or draw_p is None or away_p is None:
        return None
    return _normalize_three_way_probs(home_p, draw_p, away_p)


def _poisson_pmf(goals: int, lam: float) -> float:
    lam = max(0.05, float(lam))
    g = max(0, int(goals))
    return math.exp(-lam) * (lam**g) / math.factorial(g)


def _scoreline_distribution(
    home_xg: float,
    away_xg: float,
    max_goals: int = 8,
) -> list[tuple[int, int, float]]:
    # Poisson-mixture adds realistic tail outcomes without fully destabilizing baseline accuracy.
    base_dispersion = max(0.05, min(0.45, float(MODEL_SCORE_DISPERSION)))
    total_xg = float(home_xg + away_xg)
    margin = abs(float(home_xg - away_xg))
    dynamic_dispersion = base_dispersion + (0.06 if total_xg >= 2.9 else 0.0) + (0.04 if margin >= 0.7 else 0.0)
    dynamic_dispersion = max(0.05, min(0.5, dynamic_dispersion))
    scales = [1.0 - dynamic_dispersion, 1.0, 1.0 + dynamic_dispersion]
    weights = [0.22, 0.56, 0.22]

    rows: list[tuple[int, int, float]] = []
    for hg in range(0, int(max_goals) + 1):
        base_h = 0.0
        for weight, scale in zip(weights, scales):
            base_h += float(weight) * _poisson_pmf(hg, home_xg * scale)
        for ag in range(0, int(max_goals) + 1):
            base_a = 0.0
            for weight, scale in zip(weights, scales):
                base_a += float(weight) * _poisson_pmf(ag, away_xg * scale)
            prob = base_h * base_a

            # Controlled tail boost for genuinely open/high-event fixtures.
            if (hg + ag) >= 5 and total_xg >= 2.85:
                prob *= 1.16
            if hg >= 3 and ag >= 2 and total_xg >= 2.7:
                prob *= 1.08
            if ag >= 3 and hg >= 2 and total_xg >= 2.7:
                prob *= 1.08
            if hg == 0 and ag == 0 and total_xg >= 2.2:
                prob *= 0.62
            if (hg + ag) <= 1 and total_xg >= 2.45:
                prob *= 0.72

            rows.append((hg, ag, float(prob)))

    total_prob = sum(prob for _, _, prob in rows)
    if total_prob <= 0:
        return [(0, 0, 1.0)]
    return [(hg, ag, float(prob / total_prob)) for hg, ag, prob in rows]


def _expected_goals_from_context(
    home_attack: float,
    away_defense: float,
    away_attack: float,
    home_defense: float,
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    home_penalty: float,
    away_penalty: float,
) -> tuple[float, float]:
    home_xg = (float(home_attack) + float(away_defense)) / 2.0
    away_xg = (float(away_attack) + float(home_defense)) / 2.0

    # Calibrate xG by model outcome skew.
    edge = (home_prob - away_prob) / 100.0
    draw_bias = draw_prob / 100.0
    home_xg *= 1.0 + (0.26 * edge) - (0.08 * draw_bias)
    away_xg *= 1.0 - (0.22 * edge) - (0.08 * draw_bias)

    # Stronger availability impact with non-linear floor.
    home_xg *= max(0.5, 1.0 - (home_penalty * 1.35))
    away_xg *= max(0.5, 1.0 - (away_penalty * 1.35))

    home_xg = max(0.25, min(3.6, home_xg))
    away_xg = max(0.25, min(3.3, away_xg))
    return home_xg, away_xg


def _calibrate_goal_expectation(
    home_xg: float,
    away_xg: float,
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    goal_defaults: dict[str, float],
) -> tuple[float, float]:
    home_xg = max(0.2, float(home_xg))
    away_xg = max(0.2, float(away_xg))

    current_total = home_xg + away_xg
    league_total = float(goal_defaults.get("home_attack_default", 1.35)) + float(
        goal_defaults.get("away_attack_default", 1.2)
    )
    league_total = max(2.1, min(3.4, league_total))

    edge = abs(float(home_prob - away_prob)) / 100.0
    draw_bias = float(draw_prob) / 100.0
    total_target = league_total * (1.0 + (0.16 * edge) - (0.2 * draw_bias))
    total_target = max(2.2, min(3.8, total_target))
    blended_total = (0.45 * current_total) + (0.55 * total_target)

    scale = blended_total / max(0.3, current_total)
    home_xg *= scale
    away_xg *= scale

    return max(0.35, min(4.0, home_xg)), max(0.35, min(3.7, away_xg))


def _most_likely_scoreline(
    home_xg: float,
    away_xg: float,
    home_prob: int,
    draw_prob: int,
    away_prob: int,
) -> tuple[int, int]:
    preferred_outcome = "D"
    if home_prob >= draw_prob and home_prob >= away_prob:
        preferred_outcome = "H"
    elif away_prob >= draw_prob and away_prob >= home_prob:
        preferred_outcome = "A"
    preferred_gap = 0
    if preferred_outcome == "H":
        preferred_gap = int(home_prob - max(draw_prob, away_prob))
    elif preferred_outcome == "A":
        preferred_gap = int(away_prob - max(draw_prob, home_prob))

    expected_total = float(home_xg + away_xg)
    expected_margin = float(home_xg - away_xg)

    best_score = (1, 1)
    best_weight = -1.0
    for hg, ag, mass in _scoreline_distribution(home_xg=home_xg, away_xg=away_xg, max_goals=7):

            outcome = "D"
            if hg > ag:
                outcome = "H"
            elif ag > hg:
                outcome = "A"

            # Keep scoreline aligned with model probabilities.
            if preferred_outcome == outcome:
                alignment = 1.15
            elif outcome == "D" and draw_prob >= 28:
                alignment = 1.05
            else:
                alignment = 0.92
            if preferred_outcome in {"H", "A"} and outcome == "D" and preferred_gap >= 5:
                alignment *= 0.55

            total_distance = abs((hg + ag) - expected_total)
            margin_distance = abs((hg - ag) - expected_margin)
            shape_fit = 1.0 / (1.0 + (0.22 * total_distance) + (0.18 * margin_distance))
            weight = mass * alignment * shape_fit

            # Avoid over-selecting 0-0 when the game profile is not truly low-event.
            if hg == 0 and ag == 0:
                if expected_total >= 2.0 or draw_prob < 34:
                    weight *= 0.52
                else:
                    weight *= 0.82

            if (hg + ag) <= 1 and expected_total >= 2.35:
                weight *= 0.72

            if weight > best_weight:
                best_weight = weight
                best_score = (hg, ag)

    return best_score


def _top_scoreline_candidates(
    home_xg: float,
    away_xg: float,
    home_prob: int,
    draw_prob: int,
    away_prob: int,
    limit: int = 6,
) -> list[dict]:
    _ = (home_prob, draw_prob, away_prob)
    rows: list[tuple[tuple[int, int], float]] = [
        ((int(hg), int(ag)), float(prob))
        for hg, ag, prob in _scoreline_distribution(home_xg=home_xg, away_xg=away_xg, max_goals=8)
    ]

    rows.sort(key=lambda item: item[1], reverse=True)
    top = rows[: max(1, int(limit))]
    denom = sum(float(item[1]) for item in rows)
    if denom <= 0:
        return []

    return [
        {
            "score": f"{int(score[0])}-{int(score[1])}",
            "probabilityPct": round((float(base_prob) / denom) * 100.0, 1),
        }
        for score, base_prob in top
    ]


def _goal_outlook_probabilities(home_xg: float, away_xg: float) -> dict[str, float]:
    rows = _scoreline_distribution(home_xg=home_xg, away_xg=away_xg, max_goals=9)

    total_mass = sum(prob for _, _, prob in rows)
    if total_mass <= 0:
        return {
            "home3PlusPct": 0.0,
            "away3PlusPct": 0.0,
            "total3PlusPct": 0.0,
            "total4PlusPct": 0.0,
            "bttsPct": 0.0,
        }

    home3_plus = sum(prob for hg, _, prob in rows if hg >= 3)
    away3_plus = sum(prob for _, ag, prob in rows if ag >= 3)
    total3_plus = sum(prob for hg, ag, prob in rows if (hg + ag) >= 3)
    total4_plus = sum(prob for hg, ag, prob in rows if (hg + ag) >= 4)
    btts = sum(prob for hg, ag, prob in rows if hg >= 1 and ag >= 1)

    return {
        "home3PlusPct": round((home3_plus / total_mass) * 100.0, 1),
        "away3PlusPct": round((away3_plus / total_mass) * 100.0, 1),
        "total3PlusPct": round((total3_plus / total_mass) * 100.0, 1),
        "total4PlusPct": round((total4_plus / total_mass) * 100.0, 1),
        "bttsPct": round((btts / total_mass) * 100.0, 1),
    }


def _goal_market_probabilities(goal_outlook: dict[str, float]) -> dict[str, float]:
    over25 = float(goal_outlook.get("total3PlusPct", 0.0))
    over35 = float(goal_outlook.get("total4PlusPct", 0.0))
    btts = float(goal_outlook.get("bttsPct", 0.0))
    return {
        "over25Pct": round(over25, 1),
        "under25Pct": round(max(0.0, 100.0 - over25), 1),
        "over35Pct": round(over35, 1),
        "under35Pct": round(max(0.0, 100.0 - over35), 1),
        "bttsPct": round(btts, 1),
        "noBttsPct": round(max(0.0, 100.0 - btts), 1),
    }


def _goal_market_probabilities_from_models(
    *,
    model_bundle: ModelBundle,
    features: pd.DataFrame | None,
    goal_outlook: dict[str, float],
) -> dict[str, float]:
    fallback_over25 = float(goal_outlook.get("total3PlusPct", 0.0)) / 100.0
    fallback_over35 = float(goal_outlook.get("total4PlusPct", 0.0)) / 100.0
    fallback_btts = float(goal_outlook.get("bttsPct", 0.0)) / 100.0
    over25_prob_model = _predict_binary_probability(
        model_bundle.over25_model,
        model_bundle.over25_calibrator,
        features,
        fallback_over25,
    )
    over35_prob_model = _predict_binary_probability(
        model_bundle.over35_model,
        model_bundle.over35_calibrator,
        features,
        fallback_over35,
    )
    btts_prob_model = _predict_binary_probability(
        model_bundle.btts_model,
        model_bundle.btts_calibrator,
        features,
        fallback_btts,
    )

    # Blend calibrated model probabilities with xG-derived fallback to reduce
    # isotonic plateaus and keep fixture-level variation.
    blend_model = 0.72
    blend_fallback = 0.28
    over25_prob = (blend_model * over25_prob_model) + (blend_fallback * fallback_over25)
    over35_prob = (blend_model * over35_prob_model) + (blend_fallback * fallback_over35)
    btts_prob = (blend_model * btts_prob_model) + (blend_fallback * fallback_btts)

    over25_prob = float(max(0.0, min(1.0, over25_prob)))
    over35_prob = float(max(0.0, min(1.0, over35_prob)))
    btts_prob = float(max(0.0, min(1.0, btts_prob)))
    return {
        "over25Pct": round(over25_prob * 100.0, 1),
        "under25Pct": round((1.0 - over25_prob) * 100.0, 1),
        "over35Pct": round(over35_prob * 100.0, 1),
        "under35Pct": round((1.0 - over35_prob) * 100.0, 1),
        "bttsPct": round(btts_prob * 100.0, 1),
        "noBttsPct": round((1.0 - btts_prob) * 100.0, 1),
    }


def _refine_scoreline(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    home_prob: int,
    draw_prob: int,
    away_prob: int,
) -> tuple[int, int]:
    h = int(home_goals)
    a = int(away_goals)
    preferred = "D"
    if home_prob >= draw_prob and home_prob >= away_prob:
        preferred = "H"
    elif away_prob >= draw_prob and away_prob >= home_prob:
        preferred = "A"

    total_xg = float(home_xg + away_xg)

    if preferred == "H":
        if h <= a:
            h = a + 1
        if h == 1 and a == 0 and home_prob >= 50 and float(home_xg) >= 1.45:
            h, a = 2, 1
        if (h + a) <= 1 and total_xg >= 2.15:
            h, a = (2, 1) if away_xg >= 0.8 else (2, 0)
        if h == 1 and a == 0 and total_xg >= 2.55 and home_prob >= 47:
            h, a = 2, 1
    elif preferred == "A":
        if a <= h:
            a = h + 1
        if h == 0 and a == 1 and away_prob >= 50 and float(away_xg) >= 1.45:
            h, a = 1, 2
        if (h + a) <= 1 and total_xg >= 2.15:
            h, a = (1, 2) if home_xg >= 0.8 else (0, 2)
        if h == 0 and a == 1 and total_xg >= 2.55 and away_prob >= 47:
            h, a = 1, 2
    else:
        if h != a:
            common = int(round((float(home_xg) + float(away_xg)) / 2.0))
            common = max(0, min(3, common))
            h, a = common, common
        if h == 0 and a == 0 and draw_prob < 42:
            h, a = 1, 1
        if (h + a) == 0 and total_xg >= 1.9:
            h, a = 1, 1
        if h == 1 and a == 1 and total_xg >= 2.75 and draw_prob >= 35:
            h, a = 2, 2

    return max(0, min(h, 5)), max(0, min(a, 5))


def _prediction_explanation_payload(
    context: FeatureContext,
    home: str,
    away: str,
    home_penalty: float,
    away_penalty: float,
) -> dict:
    home_players = context.team_news_explanations.get(home, [])
    away_players = context.team_news_explanations.get(away, [])
    return {
        "homePenalty": round(float(home_penalty), 4),
        "awayPenalty": round(float(away_penalty), 4),
        "penaltyDelta": round(float(away_penalty - home_penalty), 4),
        "homeTopUnavailable": home_players[:3],
        "awayTopUnavailable": away_players[:3],
    }


def _predict_fixture(
    display_home: str,
    display_away: str,
    idx: int,
    standings: list[dict],
    model_bundle: ModelBundle,
    goal_rates: dict[str, dict[str, float]],
    goal_defaults: dict[str, float],
    kickoff: pd.Timestamp,
    fixture_payload: dict | None = None,
) -> dict:
    home = _canonical_team_name(display_home)
    away = _canonical_team_name(display_away)

    home_code = model_bundle.team_to_code.get(home)
    away_code = model_bundle.team_to_code.get(away)
    home_history_count = int(len(model_bundle.feature_context.team_history.get(home, [])))
    away_history_count = int(len(model_bundle.feature_context.team_history.get(away, [])))
    fallback_home, fallback_draw, fallback_away = _fallback_probabilities(home, away, standings)
    features: pd.DataFrame | None = None

    odds_priors = _extract_fixture_odds_priors(fixture_payload)
    if home_code is not None and away_code is not None:
        feature_payload = _build_feature_row(
            model_bundle.feature_context,
            home,
            away,
            _to_naive_timestamp(kickoff),
            int(kickoff.hour),
            int(kickoff.day_of_week),
            int(model_bundle.training_latest_season),
            int(home_code),
            int(away_code),
        )
        features = pd.DataFrame([feature_payload], columns=model_bundle.feature_columns)
        probabilities = model_bundle.model.predict_proba(features)[0]
        class_probs_raw = {str(cls): float(prob) for cls, prob in zip(model_bundle.model.classes_, probabilities)}
        class_probs = _calibrate_class_probabilities(class_probs_raw, model_bundle.probability_calibrators)
        class_probs = _apply_draw_temperature(class_probs, model_bundle.draw_temperature)
        home_prob = int(round(class_probs.get("W", 0.34) * 100))
        draw_prob = int(round(class_probs.get("D", 0.22) * 100))
        away_prob = int(round(class_probs.get("L", 0.34) * 100))
        home_prob, draw_prob, away_prob = _normalize_three_way(home_prob, draw_prob, away_prob)
        home_prob, draw_prob, away_prob = _blend_low_coverage_with_fallback(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            fallback_home=fallback_home,
            fallback_draw=fallback_draw,
            fallback_away=fallback_away,
            home_history_count=home_history_count,
            away_history_count=away_history_count,
        )
        home_prob, draw_prob, away_prob = _blend_with_odds_priors(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            odds_priors=odds_priors,
            blend_weight=MODEL_ODDS_BLEND_WEIGHT,
        )
    else:
        home_prob, draw_prob, away_prob = fallback_home, fallback_draw, fallback_away
        home_prob, draw_prob, away_prob = _blend_with_odds_priors(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            odds_priors=odds_priors,
            blend_weight=MODEL_ODDS_BLEND_WEIGHT,
        )

    home_prob, draw_prob, away_prob = _apply_availability_adjustment(
        home_prob,
        draw_prob,
        away_prob,
        model_bundle.feature_context,
        home,
        away,
    )

    home_attack = goal_rates["home_attack"].get(home, goal_defaults["home_attack_default"])
    away_defense = goal_rates["away_defense"].get(away, goal_defaults["away_defense_default"])
    away_attack = goal_rates["away_attack"].get(away, goal_defaults["away_attack_default"])
    home_defense = goal_rates["home_defense"].get(home, goal_defaults["home_defense_default"])

    home_penalty = _availability_penalty_with_resilience(model_bundle.feature_context, home, kickoff)
    away_penalty = _availability_penalty_with_resilience(model_bundle.feature_context, away, kickoff)
    context_home_xg, context_away_xg = _expected_goals_from_context(
        home_attack=home_attack,
        away_defense=away_defense,
        away_attack=away_attack,
        home_defense=home_defense,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        home_penalty=home_penalty,
        away_penalty=away_penalty,
    )
    if features is not None:
        model_home_xg = float(model_bundle.home_goals_model.predict(features)[0])
        model_away_xg = float(model_bundle.away_goals_model.predict(features)[0])
        model_home_xg = max(0.2, min(3.9, model_home_xg))
        model_away_xg = max(0.2, min(3.6, model_away_xg))
        model_home_xg *= max(0.55, 1.0 - (home_penalty * 1.15))
        model_away_xg *= max(0.55, 1.0 - (away_penalty * 1.15))
        home_xg = (0.62 * model_home_xg) + (0.38 * context_home_xg)
        away_xg = (0.62 * model_away_xg) + (0.38 * context_away_xg)
    else:
        home_xg = context_home_xg
        away_xg = context_away_xg

    home_xg, away_xg = _calibrate_goal_expectation(
        home_xg=home_xg,
        away_xg=away_xg,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        goal_defaults=goal_defaults,
    )

    pred_home, pred_away = _most_likely_scoreline(
        home_xg=home_xg,
        away_xg=away_xg,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
    )
    pred_home, pred_away = _refine_scoreline(
        home_goals=pred_home,
        away_goals=pred_away,
        home_xg=home_xg,
        away_xg=away_xg,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
    )
    top_scores = _top_scoreline_candidates(
        home_xg=home_xg,
        away_xg=away_xg,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        limit=6,
    )
    goal_outlook = _goal_outlook_probabilities(home_xg=home_xg, away_xg=away_xg)
    goal_markets = _goal_market_probabilities_from_models(
        model_bundle=model_bundle,
        features=features,
        goal_outlook=goal_outlook,
    )

    confidence = max(home_prob, draw_prob, away_prob)

    return {
        "id": idx,
        "fixture": f"{display_home} vs {display_away}",
        "home": display_home,
        "away": display_away,
        "confidence": int(confidence),
        "score": f"{pred_home}-{pred_away}",
        "probabilities": {
            "home": int(home_prob),
            "draw": int(draw_prob),
            "away": int(away_prob),
        },
        "expectedGoals": {
            "home": round(float(home_xg), 2),
            "away": round(float(away_xg), 2),
            "total": round(float(home_xg + away_xg), 2),
        },
        "topScorelines": top_scores,
        "goalOutlook": goal_outlook,
        "goalMarkets": goal_markets,
        "dataCoverage": {
            "homeMatchesInTraining": int(home_history_count),
            "awayMatchesInTraining": int(away_history_count),
            "usedFallbackProbabilities": bool(features is None),
            "usedOddsPriors": bool(odds_priors is not None and MODEL_ODDS_BLEND_WEIGHT > 0),
        },
        "explanation": _prediction_explanation_payload(
            context=model_bundle.feature_context,
            home=home,
            away=away,
            home_penalty=home_penalty,
            away_penalty=away_penalty,
        ),
    }


def build_predictions(
    all_matches: pd.DataFrame,
    standings: list[dict],
    fixtures: list[dict],
) -> list[dict]:
    if not fixtures:
        return []

    model_bundle = _get_model_bundle(all_matches)
    goal_rates, goal_defaults = _team_goal_rates(all_matches)

    predictions = []
    for idx, fixture in enumerate(fixtures[:2], start=1):
        kickoff = _fixture_kickoff(str(fixture.get("time", "")))
        predictions.append(
            _predict_fixture(
                str(fixture.get("home", "")),
                str(fixture.get("away", "")),
                idx,
                standings,
                model_bundle,
                goal_rates,
                goal_defaults,
                kickoff,
                fixture_payload=fixture,
            )
        )

    return predictions


def get_model_metrics(all_matches: pd.DataFrame) -> dict:
    bundle = _get_model_bundle(all_matches)
    return bundle.metrics


def get_model_info(all_matches: pd.DataFrame) -> dict:
    bundle = _get_model_bundle(all_matches)
    metrics = dict(bundle.metrics)
    artifact_path = str(metrics.get("modelArtifactPath", str(MODEL_ARTIFACT_PATH)))
    return {
        "available": True,
        "source": str(metrics.get("modelSource", "memory")),
        "trainingLatestSeason": int(bundle.training_latest_season),
        "featureCount": int(len(bundle.feature_columns)),
        "teamCount": int(len(bundle.team_to_code)),
        "artifact": {
            "enabled": bool(MODEL_ARTIFACT_ENABLED),
            "path": artifact_path,
            "exists": bool(MODEL_ARTIFACT_PATH.exists()),
            "savedAt": metrics.get("modelArtifactSavedAt"),
            "cacheKey": metrics.get("modelCacheKey"),
        },
        "metrics": metrics,
    }


def reset_model_cache(*, delete_artifact: bool = False) -> None:
    global _MODEL_BUNDLE, _MODEL_META
    _MODEL_BUNDLE = None
    _MODEL_META = {}
    if delete_artifact and MODEL_ARTIFACT_PATH.exists():
        try:
            MODEL_ARTIFACT_PATH.unlink()
        except Exception:
            pass


def get_available_teams(all_matches: pd.DataFrame) -> list[str]:
    model_bundle = _get_model_bundle(all_matches)
    return sorted(model_bundle.team_to_code.keys())


def predict_single_match(
    all_matches: pd.DataFrame,
    standings: list[dict],
    home_team: str,
    away_team: str,
) -> dict:
    model_bundle = _get_model_bundle(all_matches)
    goal_rates, goal_defaults = _team_goal_rates(all_matches)

    kickoff = pd.Timestamp.utcnow().tz_localize(None)
    return _predict_fixture(
        home_team,
        away_team,
        1,
        standings,
        model_bundle,
        goal_rates,
        goal_defaults,
        kickoff,
        fixture_payload=None,
    )
