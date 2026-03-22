from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class ValidationError(ValueError):
    pass


@dataclass
class PredictMatchInput:
    home_team: str
    away_team: str


@dataclass
class SimulateSeasonInput:
    iterations: int
    volatility: float
    cutoff_date: str | None


@dataclass
class TeamAvailabilityRowInput:
    team: str
    injured: float
    suspended: float
    key_impact: float


@dataclass
class TeamNewsRowInput:
    team: str
    player: str
    status: str
    position: str
    importance: float
    expected_return: str
    source: str
    updated_at: str


def validate_predict_match_payload(payload: dict) -> PredictMatchInput:
    if not isinstance(payload, dict):
        raise ValidationError("invalid_payload")

    home_team = str(payload.get("homeTeam", "")).strip()
    away_team = str(payload.get("awayTeam", "")).strip()

    if not home_team or not away_team:
        raise ValidationError("home_and_away_required")

    if home_team == away_team:
        raise ValidationError("teams_must_be_different")

    if len(home_team) > 80 or len(away_team) > 80:
        raise ValidationError("team_name_too_long")

    return PredictMatchInput(home_team=home_team, away_team=away_team)


def validate_simulate_season_payload(payload: dict) -> SimulateSeasonInput:
    if not isinstance(payload, dict):
        raise ValidationError("invalid_payload")

    try:
        iterations = int(payload.get("iterations", 1000))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("invalid_iterations") from exc

    if iterations < 100 or iterations > 10000:
        raise ValidationError("iterations_out_of_range")

    try:
        volatility = float(payload.get("volatility", 1.0))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("invalid_volatility") from exc

    if volatility < 0.6 or volatility > 1.8:
        raise ValidationError("volatility_out_of_range")

    cutoff_date = payload.get("cutoffDate")
    normalized_cutoff = None
    if cutoff_date is not None and str(cutoff_date).strip() != "":
        value = str(cutoff_date).strip()
        try:
            date.fromisoformat(value)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("invalid_cutoff_date") from exc
        normalized_cutoff = value

    return SimulateSeasonInput(
        iterations=iterations,
        volatility=volatility,
        cutoff_date=normalized_cutoff,
    )


def validate_team_availability_payload(payload: dict) -> list[TeamAvailabilityRowInput]:
    if not isinstance(payload, dict):
        raise ValidationError("invalid_payload")

    rows = payload.get("teams")
    if not isinstance(rows, list):
        raise ValidationError("invalid_team_availability_payload")

    validated: list[TeamAvailabilityRowInput] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValidationError("invalid_team_availability_row")

        team = str(row.get("team", "")).strip()
        if not team:
            raise ValidationError("team_required")
        if len(team) > 80:
            raise ValidationError("team_name_too_long")

        try:
            injured = float(row.get("injured", 0))
            suspended = float(row.get("suspended", 0))
            key_impact = float(row.get("key_impact", 0))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("invalid_team_availability_values") from exc

        if injured < 0 or suspended < 0:
            raise ValidationError("negative_availability_not_allowed")
        if key_impact < 0 or key_impact > 1:
            raise ValidationError("key_impact_out_of_range")

        validated.append(
            TeamAvailabilityRowInput(
                team=team,
                injured=injured,
                suspended=suspended,
                key_impact=key_impact,
            )
        )

    return validated


def validate_team_news_payload(payload: dict) -> list[TeamNewsRowInput]:
    if not isinstance(payload, dict):
        raise ValidationError("invalid_payload")

    rows = payload.get("players")
    if not isinstance(rows, list):
        raise ValidationError("invalid_team_news_payload")

    valid_statuses = {
        "injured",
        "out",
        "doubtful",
        "suspended",
        "banned",
        "available",
        "fit",
    }

    validated: list[TeamNewsRowInput] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValidationError("invalid_team_news_row")

        team = str(row.get("team", "")).strip()
        player = str(row.get("player", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        position = str(row.get("position", "")).strip()
        expected_return = str(row.get("expected_return", "")).strip()
        source = str(row.get("source", "")).strip()
        updated_at = str(row.get("updated_at", "")).strip()

        if not team:
            raise ValidationError("team_required")
        if not player:
            raise ValidationError("player_required")
        if len(team) > 80:
            raise ValidationError("team_name_too_long")
        if len(player) > 120:
            raise ValidationError("player_name_too_long")
        if status not in valid_statuses:
            raise ValidationError("invalid_team_news_status")

        try:
            importance = float(row.get("importance", 0))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("invalid_team_news_values") from exc

        if importance < 0 or importance > 1:
            raise ValidationError("importance_out_of_range")

        validated.append(
            TeamNewsRowInput(
                team=team,
                player=player,
                status=status,
                position=position,
                importance=importance,
                expected_return=expected_return,
                source=source,
                updated_at=updated_at,
            )
        )

    return validated
