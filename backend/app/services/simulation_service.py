from __future__ import annotations

import math
import random
from collections import defaultdict

import pandas as pd

POINTS_MAP = {"W": 3, "D": 1, "L": 0}
MAX_MATCHES = 38


def _full_standings(df: pd.DataFrame) -> list[dict]:
    standings = (
        df.assign(points=df["result"].map(POINTS_MAP).fillna(0))
        .groupby("team", as_index=False)
        .agg(played=("team", "size"), points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    )
    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings.sort_values(
        ["points", "gd", "gf", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    standings["position"] = standings.index + 1

    output = []
    for row in standings.itertuples(index=False):
        output.append(
            {
                "position": int(row.position),
                "name": row.team,
                "played": int(row.played),
                "points": int(row.points),
            }
        )
    return output


def _team_form_index(season_df: pd.DataFrame) -> dict[str, float]:
    form = {}
    for team, group in season_df.sort_values("date").groupby("team"):
        recent = group.tail(5)
        if recent.empty:
            form[team] = 1.5
            continue
        avg = recent["result"].map(POINTS_MAP).fillna(1).mean()
        form[team] = float(avg)
    return form


def _team_strength_index(standings: list[dict], form_index: dict[str, float]) -> dict[str, float]:
    base_ppg = {row["name"]: row["points"] / max(row["played"], 1) for row in standings}
    all_ppg = list(base_ppg.values())
    mean_ppg = sum(all_ppg) / max(len(all_ppg), 1)
    variance = sum((value - mean_ppg) ** 2 for value in all_ppg) / max(len(all_ppg), 1)
    std_ppg = math.sqrt(variance) if variance > 1e-9 else 1.0

    strength = {}
    for team, ppg in base_ppg.items():
        form_ppg = form_index.get(team, ppg)
        blended = (0.72 * ppg) + (0.28 * form_ppg)
        strength[team] = (blended - mean_ppg) / std_ppg
    return strength


def _match_probabilities(team_strength: float, opponent_strength: float, home_advantage: float) -> tuple[float, float, float]:
    delta = (team_strength - opponent_strength) + home_advantage

    draw_prob = max(0.16, 0.27 - abs(delta) * 0.045)
    win_prob = 1.0 / (1.0 + math.exp(-1.22 * delta))
    lose_prob = 1.0 - win_prob

    win_prob *= (1.0 - draw_prob)
    lose_prob *= (1.0 - draw_prob)

    total = win_prob + draw_prob + lose_prob
    return win_prob / total, draw_prob / total, lose_prob / total


def _simulate_team_points(
    team_name: str,
    base_points: int,
    remaining_matches: int,
    effective_strength: dict[str, float],
) -> int:
    total_points = int(base_points)
    teams = [name for name in effective_strength if name != team_name]

    for _ in range(remaining_matches):
        if not teams:
            break

        opponent = random.choice(teams)
        team_power = effective_strength.get(team_name, 0.0) + random.gauss(0.0, 0.16)
        opponent_power = effective_strength.get(opponent, 0.0) + random.gauss(0.0, 0.16)
        home_advantage = random.choice([0.15, -0.15])
        win_prob, draw_prob, _ = _match_probabilities(team_power, opponent_power, home_advantage)

        roll = random.random()
        if roll < win_prob:
            total_points += 3
        elif roll < win_prob + draw_prob:
            total_points += 1

    if remaining_matches > 0:
        volatility = max(0.8, math.sqrt(remaining_matches) * 1.1)
        total_points += int(round(random.gauss(0.0, volatility)))

    return max(total_points, 0)


def run_season_simulation(
    season_df: pd.DataFrame,
    iterations: int = 1000,
    cutoff_date: str | None = None,
    volatility: float = 1.0,
) -> dict:
    iterations = int(max(100, min(iterations, 10000)))
    volatility = float(max(0.6, min(volatility, 1.8)))

    working_df = season_df.copy()
    applied_cutoff = None
    if cutoff_date:
        try:
            cutoff_ts = pd.to_datetime(cutoff_date)
            filtered = working_df[working_df["date"] <= cutoff_ts]
            if not filtered.empty:
                working_df = filtered
                applied_cutoff = cutoff_ts.date().isoformat()
        except Exception:
            applied_cutoff = None

    standings = _full_standings(working_df)
    team_form = _team_form_index(working_df)
    team_strength = _team_strength_index(standings, team_form)

    position_counts: dict[str, list[int]] = {}
    point_totals: defaultdict[str, int] = defaultdict(int)
    champion_counts: defaultdict[str, int] = defaultdict(int)
    top4_counts: defaultdict[str, int] = defaultdict(int)
    relegation_counts: defaultdict[str, int] = defaultdict(int)

    for row in standings:
        position_counts[row["name"]] = [0] * len(standings)

    for _ in range(iterations):
        iteration_strength = {
            team: value + random.gauss(0.0, 0.42 * volatility)
            for team, value in team_strength.items()
        }
        simulated = []
        for row in standings:
            team = row["name"]
            played = row["played"]
            current_points = row["points"]

            remaining = max(0, MAX_MATCHES - played)
            final_points = _simulate_team_points(
                team_name=team,
                base_points=current_points,
                remaining_matches=remaining,
                effective_strength=iteration_strength,
            )
            simulated.append((team, final_points))

        simulated.sort(key=lambda item: (-item[1], item[0]))

        for idx, (team, pts) in enumerate(simulated):
            position_counts[team][idx] += 1
            point_totals[team] += pts
            if idx == 0:
                champion_counts[team] += 1
            if idx < 4:
                top4_counts[team] += 1
            if idx >= len(simulated) - 3:
                relegation_counts[team] += 1

    projection = []
    for team, pos_counts in position_counts.items():
        avg_points = point_totals[team] / iterations
        title_prob = (champion_counts[team] / iterations) * 100
        top4_prob = (top4_counts[team] / iterations) * 100
        relegation_prob = (relegation_counts[team] / iterations) * 100
        most_likely_position = int(max(range(len(pos_counts)), key=lambda i: pos_counts[i]) + 1)

        projection.append(
            {
                "team": team,
                "avgPoints": round(avg_points, 1),
                "titleProb": round(title_prob, 1),
                "top4Prob": round(top4_prob, 1),
                "relegationProb": round(relegation_prob, 1),
                "mostLikelyPosition": most_likely_position,
            }
        )

    projection.sort(key=lambda row: (-row["avgPoints"], row["team"]))

    return {
        "iterations": iterations,
        "volatility": volatility,
        "cutoffDate": applied_cutoff,
        "teams": projection,
    }
