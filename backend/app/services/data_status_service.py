from __future__ import annotations

import pandas as pd


def build_data_status(all_matches: pd.DataFrame) -> dict:
    if all_matches.empty:
        return {
            "isComplete": False,
            "latestSeason": None,
            "latestSeasonTeams": 0,
            "latestMatchDate": None,
            "seasonTeamCounts": [],
            "issues": ["no_data_loaded"],
        }

    team_counts = (
        all_matches.groupby("season")
        .agg(teamCount=("team", "nunique"), matchCount=("team", "size"))
        .reset_index()
        .sort_values("season")
    )

    latest_season = int(team_counts["season"].max())
    latest_row = team_counts[team_counts["season"] == latest_season].iloc[0]
    latest_team_count = int(latest_row["teamCount"])
    latest_match_date = all_matches["date"].max()

    issues = []
    if latest_team_count < 20:
        issues.append("latest_season_has_fewer_than_20_teams")

    season_team_counts = [
        {
            "season": int(row.season),
            "teamCount": int(row.teamCount),
            "matchCount": int(row.matchCount),
        }
        for row in team_counts.itertuples(index=False)
    ]

    return {
        "isComplete": latest_team_count >= 20,
        "latestSeason": latest_season,
        "latestSeasonTeams": latest_team_count,
        "latestMatchDate": latest_match_date.date().isoformat() if pd.notna(latest_match_date) else None,
        "seasonTeamCounts": season_team_counts,
        "issues": issues,
    }
