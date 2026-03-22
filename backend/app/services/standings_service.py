from __future__ import annotations

import pandas as pd


POINTS_MAP = {"W": 3, "D": 1, "L": 0}


def build_standings(df: pd.DataFrame, limit: int = 20) -> list[dict]:
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
                "change": "same",
                "changeLabel": "-",
            }
        )

    return output[:limit]
