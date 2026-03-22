from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
RESULT_MAP_HOME = {"H": "W", "D": "D", "A": "L"}
RESULT_MAP_AWAY = {"H": "L", "D": "D", "A": "W"}


def _safe_float(value, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _pick_numeric(row, candidates: list[str]) -> float:
    for key in candidates:
        if not hasattr(row, key):
            continue
        value = _safe_float(getattr(row, key), default=float("nan"))
        if not pd.isna(value):
            return float(value)
    return 0.0


def convert_match_rows(raw_df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    rows = []
    for match in df.itertuples(index=False):
        match_date = getattr(match, "Date")
        time_value = getattr(match, "Time", "15:00")
        if pd.isna(time_value) or str(time_value).strip() == "":
            time_value = "15:00"

        home_team = str(getattr(match, "HomeTeam"))
        away_team = str(getattr(match, "AwayTeam"))
        home_goals = int(getattr(match, "FTHG"))
        away_goals = int(getattr(match, "FTAG"))
        result_code = str(getattr(match, "FTR"))
        referee = str(getattr(match, "Referee", "") or "").strip()

        # Optional performance fields (present in some providers/seasons).
        home_xg = _pick_numeric(match, ["xG_H", "HXG", "HxG", "home_xg"])
        away_xg = _pick_numeric(match, ["xG_A", "AXG", "AxG", "away_xg"])
        home_shots = _pick_numeric(match, ["HS"])
        away_shots = _pick_numeric(match, ["AS"])
        home_sot = _pick_numeric(match, ["HST"])
        away_sot = _pick_numeric(match, ["AST"])

        # Pre-match odds priors. Use bookmaker close when present, otherwise average.
        odds_home = _pick_numeric(match, ["B365H", "PSH", "WHH", "VCH", "AvgH"])
        odds_draw = _pick_numeric(match, ["B365D", "PSD", "WHD", "VCD", "AvgD"])
        odds_away = _pick_numeric(match, ["B365A", "PSA", "WHA", "VCA", "AvgA"])

        rows.append(
            {
                "date": match_date.date().isoformat(),
                "time": str(time_value),
                "comp": "Premier League",
                "round": "Matchweek",
                "day": match_date.strftime("%a"),
                "venue": "Home",
                "result": RESULT_MAP_HOME.get(result_code, "D"),
                "gf": home_goals,
                "ga": away_goals,
                "opponent": away_team,
                "season": season_year,
                "team": home_team,
                "xg": home_xg,
                "xga": away_xg,
                "sh": home_shots,
                "sot": home_sot,
                "referee": referee,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }
        )

        rows.append(
            {
                "date": match_date.date().isoformat(),
                "time": str(time_value),
                "comp": "Premier League",
                "round": "Matchweek",
                "day": match_date.strftime("%a"),
                "venue": "Away",
                "result": RESULT_MAP_AWAY.get(result_code, "D"),
                "gf": away_goals,
                "ga": home_goals,
                "opponent": home_team,
                "season": season_year,
                "team": away_team,
                "xg": away_xg,
                "xga": home_xg,
                "sh": away_shots,
                "sot": away_sot,
                "referee": referee,
                "odds_home": odds_away,
                "odds_draw": odds_draw,
                "odds_away": odds_home,
            }
        )

    return pd.DataFrame(rows)


def _parse_season_code(season_code: str) -> tuple[int, int]:
    value = season_code.strip()
    if len(value) != 4 or not value.isdigit():
        raise ValueError("season-code must be a 4-digit code like 2526")

    start = int(value[:2])
    end = int(value[2:])
    if end != ((start + 1) % 100):
        raise ValueError("season-code must represent consecutive years, e.g. 2526")
    return start, end


def _season_code_for_offset(base_code: str, offset: int) -> str:
    start, end = _parse_season_code(base_code)
    return f"{(start - offset) % 100:02d}{(end - offset) % 100:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import EPL season CSV and convert to app schema")
    parser.add_argument(
        "--season-code",
        default="2526",
        help="football-data season code, e.g. 2526 for 2025/26",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=1,
        help="number of seasons to import ending at season-code (1 to 10)",
    )
    parser.add_argument(
        "--output",
        default="backend/app/data/matches_current.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    season_code = args.season_code.strip()
    _parse_season_code(season_code)
    history_years = int(args.history_years)
    if history_years < 1 or history_years > 10:
        raise ValueError("history-years must be between 1 and 10")

    frames: list[pd.DataFrame] = []
    for offset in range(history_years - 1, -1, -1):
        code = _season_code_for_offset(season_code, offset)
        season_year = 2000 + int(code[2:])
        url = BASE_URL_TEMPLATE.format(season_code=code)
        print(f"Downloading {url}")
        raw_df = pd.read_csv(url)
        frames.append(convert_match_rows(raw_df, season_year=season_year))

    converted = pd.concat(frames, ignore_index=True)
    converted = converted.sort_values(["date", "time", "team"]).reset_index(drop=True)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output_path, index=False)

    team_count = int(converted["team"].nunique())
    season_values = sorted(int(x) for x in converted["season"].dropna().unique())
    print(f"Wrote {len(converted)} rows to {output_path}")
    print(f"Seasons in output: {season_values}")
    print(f"Unique teams in output: {team_count}")


if __name__ == "__main__":
    main()
