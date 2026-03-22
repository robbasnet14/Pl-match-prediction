from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.run import load_env_file
from backend.app.services.team_news_sync_service import sync_team_news


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync real team news into backend/app/data/team_news.csv")
    parser.add_argument("--min-rows", type=int, default=5, help="minimum accepted player rows before writing")
    parser.add_argument("--dry-run", action="store_true", help="fetch and validate without writing file")
    args = parser.parse_args()

    load_env_file(ROOT_DIR / "backend" / ".env")
    result = sync_team_news(min_rows=args.min_rows, dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
