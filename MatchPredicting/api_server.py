from __future__ import annotations

import sys
from pathlib import Path

# Compatibility launcher: keeps `python MatchPredicting/api_server.py` working
# while using the new structured backend package.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.server import run_server


if __name__ == "__main__":
    run_server()
