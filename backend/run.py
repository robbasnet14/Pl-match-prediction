from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[7:].strip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def load_env_file(env_path: Path | None = None) -> None:
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        # Keep explicit shell overrides, but allow .env to fill empty shell vars.
        if key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = value


if __name__ == "__main__":
    load_env_file()
    from app.server import run_server

    run_server()
