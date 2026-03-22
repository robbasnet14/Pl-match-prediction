from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    expires_at: float
    value: dict


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> dict | None:
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: dict, ttl_seconds: float) -> None:
        self._store[key] = CacheEntry(expires_at=time.time() + ttl_seconds, value=value)

    def clear(self) -> None:
        self._store.clear()
