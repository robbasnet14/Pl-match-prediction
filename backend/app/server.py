from __future__ import annotations

from http.server import HTTPServer

from .config import API_HOST, API_PORT
from .http.handler import ApiHandler


def run_server() -> None:
    server = HTTPServer((API_HOST, API_PORT), ApiHandler)
    print(f"API running at http://{API_HOST}:{API_PORT}")
    server.serve_forever()
