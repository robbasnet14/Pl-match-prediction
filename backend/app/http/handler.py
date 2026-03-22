from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from ..data.repository import latest_season_frame, load_matches, load_team_availability_frame, load_team_news_frame
from ..data.repository import save_team_availability_frame, save_team_news_frame
from ..services.dashboard_service import build_dashboard_payload
from ..services.backtest_service import build_model_backtest
from ..services.data_status_service import build_data_status
from ..services.live_data_service import get_fixtures_with_source, get_standings_with_source
from ..services import prediction_service
from ..services.prediction_service import get_available_teams, get_model_info, predict_single_match
from ..services.team_news_sync_service import sync_team_news
from ..services.simulation_service import run_season_simulation
from .cache import TTLCache
from .validators import (
    ValidationError,
    validate_team_availability_payload,
    validate_team_news_payload,
    validate_predict_match_payload,
    validate_simulate_season_payload,
)

logger = logging.getLogger("pl_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


RESPONSE_CACHE = TTLCache()


class ApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict, request_id: str | None = None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, request_id: str):
        self._send_json(status, {"error": code, "requestId": request_id}, request_id=request_id)

    def do_OPTIONS(self):
        request_id = str(uuid.uuid4())
        self._send_json(200, {"ok": True, "requestId": request_id}, request_id=request_id)

    def do_GET(self):
        request_id = str(uuid.uuid4())
        path = urlparse(self.path).path
        logger.info("GET %s request_id=%s", path, request_id)
        try:
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "requestId": request_id,
                    },
                    request_id=request_id,
                )
                return

            if path == "/dashboard":
                cache_key = "GET:/dashboard"
                payload = RESPONSE_CACHE.get(cache_key)
                if payload is None:
                    payload = build_dashboard_payload()
                    RESPONSE_CACHE.set(cache_key, payload, ttl_seconds=10)
                response_payload = dict(payload)
                response_payload["requestId"] = request_id
                self._send_json(200, response_payload, request_id=request_id)
                return

            if path == "/fixtures/upcoming":
                season_df = latest_season_frame(load_matches())
                fixtures, source = get_fixtures_with_source(season_df)
                self._send_json(200, {"fixtures": fixtures, "source": source, "requestId": request_id}, request_id=request_id)
                return

            if path == "/teams":
                teams = get_available_teams(load_matches())
                self._send_json(200, {"teams": teams, "requestId": request_id}, request_id=request_id)
                return

            if path == "/team-availability":
                availability = load_team_availability_frame()
                rows = [] if availability is None else availability.to_dict(orient="records")
                self._send_json(200, {"teams": rows, "requestId": request_id}, request_id=request_id)
                return

            if path == "/team-news":
                team_news = load_team_news_frame()
                rows = [] if team_news is None else team_news.to_dict(orient="records")
                self._send_json(200, {"players": rows, "requestId": request_id}, request_id=request_id)
                return

            if path == "/data-status":
                status = build_data_status(load_matches())
                status["requestId"] = request_id
                self._send_json(200, status, request_id=request_id)
                return

            if path == "/model-backtest":
                result = build_model_backtest(load_matches())
                result["requestId"] = request_id
                self._send_json(200, result, request_id=request_id)
                return

            if path == "/model-info":
                result = get_model_info(load_matches())
                result["requestId"] = request_id
                self._send_json(200, result, request_id=request_id)
                return

            self._error(404, "not_found", request_id)
        except Exception:
            logger.exception("GET failure path=%s request_id=%s", path, request_id)
            self._error(500, "internal_error", request_id)

    def do_POST(self):
        request_id = str(uuid.uuid4())
        path = urlparse(self.path).path
        logger.info("POST %s request_id=%s", path, request_id)

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._error(400, "invalid_json", request_id)
            return

        try:
            if path == "/predict-match":
                validated = validate_predict_match_payload(payload)

                all_matches = load_matches()
                season_df = latest_season_frame(all_matches)
                standings, standings_source = get_standings_with_source(season_df)
                prediction = predict_single_match(
                    all_matches,
                    standings,
                    validated.home_team,
                    validated.away_team,
                )
                self._send_json(
                    200,
                    {
                        "prediction": prediction,
                        "standingsSource": standings_source,
                        "requestId": request_id,
                    },
                    request_id=request_id,
                )
                return

            if path == "/simulate-season":
                validated = validate_simulate_season_payload(payload)
                all_matches = load_matches()
                data_status = build_data_status(all_matches)
                if not data_status.get("isComplete", False):
                    self._send_json(
                        422,
                        {
                            "error": "dataset_incomplete",
                            "details": data_status,
                            "requestId": request_id,
                        },
                        request_id=request_id,
                    )
                    return

                season_df = latest_season_frame(all_matches)
                cache_key = (
                    f"POST:/simulate-season:"
                    f"i={validated.iterations}:v={validated.volatility}:c={validated.cutoff_date or 'none'}"
                )
                result = RESPONSE_CACHE.get(cache_key)
                if result is None:
                    result = run_season_simulation(
                        season_df,
                        iterations=validated.iterations,
                        cutoff_date=validated.cutoff_date,
                        volatility=validated.volatility,
                    )
                    RESPONSE_CACHE.set(cache_key, result, ttl_seconds=30)
                response_payload = dict(result)
                response_payload["requestId"] = request_id
                self._send_json(200, response_payload, request_id=request_id)
                return

            if path == "/team-availability":
                validated_rows = validate_team_availability_payload(payload)
                if not validated_rows:
                    self._error(400, "team_availability_empty", request_id)
                    return

                rows = [
                    {
                        "team": row.team,
                        "injured": row.injured,
                        "suspended": row.suspended,
                        "key_impact": row.key_impact,
                    }
                    for row in validated_rows
                ]
                frame = load_team_availability_frame()
                if frame is not None:
                    # Preserve ordering of existing file when possible.
                    current_order = frame["team"].astype(str).tolist()
                    order_map = {name: idx for idx, name in enumerate(current_order)}
                    rows.sort(key=lambda row: order_map.get(row["team"], 10**6))

                import pandas as pd  # local import to keep module load minimal

                save_team_availability_frame(pd.DataFrame(rows))
                prediction_service.reset_model_cache()
                RESPONSE_CACHE.clear()
                self._send_json(200, {"ok": True, "updated": len(rows), "requestId": request_id}, request_id=request_id)
                return

            if path == "/team-news":
                validated_rows = validate_team_news_payload(payload)
                if not validated_rows:
                    self._error(400, "team_news_empty", request_id)
                    return

                rows = [
                    {
                        "team": row.team,
                        "player": row.player,
                        "status": row.status,
                        "position": row.position,
                        "importance": row.importance,
                        "expected_return": row.expected_return,
                        "source": row.source,
                        "updated_at": row.updated_at,
                    }
                    for row in validated_rows
                ]
                rows.sort(key=lambda row: (row["team"], row["player"]))

                import pandas as pd  # local import to keep module load minimal

                save_team_news_frame(pd.DataFrame(rows))
                prediction_service.reset_model_cache()
                RESPONSE_CACHE.clear()
                self._send_json(200, {"ok": True, "updated": len(rows), "requestId": request_id}, request_id=request_id)
                return

            if path == "/team-news/sync":
                min_rows = int(payload.get("minRows", 5))
                dry_run = bool(payload.get("dryRun", False))
                result = sync_team_news(min_rows=min_rows, dry_run=dry_run)
                prediction_service.reset_model_cache()
                RESPONSE_CACHE.clear()
                result["requestId"] = request_id
                self._send_json(200, result, request_id=request_id)
                return

            self._error(404, "not_found", request_id)
        except ValidationError as exc:
            self._error(400, str(exc), request_id)
        except ValueError:
            self._error(400, "invalid_parameters", request_id)
        except Exception:
            logger.exception("POST failure path=%s request_id=%s", path, request_id)
            self._error(500, "internal_error", request_id)
