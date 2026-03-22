from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from http.server import HTTPServer

from backend.app.data import repository
from backend.app.http.handler import ApiHandler, RESPONSE_CACHE
from backend.app.services import prediction_service
from backend.app.services.data_status_service import build_data_status


class ApiServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_matches_path = repository.MATCHES_PATH
        cls._original_team_availability_path = repository.TEAM_AVAILABILITY_PATH
        cls._original_team_news_path = repository.TEAM_NEWS_PATH
        cls._original_rf_trees = prediction_service.MODEL_RF_TREES
        cls._original_rf_min_split = prediction_service.MODEL_RF_MIN_SAMPLES_SPLIT
        cls._original_rf_max_depth = prediction_service.MODEL_RF_MAX_DEPTH
        cls._original_form_window = prediction_service.MODEL_FORM_WINDOW
        cls._original_h2h_window = prediction_service.MODEL_H2H_WINDOW
        cls._temp_dir = tempfile.TemporaryDirectory()

        root = Path(__file__).resolve().parents[2]
        preferred = root / "backend" / "app" / "data" / "matches_current.csv"
        fallback = root / "MatchPredicting" / "matches.csv"
        repository.MATCHES_PATH = preferred if preferred.exists() else fallback
        repository.TEAM_AVAILABILITY_PATH = Path(cls._temp_dir.name) / "team_availability.csv"
        repository.TEAM_NEWS_PATH = Path(cls._temp_dir.name) / "team_news.csv"

        # Keep endpoint tests fast/stable while preserving production defaults in .env.
        prediction_service.MODEL_RF_TREES = 300
        prediction_service.MODEL_RF_MIN_SAMPLES_SPLIT = 2
        prediction_service.MODEL_RF_MAX_DEPTH = 8
        prediction_service.MODEL_FORM_WINDOW = 5
        prediction_service.MODEL_H2H_WINDOW = 5
        prediction_service.reset_model_cache(delete_artifact=True)
        RESPONSE_CACHE.clear()

        cls._server = HTTPServer(("127.0.0.1", 0), ApiHandler)
        cls._port = cls._server.server_address[1]
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

        cls._data_status = build_data_status(repository.load_matches())

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._server.server_close()
        cls._thread.join(timeout=2)
        repository.MATCHES_PATH = cls._original_matches_path
        repository.TEAM_AVAILABILITY_PATH = cls._original_team_availability_path
        repository.TEAM_NEWS_PATH = cls._original_team_news_path
        cls._temp_dir.cleanup()
        prediction_service.MODEL_RF_TREES = cls._original_rf_trees
        prediction_service.MODEL_RF_MIN_SAMPLES_SPLIT = cls._original_rf_min_split
        prediction_service.MODEL_RF_MAX_DEPTH = cls._original_rf_max_depth
        prediction_service.MODEL_FORM_WINDOW = cls._original_form_window
        prediction_service.MODEL_H2H_WINDOW = cls._original_h2h_window
        prediction_service.reset_model_cache(delete_artifact=True)
        RESPONSE_CACHE.clear()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._port}{path}"

    def _get_json(self, path: str) -> tuple[int, dict]:
        request = Request(self._url(path), method="GET")
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)

    def _post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        request = Request(
            self._url(path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8")
                return response.status, json.loads(body)
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body)

    def test_dashboard_endpoint(self):
        status, payload = self._get_json("/dashboard")
        self.assertEqual(status, 200)
        for key in ["summary", "fixtures", "standings", "predictions", "modelMetrics", "dataStatus", "requestId"]:
            self.assertIn(key, payload)

    def test_data_status_endpoint(self):
        status, payload = self._get_json("/data-status")
        self.assertEqual(status, 200)
        self.assertIn("isComplete", payload)
        self.assertIn("latestSeasonTeams", payload)
        self.assertIn("seasonTeamCounts", payload)
        self.assertIn("requestId", payload)

    def test_model_backtest_endpoint(self):
        status, payload = self._get_json("/model-backtest")
        self.assertEqual(status, 200)
        self.assertIn("available", payload)
        if payload.get("available"):
            self.assertIn("accuracyPct", payload)
            self.assertIn("brierScore", payload)
            self.assertIn("scorelineExactPct", payload)
        self.assertIn("requestId", payload)

    def test_model_info_endpoint(self):
        status, payload = self._get_json("/model-info")
        self.assertEqual(status, 200)
        self.assertEqual(payload.get("available"), True)
        self.assertIn("source", payload)
        self.assertIn("metrics", payload)
        self.assertIn("artifact", payload)
        self.assertIn("requestId", payload)

    def test_predict_match_endpoint(self):
        status, teams_payload = self._get_json("/teams")
        self.assertEqual(status, 200)
        teams = teams_payload.get("teams", [])
        self.assertGreaterEqual(len(teams), 2)

        status, payload = self._post_json(
            "/predict-match",
            {"homeTeam": teams[0], "awayTeam": teams[1]},
        )
        self.assertEqual(status, 200)
        self.assertIn("prediction", payload)
        self.assertIn("explanation", payload["prediction"])
        self.assertIn("homeTopUnavailable", payload["prediction"]["explanation"])
        self.assertIn("awayTopUnavailable", payload["prediction"]["explanation"])
        self.assertIn("requestId", payload)

    def test_predict_match_validation(self):
        status, payload = self._post_json(
            "/predict-match",
            {"homeTeam": "Arsenal", "awayTeam": "Arsenal"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload.get("error"), "teams_must_be_different")

    def test_simulate_season_endpoint(self):
        status, payload = self._post_json(
            "/simulate-season",
            {"iterations": 300, "volatility": 1.2},
        )

        if self._data_status.get("isComplete"):
            self.assertEqual(status, 200)
            self.assertIn("teams", payload)
            self.assertIn("requestId", payload)
        else:
            self.assertEqual(status, 422)
            self.assertEqual(payload.get("error"), "dataset_incomplete")

    def test_simulate_season_validation(self):
        status, payload = self._post_json(
            "/simulate-season",
            {"iterations": 25, "volatility": 2.5},
        )
        self.assertEqual(status, 400)
        self.assertIn(payload.get("error"), {"iterations_out_of_range", "volatility_out_of_range"})

    def test_team_news_endpoint(self):
        status, initial = self._get_json("/team-news")
        self.assertEqual(status, 200)
        self.assertEqual(initial.get("players"), [])

        status, saved = self._post_json(
            "/team-news",
            {
                "players": [
                    {
                        "team": "Arsenal",
                        "player": "Bukayo Saka",
                        "status": "injured",
                        "position": "RW",
                        "importance": 0.95,
                        "expected_return": "2026-03-12",
                        "source": "club",
                        "updated_at": "2026-02-23T10:00:00Z",
                    },
                    {
                        "team": "Arsenal",
                        "player": "Declan Rice",
                        "status": "suspended",
                        "position": "CM",
                        "importance": 0.9,
                        "expected_return": "2026-03-01",
                        "source": "league",
                        "updated_at": "2026-02-23T10:00:00Z",
                    },
                ]
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(saved.get("ok"), True)
        self.assertEqual(saved.get("updated"), 2)

        status, payload = self._get_json("/team-news")
        self.assertEqual(status, 200)
        players = payload.get("players", [])
        self.assertEqual(len(players), 2)
        self.assertEqual(players[0]["team"], "Arsenal")
        self.assertIn(players[0]["status"], {"injured", "suspended"})

    def test_team_news_sync_endpoint(self):
        with patch(
            "backend.app.http.handler.sync_team_news",
            return_value={
                "ok": True,
                "updated": 11,
                "dryRun": True,
                "minimumRequired": 3,
                "minimumSatisfied": True,
            },
        ):
            status, payload = self._post_json(
                "/team-news/sync",
                {"dryRun": True, "minRows": 3},
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload.get("ok"), True)
        self.assertEqual(payload.get("updated"), 11)
        self.assertEqual(payload.get("dryRun"), True)
        self.assertEqual(payload.get("minimumRequired"), 3)
        self.assertEqual(payload.get("minimumSatisfied"), True)


if __name__ == "__main__":
    unittest.main()
