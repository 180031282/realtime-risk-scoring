"""FastAPI endpoint tests via TestClient.

conftest.py sets STATS_DB_PATH=":memory:" before app.api is imported here,
so these tests never touch a real stats.db and never require Kafka.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # training/model.pkl is committed to the repo, so it should load.
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0"


def test_stats_shape_when_empty():
    # Use a dedicated store on app.state so this test doesn't depend on
    # ordering relative to other tests that record events.
    from app.scoring import StatsStore

    app.state.stats_store = StatsStore(db_path=":memory:")

    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_scored"] == 0
    assert body["flag_rate"] == 0.0
    assert body["throughput_per_sec"] == 0.0
    assert isinstance(body["score_histogram"], dict)
    assert len(body["score_histogram"]) == 10


def test_stats_reflects_recorded_events():
    from app.scoring import StatsStore

    store = StatsStore(db_path=":memory:")
    store.record(latency_ms=5.0, risk_score=0.1, flagged=False)
    store.record(latency_ms=15.0, risk_score=0.95, flagged=True)
    app.state.stats_store = store

    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_scored"] == 2
    assert body["flagged_count"] == 1
    assert body["flag_rate"] == 0.5
    assert body["last_updated"] is not None


def test_dashboard_serves_html_with_chartjs():
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "chart.js" in resp.text.lower() or "chart.umd" in resp.text.lower()
    assert "/stats" in resp.text


def test_root_lists_service_info():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "realtime-risk-scoring-api"
