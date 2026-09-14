"""FastAPI service: live stats API + single-file HTML/Chart.js dashboard.

This service never talks to Kafka directly. It only reads:
  * the trained model file (just to report whether it's loaded / its
    version on /health), and
  * the shared SQLite stats store the consumer writes to,

so it can run and be tested completely independently of a live broker.

Endpoints:
    GET /health     liveness + whether the model is loaded
    GET /stats      current rolling-window aggregate stats
    GET /dashboard  minimal HTML page that polls /stats and renders charts
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.models import HealthResponse, StatsResponse
from app.scoring import DEFAULT_MODEL_PATH, RiskScorer, StatsStore

STATS_DB_PATH = os.environ.get("STATS_DB_PATH", "stats.db")
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
STATS_WINDOW_SECONDS = float(os.environ.get("STATS_WINDOW_SECONDS", "300"))

app = FastAPI(
    title="Real-Time Transaction Risk Scoring API",
    description="Live stats for a synthetic-data, event-driven fraud/risk scoring demo.",
    version="1.0.0",
)

# Module-level singletons, lazily populated on startup. Kept swappable via
# app.state so tests can point at an isolated stats DB / model without
# touching the real filesystem paths.
app.state.stats_store = StatsStore(db_path=STATS_DB_PATH, window_seconds=STATS_WINDOW_SECONDS)
app.state.scorer = RiskScorer(model_path=MODEL_PATH)
try:
    app.state.scorer.load()
except FileNotFoundError:
    # The API can start (and serve /health as "model not loaded") even
    # before training/train_model.py has produced a model.pkl -- useful for
    # docker-compose bring-up ordering and for tests that don't need the
    # model.
    pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    scorer: RiskScorer = app.state.scorer
    return HealthResponse(
        status="ok",
        model_loaded=scorer.is_loaded,
        model_version=scorer.model_version if scorer.is_loaded else None,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    store: StatsStore = app.state.stats_store
    return StatsResponse(**store.get_stats())


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Real-Time Risk Scoring Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 24px; background: #0b0f14; color: #e6edf3;
  }
  h1 { font-size: 20px; margin-bottom: 4px; }
  p.sub { color: #8b949e; margin-top: 0; font-size: 13px; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 20px 0;
  }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 16px;
  }
  .card .label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 26px; font-weight: 600; margin-top: 4px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  @media (max-width: 720px) { .charts { grid-template-columns: 1fr; } }
  canvas { max-height: 280px; }
</style>
</head>
<body>
  <h1>Real-Time Transaction Risk Scoring</h1>
  <p class="sub">Live view of a synthetic transaction stream, polling <code>/stats</code> every 2s.</p>

  <div class="grid">
    <div class="card"><div class="label">Total Scored</div><div class="value" id="stat-total">-</div></div>
    <div class="card"><div class="label">Flag Rate</div><div class="value" id="stat-flagrate">-</div></div>
    <div class="card"><div class="label">Throughput /s</div><div class="value" id="stat-throughput">-</div></div>
    <div class="card"><div class="label">p95 Latency (ms)</div><div class="value" id="stat-p95">-</div></div>
  </div>

  <div class="charts">
    <div class="chart-box"><canvas id="scoreHistChart"></canvas></div>
    <div class="chart-box"><canvas id="latencyChart"></canvas></div>
  </div>

<script>
const scoreCtx = document.getElementById('scoreHistChart').getContext('2d');
const latencyCtx = document.getElementById('latencyChart').getContext('2d');

const scoreChart = new Chart(scoreCtx, {
  type: 'bar',
  data: { labels: [], datasets: [{ label: 'Score distribution', data: [], backgroundColor: '#58a6ff' }] },
  options: {
    plugins: { title: { display: true, text: 'Risk Score Distribution', color: '#e6edf3' }, legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' }, beginAtZero: true }
    }
  }
});

const latencyChart = new Chart(latencyCtx, {
  type: 'bar',
  data: { labels: ['p50', 'p95', 'p99'], datasets: [{ label: 'Latency (ms)', data: [0, 0, 0], backgroundColor: '#3fb950' }] },
  options: {
    plugins: { title: { display: true, text: 'Scoring Latency', color: '#e6edf3' }, legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' }, beginAtZero: true }
    }
  }
});

async function refresh() {
  try {
    const res = await fetch('/stats');
    const data = await res.json();

    document.getElementById('stat-total').textContent = data.total_scored.toLocaleString();
    document.getElementById('stat-flagrate').textContent = (data.flag_rate * 100).toFixed(2) + '%';
    document.getElementById('stat-throughput').textContent = data.throughput_per_sec.toFixed(1);
    document.getElementById('stat-p95').textContent = data.latency_ms_p95.toFixed(2);

    const labels = Object.keys(data.score_histogram);
    const values = Object.values(data.score_histogram);
    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = values;
    scoreChart.update();

    latencyChart.data.datasets[0].data = [data.latency_ms_p50, data.latency_ms_p95, data.latency_ms_p99];
    latencyChart.update();
  } catch (e) {
    console.error('failed to refresh stats', e);
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/")
def root() -> dict:
    return {"service": "realtime-risk-scoring-api", "docs": "/docs", "dashboard": "/dashboard"}
