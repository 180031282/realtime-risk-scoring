"""Model scoring and rolling stats aggregation.

Two responsibilities live here, both used by the consumer/scorer service and
(for stats) by the FastAPI service:

1. ``RiskScorer`` -- loads the trained XGBoost model and turns a raw
   transaction (or an already-computed feature vector) into a risk score in
   [0, 1] plus a flagged/not-flagged decision.
2. ``StatsStore`` -- a small SQLite-backed rolling window of
   (timestamp, latency_ms, risk_score, flagged) samples, with pure helper
   functions for the percentile/flag-rate/histogram math so that math can be
   unit tested with synthetic samples, with no SQLite, Kafka, or model
   involved at all.

The consumer writes to a StatsStore; the API reads from one pointed at the
same database file (or the same in-process instance, in tests).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np

from app.features import FEATURE_NAMES, transform_to_feature_vector

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "model.pkl"
DEFAULT_FLAG_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------


class RiskScorer:
    """Wraps a trained XGBoost classifier for real-time inference."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_FLAG_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None
        self._feature_names: list[str] = FEATURE_NAMES
        self._model_version: Optional[str] = None

    def load(self) -> "RiskScorer":
        bundle = joblib.load(self.model_path)
        self._model = bundle["model"]
        self._feature_names = bundle.get("feature_names", FEATURE_NAMES)
        self._model_version = bundle.get("model_version", "unknown")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    def score_vector(self, vec: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded; call RiskScorer.load() first.")
        proba = self._model.predict_proba(vec.reshape(1, -1))[0, 1]
        return float(proba)

    def score_raw(self, raw: dict) -> float:
        vec = transform_to_feature_vector(raw)
        return self.score_vector(vec)

    def is_flagged(self, score: float) -> bool:
        return score >= self.threshold


# ---------------------------------------------------------------------------
# Pure stats math (no I/O -- easy to unit test)
# ---------------------------------------------------------------------------


def compute_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute p50/p95/p99 latency from a sequence of samples.

    Returns zeros for an empty sequence rather than raising, since the
    aggregate stats endpoint should be well-defined even before any
    transaction has been scored.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compute_flag_rate(flags: Sequence[bool]) -> float:
    """Fraction of samples flagged as high-risk. 0.0 for an empty sequence."""
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)


def build_score_histogram(scores: Sequence[float], n_buckets: int = 10) -> dict[str, int]:
    """Bucket risk scores into n_buckets equal-width bins covering [0, 1].

    Bucket keys are half-open intervals like "0.0-0.1", ..., "0.9-1.0"
    (the final bucket is closed on both ends so a score of exactly 1.0 is
    counted). A score outside [0, 1] is clamped into range defensively.
    """
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(n_buckets)]
    histogram = {label: 0 for label in labels}
    for s in scores:
        clamped = min(max(float(s), 0.0), 1.0)
        idx = min(int(clamped * n_buckets), n_buckets - 1)
        histogram[labels[idx]] += 1
    return histogram


def compute_aggregate_stats(
    latencies_ms: Sequence[float],
    scores: Sequence[float],
    flags: Sequence[bool],
    elapsed_seconds: float,
) -> dict:
    """Pure aggregation: combine raw samples into the /stats payload shape.

    ``elapsed_seconds`` is the wall-clock width of the window the samples
    were drawn from, used only to compute throughput (events/sec).
    """
    n = len(latencies_ms)
    percentiles = compute_percentiles(latencies_ms)
    flag_rate = compute_flag_rate(flags)
    avg_score = float(np.mean(scores)) if scores else 0.0
    throughput = n / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "total_scored": n,
        "flagged_count": sum(1 for f in flags if f),
        "flag_rate": flag_rate,
        "throughput_per_sec": throughput,
        "latency_ms_p50": percentiles["p50"],
        "latency_ms_p95": percentiles["p95"],
        "latency_ms_p99": percentiles["p99"],
        "avg_risk_score": avg_score,
        "score_histogram": build_score_histogram(scores),
        "window_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# SQLite-backed rolling stats store
# ---------------------------------------------------------------------------


class StatsStore:
    """Rolling window store of scored-transaction samples, backed by SQLite.

    A file-based store lets the consumer process (the writer) and the API
    process (the reader) share state across process boundaries without a
    live Kafka connection on the read side -- the API never touches Kafka
    at all, it only ever reads this store.
    """

    def __init__(self, db_path: str = ":memory:", window_seconds: float = 300.0):
        self.db_path = db_path
        self.window_seconds = window_seconds
        # check_same_thread=False: FastAPI/uvicorn may call get_stats from a
        # different thread than the one that opened the connection.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        if db_path != ":memory:":
            # WAL mode allows the consumer (writer) and the API (reader) to
            # both hold open connections to the same file concurrently
            # without "database is locked" errors under docker-compose.
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_events (
                ts REAL NOT NULL,
                latency_ms REAL NOT NULL,
                risk_score REAL NOT NULL,
                flagged INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scored_events_ts ON scored_events(ts)"
        )
        self._conn.commit()

    def record(
        self,
        latency_ms: float,
        risk_score: float,
        flagged: bool,
        ts: Optional[float] = None,
    ) -> None:
        ts = time.time() if ts is None else ts
        self._conn.execute(
            "INSERT INTO scored_events (ts, latency_ms, risk_score, flagged) VALUES (?, ?, ?, ?)",
            (ts, latency_ms, risk_score, int(flagged)),
        )
        self._conn.commit()

    def prune(self, now: Optional[float] = None) -> None:
        """Drop samples older than the rolling window to bound storage."""
        now = time.time() if now is None else now
        cutoff = now - self.window_seconds
        self._conn.execute("DELETE FROM scored_events WHERE ts < ?", (cutoff,))
        self._conn.commit()

    def get_stats(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        cutoff = now - self.window_seconds
        rows = self._conn.execute(
            "SELECT ts, latency_ms, risk_score, flagged FROM scored_events WHERE ts >= ? ORDER BY ts",
            (cutoff,),
        ).fetchall()

        if not rows:
            stats = compute_aggregate_stats([], [], [], 0.0)
            stats["last_updated"] = None
            return stats

        timestamps = [r[0] for r in rows]
        latencies = [r[1] for r in rows]
        scores = [r[2] for r in rows]
        flags = [bool(r[3]) for r in rows]

        elapsed = max(now - min(timestamps), 1e-6)
        stats = compute_aggregate_stats(latencies, scores, flags, elapsed)
        stats["last_updated"] = max(timestamps)
        return stats

    def close(self) -> None:
        self._conn.close()
