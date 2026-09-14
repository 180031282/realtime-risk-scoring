"""Unit tests for app/scoring.py: model inference and stats aggregation math.

None of these tests require a live Kafka broker -- RiskScorer only reads the
committed training/model.pkl from disk, and StatsStore uses an in-memory
SQLite database.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from app.features import FEATURE_NAMES, transform_to_feature_vector
from app.scoring import (
    RiskScorer,
    StatsStore,
    build_score_histogram,
    compute_aggregate_stats,
    compute_flag_rate,
    compute_percentiles,
)


# ---------------------------------------------------------------------------
# RiskScorer / model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_scorer() -> RiskScorer:
    scorer = RiskScorer()
    scorer.load()
    return scorer


def test_scorer_loads_committed_model(loaded_scorer: RiskScorer):
    assert loaded_scorer.is_loaded
    assert loaded_scorer.model_version is not None


def test_score_vector_returns_valid_probability(loaded_scorer: RiskScorer, sample_raw_transaction):
    vec = transform_to_feature_vector(sample_raw_transaction)
    score = loaded_scorer.score_vector(vec)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_raw_matches_score_vector(loaded_scorer: RiskScorer, sample_raw_transaction):
    vec = transform_to_feature_vector(sample_raw_transaction)
    assert loaded_scorer.score_raw(sample_raw_transaction) == pytest.approx(
        loaded_scorer.score_vector(vec)
    )


def test_all_generated_transactions_score_in_valid_range(loaded_scorer: RiskScorer, many_raw_transactions):
    for raw in many_raw_transactions:
        score = loaded_scorer.score_raw(raw)
        assert 0.0 <= score <= 1.0


def test_flag_threshold_behavior():
    scorer = RiskScorer(threshold=0.7)
    assert scorer.is_flagged(0.8) is True
    assert scorer.is_flagged(0.7) is True
    assert scorer.is_flagged(0.69) is False


def test_score_vector_requires_model_loaded():
    scorer = RiskScorer()  # not loaded
    with pytest.raises(RuntimeError):
        scorer.score_vector(np.zeros(len(FEATURE_NAMES)))


# ---------------------------------------------------------------------------
# Pure stats math
# ---------------------------------------------------------------------------


def test_compute_percentiles_matches_numpy_reference():
    latencies = [float(x) for x in range(1, 101)]  # 1..100
    result = compute_percentiles(latencies)
    assert result["p50"] == pytest.approx(np.percentile(latencies, 50))
    assert result["p95"] == pytest.approx(np.percentile(latencies, 95))
    assert result["p99"] == pytest.approx(np.percentile(latencies, 99))


def test_compute_percentiles_empty_is_zero():
    result = compute_percentiles([])
    assert result == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_compute_flag_rate_basic():
    flags = [True, True, False, False, False, False, False, False, False, False]
    assert compute_flag_rate(flags) == pytest.approx(0.2)


def test_compute_flag_rate_empty_is_zero():
    assert compute_flag_rate([]) == 0.0


def test_compute_flag_rate_all_flagged():
    assert compute_flag_rate([True, True, True]) == pytest.approx(1.0)


def test_build_score_histogram_buckets_correctly():
    scores = [0.05, 0.15, 0.15, 0.99, 1.0, 0.5]
    hist = build_score_histogram(scores, n_buckets=10)
    assert sum(hist.values()) == len(scores)
    assert hist["0.0-0.1"] == 1
    assert hist["0.1-0.2"] == 2
    assert hist["0.9-1.0"] == 2  # 0.99 and 1.0 (closed on the right)
    assert hist["0.5-0.6"] == 1


def test_build_score_histogram_clamps_out_of_range_values():
    hist = build_score_histogram([-0.5, 1.5], n_buckets=10)
    assert hist["0.0-0.1"] == 1
    assert hist["0.9-1.0"] == 1


def test_compute_aggregate_stats_end_to_end():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    scores = [0.1, 0.2, 0.9, 0.95, 0.3]
    flags = [False, False, True, True, False]
    stats = compute_aggregate_stats(latencies, scores, flags, elapsed_seconds=5.0)

    assert stats["total_scored"] == 5
    assert stats["flagged_count"] == 2
    assert stats["flag_rate"] == pytest.approx(0.4)
    assert stats["throughput_per_sec"] == pytest.approx(1.0)
    assert stats["avg_risk_score"] == pytest.approx(np.mean(scores))
    assert stats["latency_ms_p50"] == pytest.approx(np.percentile(latencies, 50))
    assert sum(stats["score_histogram"].values()) == 5


def test_compute_aggregate_stats_zero_elapsed_gives_zero_throughput():
    stats = compute_aggregate_stats([1.0], [0.5], [False], elapsed_seconds=0.0)
    assert stats["throughput_per_sec"] == 0.0


# ---------------------------------------------------------------------------
# StatsStore (SQLite-backed, in-memory for tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def stats_store() -> StatsStore:
    store = StatsStore(db_path=":memory:", window_seconds=60.0)
    yield store
    store.close()


def test_stats_store_empty_returns_zeroed_stats(stats_store: StatsStore):
    stats = stats_store.get_stats()
    assert stats["total_scored"] == 0
    assert stats["flag_rate"] == 0.0
    assert stats["last_updated"] is None


def test_stats_store_records_and_aggregates(stats_store: StatsStore):
    now = time.time()
    stats_store.record(latency_ms=10.0, risk_score=0.2, flagged=False, ts=now)
    stats_store.record(latency_ms=20.0, risk_score=0.9, flagged=True, ts=now)
    stats_store.record(latency_ms=30.0, risk_score=0.4, flagged=False, ts=now)

    stats = stats_store.get_stats(now=now + 1)
    assert stats["total_scored"] == 3
    assert stats["flagged_count"] == 1
    assert stats["flag_rate"] == pytest.approx(1 / 3)
    assert stats["last_updated"] == pytest.approx(now)


def test_stats_store_excludes_samples_outside_window(stats_store: StatsStore):
    now = time.time()
    # This sample is far outside the 60s rolling window.
    stats_store.record(latency_ms=999.0, risk_score=1.0, flagged=True, ts=now - 500)
    stats_store.record(latency_ms=15.0, risk_score=0.3, flagged=False, ts=now)

    stats = stats_store.get_stats(now=now + 1)
    assert stats["total_scored"] == 1
    assert stats["latency_ms_p50"] == pytest.approx(15.0)


def test_stats_store_prune_removes_old_rows(stats_store: StatsStore):
    now = time.time()
    stats_store.record(latency_ms=1.0, risk_score=0.1, flagged=False, ts=now - 500)
    stats_store.record(latency_ms=2.0, risk_score=0.1, flagged=False, ts=now)

    stats_store.prune(now=now)
    row_count = stats_store._conn.execute("SELECT COUNT(*) FROM scored_events").fetchone()[0]
    assert row_count == 1
