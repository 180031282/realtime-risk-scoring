"""Tests for the consumer's pure scoring/result-building logic.

score_and_build_result is the part of app/consumer.py that has nothing to
do with Kafka -- it just takes a raw transaction dict and a loaded
RiskScorer and produces the scored-output event. No broker required.
"""
from __future__ import annotations

from app.consumer import score_and_build_result
from app.scoring import RiskScorer


def test_score_and_build_result_shape(sample_raw_transaction):
    scorer = RiskScorer().load()
    result = score_and_build_result(scorer, sample_raw_transaction)

    assert result["transaction_id"] == sample_raw_transaction["transaction_id"]
    assert result["customer_id"] == sample_raw_transaction["customer_id"]
    assert result["amount"] == sample_raw_transaction["amount"]
    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["flagged"], bool)
    assert result["latency_ms"] >= 0.0
    assert result["scored_at"] > 0


def test_score_and_build_result_flag_consistent_with_threshold(sample_raw_transaction):
    scorer = RiskScorer(threshold=0.0).load()  # flag everything
    result = score_and_build_result(scorer, sample_raw_transaction)
    assert result["flagged"] is True

    scorer_high = RiskScorer(threshold=1.01).load()  # flag nothing
    result_high = score_and_build_result(scorer_high, sample_raw_transaction)
    assert result_high["flagged"] is False
