"""Pydantic models shared across the API, consumer, and stats store."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ScoredTransaction(BaseModel):
    """The event shape published to the ``transactions.scored`` Kafka topic."""

    transaction_id: str
    customer_id: str
    timestamp: float
    amount: float
    merchant_category: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    latency_ms: float
    scored_at: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool
    model_version: Optional[str] = None


class StatsResponse(BaseModel):
    """Aggregate real-time stats served by GET /stats."""

    total_scored: int
    flagged_count: int
    flag_rate: float
    throughput_per_sec: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    avg_risk_score: float
    score_histogram: dict[str, int]
    window_seconds: float
    last_updated: Optional[float] = None
