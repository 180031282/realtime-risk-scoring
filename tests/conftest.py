"""Shared pytest fixtures and test-environment setup.

Importantly: this sets STATS_DB_PATH to an in-memory SQLite database
*before* app.api is imported anywhere in the test session, so running the
test suite never creates, reads, or mutates a real stats.db on disk (which
could otherwise collide with a locally running consumer/API pair), and
never requires a live Kafka broker.
"""
from __future__ import annotations

import os
import random

import pytest

os.environ.setdefault("STATS_DB_PATH", ":memory:")

from app.generator import generate_customer_pool, generate_transaction  # noqa: E402


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)


@pytest.fixture
def customer_pool(rng: random.Random):
    return generate_customer_pool(20, rng)


@pytest.fixture
def sample_raw_transaction(rng: random.Random, customer_pool) -> dict:
    raw, _label = generate_transaction(rng, customer_pool, fraud_rate=0.1)
    return raw


@pytest.fixture
def many_raw_transactions(rng: random.Random, customer_pool) -> list[dict]:
    return [generate_transaction(rng, customer_pool, fraud_rate=0.1)[0] for _ in range(200)]
