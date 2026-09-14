"""Unit tests for app/features.py -- pure feature transform, no I/O."""
from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    FEATURE_NAMES,
    MERCHANT_CATEGORIES,
    FeatureValidationError,
    transform_batch,
    transform_to_feature_vector,
)

SAMPLE_RAW = {
    "transaction_id": "tx-1",
    "customer_id": "cust_00001",
    "timestamp": 1_700_000_000.0,
    "amount": 120.0,
    "merchant_category": "electronics",
    "hour_of_day": 14,
    "day_of_week": 2,
    "customer_txn_velocity_1h": 1,
    "customer_avg_amount_30d": 60.0,
    "distance_from_home_km": 5.0,
    "is_new_device": False,
    "is_new_merchant": True,
    "card_present": True,
    "account_age_days": 400,
    "num_failed_attempts_last_hour": 0,
}


def test_feature_vector_shape_matches_feature_names():
    vec = transform_to_feature_vector(SAMPLE_RAW)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec.dtype == np.float64


def test_numeric_values_pass_through_correctly():
    vec = transform_to_feature_vector(SAMPLE_RAW)
    values = dict(zip(FEATURE_NAMES, vec))
    assert values["amount"] == 120.0
    assert values["hour_of_day"] == 14
    assert values["day_of_week"] == 2
    assert values["customer_txn_velocity_1h"] == 1
    assert values["customer_avg_amount_30d"] == 60.0
    assert values["distance_from_home_km"] == 5.0
    assert values["account_age_days"] == 400
    assert values["num_failed_attempts_last_hour"] == 0


def test_amount_to_avg_ratio_is_computed():
    vec = transform_to_feature_vector(SAMPLE_RAW)
    values = dict(zip(FEATURE_NAMES, vec))
    assert values["amount_to_avg_ratio"] == pytest.approx(120.0 / 60.0)


def test_amount_to_avg_ratio_guards_against_zero_avg():
    raw = dict(SAMPLE_RAW, customer_avg_amount_30d=0.0)
    vec = transform_to_feature_vector(raw)
    values = dict(zip(FEATURE_NAMES, vec))
    # Falls back to using amount itself rather than dividing by zero / NaN.
    assert values["amount_to_avg_ratio"] == pytest.approx(raw["amount"])
    assert np.isfinite(vec).all()


def test_boolean_fields_are_encoded_as_zero_or_one():
    vec = transform_to_feature_vector(SAMPLE_RAW)
    values = dict(zip(FEATURE_NAMES, vec))
    assert values["is_new_device"] == 0.0
    assert values["is_new_merchant"] == 1.0
    assert values["card_present"] == 1.0


def test_merchant_category_one_hot_encoding():
    vec = transform_to_feature_vector(SAMPLE_RAW)
    values = dict(zip(FEATURE_NAMES, vec))
    for category in MERCHANT_CATEGORIES:
        expected = 1.0 if category == "electronics" else 0.0
        assert values[f"merchant_{category}"] == expected


def test_unknown_merchant_category_is_all_zero_one_hot():
    raw = dict(SAMPLE_RAW, merchant_category="not_a_real_category")
    vec = transform_to_feature_vector(raw)
    values = dict(zip(FEATURE_NAMES, vec))
    merchant_cols = [values[f"merchant_{c}"] for c in MERCHANT_CATEGORIES]
    assert sum(merchant_cols) == 0.0


def test_missing_required_field_raises():
    raw = dict(SAMPLE_RAW)
    del raw["amount"]
    with pytest.raises(FeatureValidationError):
        transform_to_feature_vector(raw)


def test_transform_batch_stacks_vectors():
    raws = [SAMPLE_RAW, dict(SAMPLE_RAW, amount=999.0, merchant_category="crypto")]
    X = transform_batch(raws)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert X[1][FEATURE_NAMES.index("amount")] == 999.0


def test_transform_batch_empty_input():
    X = transform_batch([])
    assert X.shape == (0, len(FEATURE_NAMES))


def test_generated_transactions_transform_cleanly(many_raw_transactions):
    """Every transaction the shared generator produces must be transformable."""
    X = transform_batch(many_raw_transactions)
    assert X.shape == (len(many_raw_transactions), len(FEATURE_NAMES))
    assert np.isfinite(X).all()
