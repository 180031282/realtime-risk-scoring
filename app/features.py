"""Transaction schema and feature engineering.

This module is the single source of truth for the raw transaction schema
(what the producer emits onto Kafka) and for the transform from a raw
transaction dict into the fixed-order numeric feature vector consumed by the
XGBoost model.

Keeping this transform pure (no I/O, no globals mutated, deterministic given
its input) means:
  * it is trivially unit-testable in isolation (see tests/test_features.py),
  * both the offline training script and the real-time consumer import the
    exact same code path, which eliminates train/serve skew,
  * it has no dependency on Kafka, FastAPI, or XGBoost, so importing it never
    requires a broker or a model file.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Merchant category vocabulary. Fixed and small on purpose -- this is a demo
# pipeline, not a general-purpose merchant taxonomy.
MERCHANT_CATEGORIES: list[str] = [
    "grocery",
    "restaurant",
    "gas_station",
    "online_retail",
    "electronics",
    "travel",
    "gaming",
    "crypto",
    "jewelry",
    "utilities",
]

# Numeric / boolean fields read directly (or derived) from the raw
# transaction dict, in the exact order fed to the model.
NUMERIC_FEATURES: list[str] = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "customer_txn_velocity_1h",
    "customer_avg_amount_30d",
    "amount_to_avg_ratio",
    "distance_from_home_km",
    "is_new_device",
    "is_new_merchant",
    "card_present",
    "account_age_days",
    "num_failed_attempts_last_hour",
]

# One-hot encoded merchant category columns, appended after the numeric
# block. This full list is the exact column order the model was trained on.
FEATURE_NAMES: list[str] = NUMERIC_FEATURES + [
    f"merchant_{c}" for c in MERCHANT_CATEGORIES
]

# Fields every raw transaction event must carry. amount/merchant_category/etc
# are business fields; transaction_id/customer_id/timestamp are metadata that
# flow through to the scored output but are not fed to the model directly.
REQUIRED_RAW_FIELDS: list[str] = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant_category",
    "hour_of_day",
    "day_of_week",
    "customer_txn_velocity_1h",
    "customer_avg_amount_30d",
    "distance_from_home_km",
    "is_new_device",
    "is_new_merchant",
    "card_present",
    "account_age_days",
    "num_failed_attempts_last_hour",
]


class FeatureValidationError(ValueError):
    """Raised when a raw transaction dict is missing required fields."""


def _one_hot_merchant(category: str) -> list[float]:
    """One-hot encode a merchant category into MERCHANT_CATEGORIES-length list.

    Unknown categories map to the all-zero vector rather than raising, so the
    pipeline degrades gracefully if a new merchant category shows up in
    production traffic that wasn't present at training time.
    """
    return [1.0 if category == c else 0.0 for c in MERCHANT_CATEGORIES]


def transform_to_feature_vector(raw: Mapping[str, Any]) -> np.ndarray:
    """Transform a raw transaction dict into a model-ready feature vector.

    Parameters
    ----------
    raw:
        A mapping with at least the keys in ``REQUIRED_RAW_FIELDS``, shaped
        like the JSON events published to the ``transactions.raw`` topic.

    Returns
    -------
    np.ndarray of shape ``(len(FEATURE_NAMES),)`` and dtype float64, in the
    exact column order of ``FEATURE_NAMES``.

    Raises
    ------
    FeatureValidationError
        If a required field is missing.
    """
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in raw]
    if missing:
        raise FeatureValidationError(
            f"raw transaction is missing required field(s): {missing}"
        )

    amount = float(raw["amount"])
    customer_avg_amount_30d = float(raw["customer_avg_amount_30d"])
    # Guard against division by zero for brand-new customers with no history.
    amount_to_avg_ratio = amount / customer_avg_amount_30d if customer_avg_amount_30d > 0 else amount

    numeric_values = [
        amount,
        float(raw["hour_of_day"]),
        float(raw["day_of_week"]),
        float(raw["customer_txn_velocity_1h"]),
        customer_avg_amount_30d,
        amount_to_avg_ratio,
        float(raw["distance_from_home_km"]),
        float(bool(raw["is_new_device"])),
        float(bool(raw["is_new_merchant"])),
        float(bool(raw["card_present"])),
        float(raw["account_age_days"]),
        float(raw["num_failed_attempts_last_hour"]),
    ]

    merchant_values = _one_hot_merchant(raw["merchant_category"])

    return np.array(numeric_values + merchant_values, dtype=np.float64)


def transform_batch(raws: list[Mapping[str, Any]]) -> np.ndarray:
    """Vectorized convenience wrapper around transform_to_feature_vector."""
    if not raws:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([transform_to_feature_vector(r) for r in raws])
