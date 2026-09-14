"""Shared synthetic transaction generator.

Both the offline training script (training/train_model.py) and the online
producer (app/producer.py) import this module so that the distribution of
transactions the model is trained on matches the distribution it sees at
"real-time" scoring time.

Rather than calling sklearn.datasets.make_classification (which produces
features with no semantic meaning), this module hand-crafts a generative
process for synthetic payment transactions with a handful of customer
"profiles" and an explicit -- but deliberately noisy and overlapping --
notion of what makes a transaction fraud-like. The noise/overlap is
intentional: it keeps the resulting classification problem from being
trivially separable, so the trained model's precision/recall/AUC numbers
look like a real (if easy) risk-scoring problem rather than a toy with 1.0
AUC.

None of this data is real. All customer ids, amounts, locations, and device
flags are synthesized by this module for demo purposes only.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass

from app.features import MERCHANT_CATEGORIES

# Categories that are disproportionately attractive to fraudsters because
# goods/value can be moved or liquidated quickly (classic "cash-out"
# categories in the fraud literature this demo is loosely inspired by).
HIGH_RISK_CATEGORIES = {"crypto", "electronics", "jewelry", "travel", "gaming"}
LOW_RISK_CATEGORIES = {"grocery", "restaurant", "gas_station", "utilities"}


@dataclass
class CustomerProfile:
    customer_id: str
    typical_amount: float
    typical_hour: int
    account_age_days: int
    home_region: str


def generate_customer_pool(n_customers: int, rng: random.Random) -> list[CustomerProfile]:
    """Create a fixed pool of synthetic customer spending profiles."""
    profiles = []
    for i in range(n_customers):
        profiles.append(
            CustomerProfile(
                customer_id=f"cust_{i:05d}",
                typical_amount=max(5.0, rng.gauss(85.0, 60.0)),
                typical_hour=rng.randint(7, 22),
                account_age_days=rng.randint(15, 3000),
                home_region=rng.choice(["NA", "EU", "APAC", "LATAM"]),
            )
        )
    return profiles


def _pick_category(rng: random.Random, is_fraud: bool) -> str:
    if is_fraud and rng.random() < 0.65:
        return rng.choice(list(HIGH_RISK_CATEGORIES))
    return rng.choice(MERCHANT_CATEGORIES)


def generate_transaction(
    rng: random.Random,
    customers: list[CustomerProfile],
    fraud_rate: float = 0.03,
) -> tuple[dict, int]:
    """Generate a single synthetic raw transaction event.

    Returns
    -------
    (raw_transaction_dict, label) where label is 1 for a fraud-like
    (label-generating-process) transaction and 0 otherwise. The producer
    only ever publishes ``raw_transaction_dict`` (the label is never known
    in real time -- that's the entire point of the scorer); the label is
    used by the training script to build a supervised dataset.
    """
    profile = rng.choice(customers)
    is_fraud = rng.random() < fraud_rate
    category = _pick_category(rng, is_fraud)

    if is_fraud:
        # Fraudulent-pattern transactions: larger, odd-hour, higher velocity,
        # further from home, more likely on a new device / new merchant /
        # card-not-present, with more recent failed auth attempts. Every
        # signal is noisy (gaussian / bernoulli draws) and the separation
        # between the two branches is deliberately modest, so fraud and
        # legitimate transactions overlap substantially in feature space --
        # same as in practice, where no single feature cleanly separates
        # the classes.
        amount = max(1.0, rng.gauss(profile.typical_amount * 2.2, profile.typical_amount * 1.6))
        hour_of_day = rng.choice(
            [rng.randint(0, 6), rng.randint(0, 23)]  # skewed to late night, sometimes any hour
        )
        velocity_1h = max(0, int(rng.gauss(2.0, 2.0)))
        distance_km = max(0.0, rng.gauss(220, 280))
        is_new_device = rng.random() < 0.35
        is_new_merchant = rng.random() < 0.4
        card_present = rng.random() < 0.35
        failed_attempts = max(0, int(rng.gauss(1.0, 1.3)))
    else:
        amount = max(1.0, rng.gauss(profile.typical_amount, profile.typical_amount * 0.45))
        hour_of_day = max(0, min(23, int(rng.gauss(profile.typical_hour, 4))))
        velocity_1h = max(0, int(rng.gauss(0.7, 1.0)))
        distance_km = max(0.0, rng.gauss(15, 35))
        is_new_device = rng.random() < 0.08
        is_new_merchant = rng.random() < 0.2
        card_present = rng.random() < 0.75
        failed_attempts = 0 if rng.random() < 0.88 else 1

    day_of_week = rng.randint(0, 6)

    # Simulate imperfect ground-truth labeling (e.g. a fraud investigation
    # that reverses an initial call, or a missed case). This caps the
    # achievable precision/recall/AUC well below a suspicious 1.0, which is
    # what you'd expect from a real labeled fraud dataset.
    label = is_fraud
    if rng.random() < 0.008:
        label = not label

    raw = {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": profile.customer_id,
        "timestamp": time.time(),
        "amount": round(amount, 2),
        "merchant_category": category,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "customer_txn_velocity_1h": velocity_1h,
        "customer_avg_amount_30d": round(profile.typical_amount, 2),
        "distance_from_home_km": round(distance_km, 2),
        "is_new_device": is_new_device,
        "is_new_merchant": is_new_merchant,
        "card_present": card_present,
        "account_age_days": profile.account_age_days,
        "num_failed_attempts_last_hour": failed_attempts,
    }
    return raw, int(label)
