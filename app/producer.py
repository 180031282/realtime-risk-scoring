"""Kafka producer: simulates a live stream of synthetic payment transactions.

Publishes JSON-encoded transaction events onto the ``transactions.raw``
topic at a configurable rate, using the shared synthetic generator in
app/generator.py (the same generator the training script uses, so the
production-time distribution matches what the model was trained on).

Run directly:

    .venv/bin/python -m app.producer --rate 20 --fraud-rate 0.03

Configuration is via CLI flags or environment variables (CLI flags win):
    KAFKA_BOOTSTRAP_SERVERS  (default: localhost:9092)
    KAFKA_RAW_TOPIC          (default: transactions.raw)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

from confluent_kafka import KafkaException, Producer

from app.generator import generate_customer_pool, generate_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s producer %(levelname)s %(message)s")
logger = logging.getLogger("producer")

DEFAULT_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_RAW_TOPIC = os.environ.get("KAFKA_RAW_TOPIC", "transactions.raw")


def _delivery_report(err, msg) -> None:
    if err is not None:
        logger.error("delivery failed: %s", err)


def build_producer(bootstrap_servers: str) -> Producer:
    return Producer({"bootstrap.servers": bootstrap_servers})


def run(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = DEFAULT_RAW_TOPIC,
    rate_per_sec: float = 10.0,
    fraud_rate: float = 0.03,
    n_customers: int = 500,
    max_events: int | None = None,
    seed: int | None = None,
) -> None:
    """Publish synthetic transactions to Kafka at roughly ``rate_per_sec``.

    ``max_events`` of None runs forever (Ctrl+C to stop); a finite value is
    mainly useful for smoke-testing the producer without a long-running
    process.
    """
    rng = random.Random(seed)
    customers = generate_customer_pool(n_customers, rng)
    producer = build_producer(bootstrap_servers)

    interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
    sent = 0
    logger.info(
        "starting producer: topic=%s rate=%.2f/s fraud_rate=%.3f bootstrap=%s",
        topic, rate_per_sec, fraud_rate, bootstrap_servers,
    )

    try:
        while max_events is None or sent < max_events:
            raw, _label = generate_transaction(rng, customers, fraud_rate=fraud_rate)
            payload = json.dumps(raw).encode("utf-8")
            try:
                producer.produce(
                    topic,
                    key=raw["customer_id"].encode("utf-8"),
                    value=payload,
                    callback=_delivery_report,
                )
                producer.poll(0)
            except BufferError:
                logger.warning("local producer queue full, flushing")
                producer.flush()
                continue
            except KafkaException as exc:
                logger.error("failed to produce message: %s", exc)

            sent += 1
            if sent % 100 == 0:
                logger.info("sent %d events", sent)

            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    finally:
        producer.flush(10)
        logger.info("producer stopped after sending %d events", sent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic transaction producer")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_RAW_TOPIC)
    parser.add_argument("--rate", type=float, default=10.0, help="events per second")
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    parser.add_argument("--n-customers", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        rate_per_sec=args.rate,
        fraud_rate=args.fraud_rate,
        n_customers=args.n_customers,
        max_events=args.max_events,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
