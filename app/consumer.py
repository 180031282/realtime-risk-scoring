"""Kafka consumer / real-time scorer.

Consumes raw transaction events from ``transactions.raw``, extracts model
features, scores each transaction with the trained XGBoost model, measures
per-message scoring latency, publishes the scored result onto
``transactions.scored``, and records rolling aggregate stats (throughput,
latency percentiles, flag rate, score distribution) into a SQLite-backed
``StatsStore`` that the FastAPI service reads from.

Run directly:

    .venv/bin/python -m app.consumer

Configuration is via environment variables:
    KAFKA_BOOTSTRAP_SERVERS  (default: localhost:9092)
    KAFKA_RAW_TOPIC          (default: transactions.raw)
    KAFKA_SCORED_TOPIC       (default: transactions.scored)
    KAFKA_CONSUMER_GROUP     (default: risk-scorer)
    MODEL_PATH               (default: training/model.pkl)
    STATS_DB_PATH            (default: stats.db)
    RISK_FLAG_THRESHOLD      (default: 0.5)
"""
from __future__ import annotations

import json
import logging
import os
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from app.features import FeatureValidationError, transform_to_feature_vector
from app.scoring import DEFAULT_FLAG_THRESHOLD, RiskScorer, StatsStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s consumer %(levelname)s %(message)s")
logger = logging.getLogger("consumer")

DEFAULT_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_RAW_TOPIC = os.environ.get("KAFKA_RAW_TOPIC", "transactions.raw")
DEFAULT_SCORED_TOPIC = os.environ.get("KAFKA_SCORED_TOPIC", "transactions.scored")
DEFAULT_GROUP_ID = os.environ.get("KAFKA_CONSUMER_GROUP", "risk-scorer")
DEFAULT_STATS_DB_PATH = os.environ.get("STATS_DB_PATH", "stats.db")


def score_and_build_result(scorer: RiskScorer, raw: dict) -> dict:
    """Score one raw transaction and build the scored-output event.

    Pulled out of the Kafka read/write loop so it can be unit tested (and
    reused by the API/tests) without a broker.
    """
    start = time.perf_counter()
    vec = transform_to_feature_vector(raw)
    score = scorer.score_vector(vec)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "transaction_id": raw["transaction_id"],
        "customer_id": raw["customer_id"],
        "timestamp": raw["timestamp"],
        "amount": raw["amount"],
        "merchant_category": raw["merchant_category"],
        "risk_score": score,
        "flagged": scorer.is_flagged(score),
        "latency_ms": latency_ms,
        "scored_at": time.time(),
    }


def build_consumer(bootstrap_servers: str, group_id: str, topic: str) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([topic])
    return consumer


def build_producer(bootstrap_servers: str) -> Producer:
    return Producer({"bootstrap.servers": bootstrap_servers})


def run(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    raw_topic: str = DEFAULT_RAW_TOPIC,
    scored_topic: str = DEFAULT_SCORED_TOPIC,
    group_id: str = DEFAULT_GROUP_ID,
    model_path: str | None = None,
    stats_db_path: str = DEFAULT_STATS_DB_PATH,
    threshold: float = DEFAULT_FLAG_THRESHOLD,
    max_messages: int | None = None,
) -> None:
    scorer = RiskScorer(threshold=threshold) if model_path is None else RiskScorer(
        model_path=model_path, threshold=threshold
    )
    scorer.load()
    logger.info("loaded model version=%s from %s", scorer.model_version, scorer.model_path)

    stats_store = StatsStore(db_path=stats_db_path)
    consumer = build_consumer(bootstrap_servers, group_id, raw_topic)
    producer = build_producer(bootstrap_servers)

    processed = 0
    logger.info(
        "starting consumer: raw_topic=%s scored_topic=%s group=%s bootstrap=%s",
        raw_topic, scored_topic, group_id, bootstrap_servers,
    )

    try:
        while max_messages is None or processed < max_messages:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            try:
                raw = json.loads(msg.value())
                result = score_and_build_result(scorer, raw)
            except (FeatureValidationError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("skipping malformed message: %s", exc)
                continue

            producer.produce(
                scored_topic,
                key=result["customer_id"].encode("utf-8"),
                value=json.dumps(result).encode("utf-8"),
            )
            producer.poll(0)

            stats_store.record(
                latency_ms=result["latency_ms"],
                risk_score=result["risk_score"],
                flagged=result["flagged"],
            )

            processed += 1
            if processed % 100 == 0:
                stats_store.prune()
                logger.info("processed %d events", processed)
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    finally:
        producer.flush(10)
        stats_store.close()
        consumer.close()
        logger.info("consumer stopped after processing %d events", processed)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
