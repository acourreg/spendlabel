"""Consumers service entrypoint — PyFlink placeholder stub.

For now this is a stub: it connects to Kafka, consumes ``cpv-raw``, prints each
record's ocid, and stops once the stream is idle. It writes nothing to
PostgreSQL yet — the real PyFlink classification + sink lands per paradigm
later (the job classes under ``consumers/`` and the ``db`` layer are kept for
that). The ``--paradigm`` flag is accepted but only labels the log output.

Usage::

    python main.py --paradigm hardcoded
"""

from __future__ import annotations

import argparse
import json
import time

from confluent_kafka import Consumer

from config import settings

PARADIGMS = [
    "hardcoded",
    "spark_ml",
    "deeplearning_onnx",
    "solver",
    "langchain",
    "n8n",
    "mcp",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a spendlabel consumer paradigm (stub).")
    parser.add_argument("--paradigm", required=True, choices=PARADIGMS,
                        help="Which classification paradigm to label this run as.")
    parser.add_argument("--idle-timeout", type=float, default=15.0,
                        help="Seconds without a message before stopping.")
    args = parser.parse_args()

    # Real consumers take over per paradigm as they're implemented; the rest
    # fall through to the stub below (connect, print ocid, stop).
    if args.paradigm == "hardcoded":
        from consumers.hardcoded.job import HardcodedJob
        HardcodedJob().run(idle_timeout=args.idle_timeout)
        return
    if args.paradigm == "deeplearning_onnx":
        from consumers.deeplearning_onnx.job import DeepLearningJob
        DeepLearningJob().run(idle_timeout=args.idle_timeout)
        return
    if args.paradigm == "spark_ml":
        from consumers.spark_ml.job import SparkMLConsumer
        SparkMLConsumer().run(idle_timeout=args.idle_timeout)
        return

    bootstrap = settings.kafka.bootstrap_servers
    topic = settings.kafka.topic_raw
    tag = f"[{args.paradigm}]"

    print(f"{tag} connecting to Kafka at {bootstrap}, subscribing to '{topic}'", flush=True)
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"spendlabel-{args.paradigm}",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])

    received = 0
    last_seen = time.monotonic()
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                if time.monotonic() - last_seen > args.idle_timeout:
                    print(f"{tag} no messages for {args.idle_timeout:.0f}s — stopping. "
                          f"received={received}", flush=True)
                    break
                continue
            if msg.error():
                print(f"{tag} kafka error: {msg.error()}", flush=True)
                continue

            last_seen = time.monotonic()
            received += 1
            ocid = None
            try:
                payload = json.loads(msg.value())
                ocid = payload.get("ocid") or payload.get("id")
            except (ValueError, AttributeError):
                pass
            print(f"{tag} received ocid={ocid}", flush=True)
    finally:
        consumer.close()
        print(f"{tag} run complete", flush=True)


if __name__ == "__main__":
    main()
