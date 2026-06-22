"""Consumers service entrypoint — Confluent Kafka consumer dispatcher.

Dispatches ``--paradigm`` to its consumer (registry in consumers/paradigms.py):
each subscribes to ``cpv-raw`` via the shared consumer loop, classifies, and
writes predictions to PostgreSQL. An unknown paradigm falls through to the
print-only debug consumer below.

Usage::

    python main.py --paradigm hardcoded
"""

from __future__ import annotations

import argparse
import json
import time

from confluent_kafka import Consumer

from config import settings
from consumers.paradigms import DISPATCH as _DISPATCH

# Single source of truth for the registry lives in consumers/paradigms.py, so
# main.py / accuracy.py / rerun.sh can never drift apart (audit fix B5). Every
# listed paradigm is fully implemented and dispatched below; the print-only stub
# further down is now only reached for an unknown --paradigm.
PARADIGMS = list(_DISPATCH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a spendlabel consumer paradigm.")
    parser.add_argument("--paradigm", required=True, choices=PARADIGMS,
                        help="Which classification paradigm to run.")
    parser.add_argument("--idle-timeout", type=float, default=15.0,
                        help="Seconds without a message before stopping.")
    args = parser.parse_args()

    if args.paradigm in _DISPATCH:
        import importlib
        mod_name, cls_name = _DISPATCH[args.paradigm]
        getattr(importlib.import_module(mod_name), cls_name)().run(idle_timeout=args.idle_timeout)
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
