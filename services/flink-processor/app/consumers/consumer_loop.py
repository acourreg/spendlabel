"""Shared Kafka consume loop for every paradigm (tronc commun).

Consumes ``cpv-raw``, hands each batch of texts to a paradigm's ``Classifier``
(``classify(texts) -> [predicted_cpv | None]``), measures per-message latency, and
bulk-writes predictions to PostgreSQL. ``ground_truth`` / ``is_correct`` are left
NULL — the consumer never sees the truth; ``consumers/accuracy.py`` fills them in
after the run.

It also reports live progress (processed / total, where total = the topic's message
count via Kafka watermarks) to the ``run_progress`` table, which the ui-chart polls
every 5s to draw a per-lane progress bar.

Per-message latency = (time to classify the batch) / batch size, computed the same
way for every paradigm so the numbers are comparable.
"""

from __future__ import annotations

import json
import os
import time

from confluent_kafka import Consumer, TopicPartition

from config import settings
from db import connection

# Small batch on purpose: the MiniLM attention tensors scale with batch size, and
# the consumer shares a ~4GB Docker VM with kafka/postgres. 32 keeps onnxruntime
# well under the memory ceiling while staying fast enough on 47k records.
BATCH = 32


def _topic_total(consumer: Consumer, topic: str) -> int:
    """Total messages currently in the topic (sum of per-partition watermarks)."""
    try:
        md = consumer.list_topics(topic, timeout=10)
        total = 0
        for pid in md.topics[topic].partitions:
            lo, hi = consumer.get_watermark_offsets(
                TopicPartition(topic, pid), timeout=10, cached=False
            )
            total += max(0, hi - lo)
        return total
    except Exception:
        return 0


def run_consumer(paradigm: str, classifier, idle_timeout: float = 15.0,
                 max_seconds: float | None = None, max_records: int | None = None) -> None:
    """Drive a paradigm's classifier over cpv-raw and write predictions to Postgres.

    ``classifier`` only needs a ``classify(texts: list[str]) -> list[str | None]``
    method — all Kafka / DB / latency / progress plumbing lives here.

    Two caps bound a run (whichever hits first), so a benchmark never blocks for
    hours on the slow ML paradigms:
      - ``max_records`` (env MAX_RUN_RECORDS, 0 = unlimited): stop after N messages.
        With offset reset, every paradigm processes the SAME first N records of
        cpv-raw → an apples-to-apples comparison.
      - ``max_seconds`` (env MAX_RUN_SECONDS, default 300s): wall-clock safety net.
    Accuracy is scored over whatever was processed.
    """
    if max_seconds is None:
        max_seconds = float(os.getenv("MAX_RUN_SECONDS", "300"))
    if max_records is None:
        max_records = int(os.getenv("MAX_RUN_RECORDS", "0"))  # 0 = unlimited
    tag = f"[{paradigm}]"
    connection.init_schema()  # idempotent

    bootstrap = settings.kafka.bootstrap_servers
    topic = settings.kafka.topic_raw
    print(f"{tag} connecting to Kafka at {bootstrap}, subscribing to '{topic}'", flush=True)
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"spendlabel-{paradigm}",
            "auto.offset.reset": "earliest",
        }
    )

    total = _topic_total(consumer, topic)
    connection.upsert_progress(paradigm, 0, total, "running")
    print(f"{tag} {total} messages to process", flush=True)

    consumer.subscribe([topic])

    buf: list[tuple] = []  # (ocid, text, value_gbp, supplier_name)
    processed = 0
    start = time.monotonic()
    last_seen = start

    def flush() -> None:
        nonlocal processed
        if not buf:
            return
        t0 = time.perf_counter()
        preds = classifier.classify([b[1] for b in buf])
        per_ms = (time.perf_counter() - t0) * 1000.0 / len(buf)
        rows = [
            (b[0], paradigm, preds[i], None, None, per_ms, b[2], b[3])
            for i, b in enumerate(buf)
        ]
        connection.insert_classifications_batch(rows)
        processed += len(buf)
        buf.clear()
        connection.upsert_progress(paradigm, processed, total, "running")

    try:
        while True:
            if time.monotonic() - start > max_seconds:
                print(f"{tag} max runtime {max_seconds:.0f}s reached — stopping at "
                      f"processed={processed}/{total} (partial slice)", flush=True)
                break
            msg = consumer.poll(1.0)
            if msg is None:
                flush()
                if time.monotonic() - last_seen > idle_timeout:
                    print(f"{tag} no messages for {idle_timeout:.0f}s — stopping. "
                          f"processed={processed}", flush=True)
                    break
                continue
            if msg.error():
                print(f"{tag} kafka error: {msg.error()}", flush=True)
                continue

            last_seen = time.monotonic()
            try:
                rec = json.loads(msg.value())
            except (ValueError, AttributeError):
                continue
            # NB: cpv_ground_truth is deliberately NOT read here.
            ocid = rec.get("ocid")
            if not ocid:
                continue
            text = (rec.get("title") or "") + " " + (rec.get("description") or "")
            buf.append((ocid, text, rec.get("value_gbp"), rec.get("supplier_name")))
            if max_records and processed + len(buf) >= max_records:
                del buf[max_records - processed:]  # trim to hit the cap exactly
                flush()
                print(f"{tag} record cap {max_records} reached — stopping.", flush=True)
                break
            if len(buf) >= BATCH:
                flush()
                if processed % (BATCH * 20) == 0:
                    print(f"{tag} classified {processed}/{total}...", flush=True)
    finally:
        flush()
        connection.upsert_progress(paradigm, processed, total, "done")
        consumer.close()
        print(f"{tag} run complete (processed={processed})", flush=True)
