"""Kafka producer — reads the dataset tarball in memory and publishes one JSON
message per ocid to the ``cpv-raw`` topic.

The archive is read straight from ``data/raw/dataset.tar.gz`` via tarfile +
io.TextIOWrapper — members are never extracted to disk.

REAL CSV COLUMNS (verified with csv.DictReader on the actual dataset; the
mapping in the task brief was approximate, these are the exact names):

  main.csv  (driver, one logical record per `ocid`):
    ocid                              -> join key + message ocid
    tender_title                      -> title   (the plain `title` col is mostly empty)
    title                             -> title fallback
    tender_description                -> description
    tender_value_amount               -> value
    tender_value_currency             -> currency
    tender_procurementMethod          -> procurement_method
    tender_procurementMethodDetails   -> procurement_method fallback
    tender_classification_id          -> primary CPV (NOT used as ground truth here;
                                         the brief asks for additionalClassifications)

  tender_additionalClassifications.csv  (join on `main_ocid`):
    id           -> CPV code (e.g. '44115200'); ground truth = first 2 chars
    scheme       -> 'CPV'
    description  -> CPV label (unused; we map our own CPV_LABELS)
    main_ocid    -> join key

  awards.csv  (join on `main_ocid`):
    main_ocid    -> join key
    date         -> award_date (real award date; main.csv has no award date)
    value_amount, value_currency, status, description
    >>> NOTE: there is NO supplier_name column anywhere in the dataset, so
        `supplier_name` is always null. <<<

Message format (format B — flexible): every field is optional except `ocid`
and `title`. Empty string or missing CSV cell -> null. dict.get() everywhere,
never a KeyError.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import csv

from confluent_kafka import Producer
from dotenv import load_dotenv

DATASET_PATH = os.getenv("DATASET_PATH", "data/raw/dataset.tar.gz")
TOPIC = "cpv-raw"

CPV_LABELS = {
    "45": "Construction",
    "48": "Software",
    "60": "Transport",
    "71": "Engineering",
    "72": "IT Services",
    "73": "Research",
    "75": "Public administration",
    "79": "Business services",
    "80": "Education",
    "85": "Health",
    "90": "Environmental",
    "98": "Other community services",
}


def _clean(value):
    """Normalise a CSV cell: missing or empty/whitespace -> None."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _reader(tar: tarfile.TarFile, filename: str) -> csv.DictReader:
    """Return a DictReader over a tar member, read as text without extracting."""
    member = next(m for m in tar.getmembers() if m.name.split("/")[-1] == filename)
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"{filename} not found in {DATASET_PATH}")
    return csv.DictReader(io.TextIOWrapper(fileobj, encoding="utf-8", errors="replace"))


def _build_cpv_map(tar: tarfile.TarFile) -> dict[str, str]:
    """ocid -> 2-digit CPV ground truth (first additionalClassification seen)."""
    cpv = {}
    for row in _reader(tar, "tender_additionalClassifications.csv"):
        ocid = _clean(row.get("main_ocid"))
        code = _clean(row.get("id"))
        if ocid and code and ocid not in cpv:
            cpv[ocid] = code[:2]
    return cpv


def _build_award_map(tar: tarfile.TarFile) -> dict[str, dict]:
    """ocid -> first award row (for award_date; no supplier_name in the data)."""
    awards = {}
    for row in _reader(tar, "awards.csv"):
        ocid = _clean(row.get("main_ocid"))
        if ocid and ocid not in awards:
            awards[ocid] = row
    return awards


def _build_message(main_row: dict, cpv_map: dict, award_map: dict):
    """Assemble one flexible message; return None if a required field is missing."""
    ocid = _clean(main_row.get("ocid"))
    title = _clean(main_row.get("tender_title")) or _clean(main_row.get("title"))
    if not ocid or not title:  # required fields
        return None

    amount = _clean(main_row.get("tender_value_amount"))
    currency = _clean(main_row.get("tender_value_currency"))
    value_gbp = None
    if amount and currency == "GBP":
        try:
            value_gbp = float(amount)
        except ValueError:
            value_gbp = None

    # Ground truth: prefer additionalClassifications (per brief); fall back to
    # main.csv's primary `tender_classification_id` so coverage isn't sparse —
    # most records have a primary CPV but no additionalClassification row.
    cpv_gt = cpv_map.get(ocid)
    if cpv_gt is None:
        primary = _clean(main_row.get("tender_classification_id"))
        cpv_gt = primary[:2] if primary else None
    award = award_map.get(ocid) or {}

    return {
        "ocid": ocid,
        "title": title,
        "description": _clean(main_row.get("tender_description")),
        "value_gbp": value_gbp,
        "currency": currency,
        "cpv_ground_truth": cpv_gt,
        "cpv_label": CPV_LABELS.get(cpv_gt) if cpv_gt else None,
        "supplier_name": None,  # absent from the dataset (see module docstring)
        "procurement_method": (
            _clean(main_row.get("tender_procurementMethod"))
            or _clean(main_row.get("tender_procurementMethodDetails"))
        ),
        "award_date": _clean(award.get("date")),
    }


def _kafka_config() -> dict:
    """Build confluent-kafka config from KAFKA_MODE (local | confluent)."""
    load_dotenv("config/confluent.env")
    mode = os.getenv("KAFKA_MODE", "local").lower()
    base = {"client.id": "spendlabel-producer"}

    if mode == "confluent":
        return {
            **base,
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "sasl.mechanisms": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
            "sasl.username": os.getenv("KAFKA_API_KEY"),
            "sasl.password": os.getenv("KAFKA_API_SECRET"),
        }
    # local docker-compose dev
    return {**base, "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")}


def _on_delivery(err, msg) -> None:
    if err is not None:
        print(f"delivery failed for {msg.key()}: {err}", file=sys.stderr, flush=True)


def main() -> None:
    conf = _kafka_config()
    print(f"producer: mode={os.getenv('KAFKA_MODE', 'local')} "
          f"bootstrap={conf.get('bootstrap.servers')} topic={TOPIC}", flush=True)
    producer = Producer(conf)

    # Optional cap for smoke-testing (LIMIT=100 → stop after 100 messages).
    limit = int(os.getenv("LIMIT", "0")) or None

    sent = 0
    skipped = 0
    seen: set[str] = set()

    with tarfile.open(DATASET_PATH, "r:gz") as tar:
        # Side tables fit comfortably in memory; main.csv is streamed.
        cpv_map = _build_cpv_map(tar)
        award_map = _build_award_map(tar)
        print(f"loaded {len(cpv_map)} CPV codes, {len(award_map)} awards", flush=True)

        for main_row in _reader(tar, "main.csv"):
            ocid = _clean(main_row.get("ocid"))
            if ocid and ocid in seen:  # one JSON per ocid
                continue
            message = _build_message(main_row, cpv_map, award_map)
            if message is None:
                skipped += 1
                continue
            seen.add(message["ocid"])

            producer.produce(
                TOPIC,
                key=message["ocid"].encode("utf-8"),
                value=json.dumps(message).encode("utf-8"),
                on_delivery=_on_delivery,
            )
            sent += 1

            # After each 1000 messages: serve delivery callbacks + log progress.
            if sent % 1000 == 0:
                producer.poll(0)
                producer.flush()
                print(f"  produced {sent} messages...", flush=True)

            if limit and sent >= limit:
                print(f"  LIMIT={limit} reached, stopping early", flush=True)
                break

    producer.flush()
    print(f"done: produced={sent} skipped(no ocid/title)={skipped} unique_ocids={len(seen)}",
          flush=True)


if __name__ == "__main__":
    main()
