"""Common accuracy scorer — scores a finished run, the same way for every paradigm.

Run AFTER a paradigm has consumed and written its predictions (ground_truth left
NULL by the consumer). Orchestrated by scripts/rerun.sh:

    python -m consumers.accuracy --paradigm hardcoded

This is the ONLY place that reads the CPV ground truth. The consumer never sees it;
here we re-derive it from the dataset, join on ocid to score each prediction
(is_correct per message), then materialise the per-paradigm aggregate (accuracy /
avg_latency / total) into the ``metrics`` table — which the ui-chart reads, so
results show up in the dashboard right after a run.

NOTE on units: ``metrics.accuracy`` is a 0–1 fraction (the ui-chart multiplies by
100 for display). ``is_correct`` is NULL when the consumer made no prediction
(e.g. hardcoded keyword miss) — such rows are excluded from accuracy (AVG ignores
NULL), not counted wrong. ML heads (spark_ml, deeplearning_onnx) always predict, so
they have no NULLs and accuracy = correct / total.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import tarfile

import psycopg2
from psycopg2.extras import execute_batch

DATASET_PATH = os.getenv("DATASET_PATH", "data/raw/dataset.tar.gz")

PARADIGMS = [
    "hardcoded", "spark_ml", "deeplearning_onnx", "solver", "langchain", "n8n", "mcp",
]


def _clean(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def _reader(tar: tarfile.TarFile, filename: str) -> csv.DictReader:
    member = next(m for m in tar.getmembers() if m.name.split("/")[-1] == filename)
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"{filename} not found in {DATASET_PATH}")
    return csv.DictReader(io.TextIOWrapper(fileobj, encoding="utf-8", errors="replace"))


def build_ground_truth() -> dict[str, str]:
    """ocid -> 2-digit CPV ground truth.

    Same derivation as the producer so the score matches what was emitted:
    additionalClassifications first, else main.csv's primary tender_classification_id.
    """
    gt: dict[str, str] = {}
    with tarfile.open(DATASET_PATH, "r:gz") as tar:
        for row in _reader(tar, "tender_additionalClassifications.csv"):
            ocid = _clean(row.get("main_ocid"))
            code = _clean(row.get("id"))
            if ocid and code and ocid not in gt:
                gt[ocid] = code[:2]
        for row in _reader(tar, "main.csv"):
            ocid = _clean(row.get("ocid"))
            if ocid and ocid not in gt:
                primary = _clean(row.get("tender_classification_id"))
                if primary:
                    gt[ocid] = primary[:2]
    return gt


def _connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DB", "spendlabel"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a consumer run and materialise metrics.")
    parser.add_argument("--paradigm", required=True, choices=PARADIGMS)
    args = parser.parse_args()
    paradigm = args.paradigm

    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            # 1. rows still needing ground truth
            cur.execute(
                "SELECT ocid, predicted_cpv FROM classifications "
                "WHERE paradigm = %s AND ground_truth IS NULL",
                (paradigm,),
            )
            pending = cur.fetchall()
            print(f"[{paradigm}] {len(pending)} rows to score")

            # 2. join with the dataset on ocid
            gt = build_ground_truth()
            print(f"[{paradigm}] loaded {len(gt)} ground-truth CPV codes from dataset")

            # 3. per-message score: is_correct = predicted == truth (NULL if no prediction)
            updates = []
            no_truth = 0
            for ocid, predicted in pending:
                truth = gt.get(ocid)
                if truth is None:
                    no_truth += 1
                    continue
                is_correct = None if predicted is None else (predicted == truth)
                updates.append((truth, is_correct, ocid, paradigm))

            execute_batch(
                cur,
                "UPDATE classifications SET ground_truth = %s, is_correct = %s "
                "WHERE ocid = %s AND paradigm = %s",
                updates,
            )
            print(f"[{paradigm}] scored {len(updates)} rows "
                  f"({no_truth} had no ground truth in the dataset)")

            # 4. aggregate into metrics (accuracy as a 0–1 fraction) for the ui-chart
            cur.execute(
                """
                INSERT INTO metrics (paradigm, accuracy, avg_latency, total_records, run_at)
                SELECT %s,
                       ROUND(AVG(is_correct::int)::numeric, 4),
                       ROUND(AVG(latency_ms)::numeric, 2),
                       COUNT(*),
                       NOW()
                FROM classifications WHERE paradigm = %s
                ON CONFLICT (paradigm) DO UPDATE SET
                    accuracy      = EXCLUDED.accuracy,
                    avg_latency   = EXCLUDED.avg_latency,
                    total_records = EXCLUDED.total_records,
                    run_at        = EXCLUDED.run_at
                """,
                (paradigm, paradigm),
            )

            cur.execute(
                "SELECT accuracy, avg_latency, total_records FROM metrics WHERE paradigm = %s",
                (paradigm,),
            )
            acc, lat, n = cur.fetchone()
            print(f"[{paradigm}] metrics: accuracy={acc} ({(acc or 0) * 100:.1f}%) "
                  f"avg_latency={lat}ms total_records={n}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
