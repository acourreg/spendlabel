"""PostgreSQL access layer for the consumers service.

A small synchronous psycopg2 connection pool (the consumers are sync), plus
helpers to apply the schema, insert one classification, and materialise the
per-paradigm ``metrics`` aggregate after a run.

Connection settings come from environment variables (see config/confluent.env.example):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
or a single DATABASE_URL (takes precedence).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.pool import SimpleConnectionPool

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: SimpleConnectionPool | None = None


def _dsn() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    return (
        f"host={os.getenv('POSTGRES_HOST', 'postgres')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'spendlabel')} "
        f"user={os.getenv('POSTGRES_USER', 'spendlabel')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'spendlabel')}"
    )


def get_pool() -> SimpleConnectionPool:
    """Return the process-wide connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX", "10")), dsn=_dsn())
    return _pool


@contextmanager
def get_conn():
    """Borrow a connection from the pool, committing on success."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_schema() -> None:
    """Apply schema.sql (idempotent — uses CREATE TABLE IF NOT EXISTS)."""
    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)


def upsert_progress(paradigm: str, processed: int, total: int, status: str) -> None:
    """Record live run progress (processed/total) for the ui-chart progress bars."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_progress (paradigm, processed, total, status, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (paradigm) DO UPDATE SET
                processed  = EXCLUDED.processed,
                total      = EXCLUDED.total,
                status     = EXCLUDED.status,
                updated_at = NOW()
            """,
            (paradigm, processed, total, status),
        )


def insert_classification(
    *,
    ocid: str,
    paradigm: str,
    predicted_cpv: str | None,
    ground_truth: str | None,
    latency_ms: float | None,
    value_gbp: float | None = None,
    supplier_name: str | None = None,
) -> None:
    """Insert a single processed message into ``classifications``."""
    is_correct = (
        None if predicted_cpv is None or ground_truth is None else predicted_cpv == ground_truth
    )
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO classifications
                (ocid, paradigm, predicted_cpv, ground_truth, is_correct,
                 latency_ms, value_gbp, supplier_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (ocid, paradigm, predicted_cpv, ground_truth, is_correct,
             latency_ms, value_gbp, supplier_name),
        )


def insert_classifications_batch(rows: list[tuple]) -> None:
    """Bulk-insert classification rows in one transaction.

    Each row: (ocid, paradigm, predicted_cpv, ground_truth, is_correct,
               latency_ms, value_gbp, supplier_name). Used by the embedding-based
    ML consumers, which classify in batches.
    """
    if not rows:
        return
    from psycopg2.extras import execute_batch

    with get_conn() as conn, conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO classifications
                (ocid, paradigm, predicted_cpv, ground_truth, is_correct,
                 latency_ms, value_gbp, supplier_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
            page_size=500,
        )


def materialize_metrics(paradigm: str) -> None:
    """Recompute and upsert the ``metrics`` aggregate for one paradigm.

    Call this after a run so the ui-chart service can read fresh accuracy /
    latency / count without scanning ``classifications``.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO metrics (paradigm, accuracy, avg_latency, total_records, run_at)
            SELECT
                %s AS paradigm,
                AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS accuracy,
                AVG(latency_ms) AS avg_latency,
                COUNT(*) AS total_records,
                NOW() AS run_at
            FROM classifications
            WHERE paradigm = %s
            ON CONFLICT (paradigm) DO UPDATE SET
                accuracy      = EXCLUDED.accuracy,
                avg_latency   = EXCLUDED.avg_latency,
                total_records = EXCLUDED.total_records,
                run_at        = EXCLUDED.run_at
            """,
            (paradigm, paradigm),
        )
