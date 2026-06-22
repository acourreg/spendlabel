"""ui-chart service — FastAPI + Jinja2 dashboard.

Reads PostgreSQL only (never touches Kafka). Serves a single HTML page that
renders the benchmark metrics and a D3 sunburst from JSON APIs.

Run:  uvicorn main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from db.connection import get_conn

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="spendlabel ui-chart")

# CPV 2-digit category → human label.
CPV_LABELS = {
    "45": "Construction",
    "48": "Software",
    "60": "Transport",
    "71": "Engineering",
    "72": "IT Services",
    "73": "Research",
    "75": "Public admin",
    "79": "Business services",
    "80": "Education",
    "85": "Health",
    "90": "Environmental",
    "98": "Other community services",
    # extended divisions so fewer sectors fall back to "CPV NN"
    "09": "Petroleum",
    "15": "Food & beverages",
    "31": "Electrical equipment",
    "33": "Medical equipment",
    "38": "Lab instruments",
    "44": "Construction materials",
    "50": "Repair services",
    "55": "Hotel & catering",
    "63": "Transport support",
    "64": "Postal & telecoms",
    "66": "Financial services",
    "77": "Agricultural",
    "92": "Recreation",
}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """Per-paradigm aggregate. Returns [] if the DB is empty/unavailable."""
    # total exec = REAL wall-clock = span of processed_at (max-min). This is correct
    # for concurrent paradigms (the LLM ones run many calls in parallel, so the naive
    # avg_latency×records serial sum massively overstates it — e.g. 432min vs 47min real).
    sql = (
        "SELECT m.paradigm, m.accuracy, m.avg_latency, m.total_records, m.run_at, "
        "EXTRACT(EPOCH FROM (MAX(c.processed_at) - MIN(c.processed_at))) AS exec_seconds "
        "FROM metrics m LEFT JOIN classifications c ON c.paradigm = m.paradigm "
        "GROUP BY m.paradigm, m.accuracy, m.avg_latency, m.total_records, m.run_at "
        "ORDER BY m.accuracy DESC"
    )
    rows = _query(sql)
    out = []
    for paradigm, accuracy, avg_latency, total_records, run_at, exec_seconds in rows:
        avg_latency_ms = float(avg_latency) if avg_latency is not None else None
        records = int(total_records) if total_records is not None else 0
        span = float(exec_seconds) if exec_seconds is not None else 0.0
        if span > 0:
            total_exec_seconds = span                                   # real wall-clock
        elif avg_latency_ms is not None:
            total_exec_seconds = avg_latency_ms * records / 1000.0       # fallback (no rows)
        else:
            total_exec_seconds = None
        out.append(
            {
                "paradigm": paradigm,
                "accuracy": float(accuracy) if accuracy is not None else None,
                "avg_latency_ms": avg_latency_ms,
                "total_records": records,
                "total_exec_seconds": total_exec_seconds,
                "run_at": run_at.isoformat() if run_at is not None else None,
            }
        )
    return JSONResponse(out)


@app.get("/api/sunburst")
def api_sunburst(paradigm: str = Query(...)) -> JSONResponse:
    """Spend-by-CPV breakdown for one paradigm, grouped by what the paradigm
    PREDICTED (not the ground truth — that's identical across paradigms, so the
    pie must reflect each paradigm's own sector attribution). off_contract_spend
    = spend the paradigm placed in this sector but misclassified. Empty if no data.
    """
    sql = (
        "SELECT predicted_cpv AS cpv_code, "
        "SUM(value_gbp) AS total_spend, "
        "SUM(CASE WHEN is_correct = false THEN value_gbp ELSE 0 END) AS off_contract_spend, "
        "COUNT(*) AS record_count "
        "FROM classifications "
        "WHERE paradigm = %s "
        "GROUP BY predicted_cpv"
    )
    rows = _query(sql, (paradigm,))
    slices = []
    for cpv_code, total_spend, off_contract_spend, _record_count in rows:
        if cpv_code is None:
            continue
        slices.append(
            {
                "cpv_code": cpv_code,
                "label": CPV_LABELS.get(cpv_code, f"CPV {cpv_code}"),
                "spend": float(total_spend or 0.0),
                "off_contract": float(off_contract_spend or 0.0),
            }
        )
    slices.sort(key=lambda s: s["spend"], reverse=True)
    return JSONResponse({"paradigm": paradigm, "slices": slices})


@app.get("/api/progress")
def api_progress() -> JSONResponse:
    """Live per-paradigm run progress (processed/total). [] if not tracked yet."""
    rows = _query(
        "SELECT paradigm, processed, total, status, updated_at "
        "FROM run_progress ORDER BY paradigm"
    )
    out = []
    for paradigm, processed, total, status, updated_at in rows:
        processed = int(processed or 0)
        total = int(total or 0)
        out.append(
            {
                "paradigm": paradigm,
                "processed": processed,
                "total": total,
                "status": status,
                "pct": (processed / total * 100.0) if total else 0.0,
                "updated_at": updated_at.isoformat() if updated_at is not None else None,
            }
        )
    return JSONResponse(out)


def _query(sql: str, params: tuple = ()) -> list:
    """Run a read query, returning [] on any DB error (viz tool, not prod)."""
    try:
        conn = get_conn()
    except Exception:
        return []
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception:
        return []
    finally:
        conn.close()
