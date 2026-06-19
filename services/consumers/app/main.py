"""Consumers service entrypoint.

Selects one PyFlink consumer paradigm via ``--paradigm`` and runs it. Each job
reads from Kafka (``cpv-raw``), classifies, and writes results to PostgreSQL
through ``db.connection``; the per-paradigm aggregate is materialised on exit.

Usage::

    python main.py --paradigm hardcoded
    python main.py --paradigm spark_ml
    ...
"""

from __future__ import annotations

import argparse

from consumers.hardcoded.job import HardcodedJob
from consumers.spark_ml.job import SparkMLJob
from consumers.kstreams_onnx.job import OnnxJob
from consumers.solver.job import SolverJob
from consumers.langchain.job import LangChainJob
from consumers.n8n.job import N8nJob
from consumers.mcp.job import McpJob
from db import connection

# paradigm flag value → PyFlink job class
PARADIGMS = {
    "hardcoded": HardcodedJob,
    "spark_ml": SparkMLJob,
    "kstreams_onnx": OnnxJob,
    "solver": SolverJob,
    "langchain": LangChainJob,
    "n8n": N8nJob,
    "mcp": McpJob,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a spendlabel consumer paradigm.")
    parser.add_argument(
        "--paradigm",
        required=True,
        choices=sorted(PARADIGMS),
        help="Which classification paradigm to run.",
    )
    args = parser.parse_args()

    # Ensure tables exist before the job starts writing.
    connection.init_schema()

    job = PARADIGMS[args.paradigm]()
    try:
        job.run()
    finally:
        # Roll up accuracy / latency / count for the ui-chart service.
        connection.materialize_metrics(args.paradigm)


if __name__ == "__main__":
    main()
