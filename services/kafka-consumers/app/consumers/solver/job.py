"""Constraint-solver CPV classifier — job entrypoint.

Wires the shared consumer loop to the Z3 MaxSMT solver classifier (DSL rules ->
per-record constraint solve). No ML, no embeddings.

Usage::

    python main.py --paradigm solver
"""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.solver.classifier import Classifier, PARADIGM


class Job:
    """Per-record constraint-solver consumer (Z3 weighted-MaxSMT over DSL rules)."""

    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    Job().run()


if __name__ == "__main__":
    main()
