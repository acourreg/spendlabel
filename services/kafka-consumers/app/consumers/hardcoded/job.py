"""Hardcoded rule-based CPV classifier — Kafka consumer entrypoint.

Consumes ``cpv-raw`` via the shared consumer loop and the keyword classifier.
"""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.hardcoded.classifier import Classifier, PARADIGM


class HardcodedJob:
    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    HardcodedJob().run()


if __name__ == "__main__":
    main()
