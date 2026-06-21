"""Hardcoded rule-based CPV classifier — job entrypoint.

Wires the shared consumer loop to the hardcoded keyword classifier. ``run()``
drives Kafka -> classify -> PostgreSQL via ``consumers/consumer_loop``;
``process_stream`` is the placeholder for a future PyFlink pipeline.

Usage::

    python main.py --paradigm hardcoded
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings
from consumers.consumer_loop import run_consumer
from consumers.hardcoded.classifier import Classifier, PARADIGM


class HardcodedJob(BaseFlinkJob):
    """Rule-based CPV classifier, backed by the keyword classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-hardcoded-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_hardcoded,
            group_id=settings.kafka.group_hardcoded,
        )

    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)

    def process_stream(self, stream):
        """Classify per record — exposed for a future PyFlink pipeline."""
        raise NotImplementedError


def main() -> None:
    HardcodedJob().run()


if __name__ == "__main__":
    main()
