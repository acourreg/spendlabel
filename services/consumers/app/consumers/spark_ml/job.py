"""Spark ML micro-batch CPV classifier — PyFlink job.

Paradigm: traditional ML pipeline (TF-IDF + logistic regression or random
forest) trained offline with PySpark, invoked here for inference on each
incoming contract record.

Usage::

    python main.py --paradigm spark_ml

TODO:
    - Load a pre-trained Spark ML PipelineModel from disk.
    - Implement ``process_stream`` to run inference per record.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class SparkMLJob(BaseFlinkJob):
    """Spark ML (TF-IDF + classifier) CPV predictor."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-sparkml-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_sparkml,
            group_id=settings.kafka.group_sparkml,
        )

    def process_stream(self, stream: object) -> object:
        """Run Spark ML inference on each contract record.

        TODO: implement micro-batch ML classification.
        """
        raise NotImplementedError


def main() -> None:
    SparkMLJob().run()


if __name__ == "__main__":
    main()
