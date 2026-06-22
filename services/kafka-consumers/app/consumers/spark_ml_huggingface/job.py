"""spark_ml (HuggingFace) — classical ML CPV classifier — Kafka consumer entrypoint.

A RandomForest over MiniLM embeddings (ONNX), consuming ``cpv-raw`` via the shared
consumer loop.

Usage::

    python main.py --paradigm spark_ml_huggingface
"""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.spark_ml_huggingface.classifier import Classifier, PARADIGM


class SparkMLConsumer:
    """MiniLM embeddings -> RandomForest (ONNX) over cpv-raw."""

    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    SparkMLConsumer().run()


if __name__ == "__main__":
    main()
