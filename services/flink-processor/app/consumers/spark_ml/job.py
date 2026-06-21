"""spark_ml — classical ML CPV classifier (the one non-Flink consumer).

Unlike every other paradigm in this service, spark_ml does **not** run through the
Flink pipeline (it is *not* a ``BaseFlinkJob``). A RandomForest over MiniLM
embeddings is a batch-style classical model — wrapping it in a streaming Flink
topology would make no sense. So this paradigm consumes ``cpv-raw`` directly from
Kafka via the shared ``consumers/consumer_loop`` and the RandomForest classifier.

It lives alongside the Flink jobs under ``consumers/spark_ml/`` only for repo
symmetry; structurally it is the single consumer that stands apart.

Usage::

    python main.py --paradigm spark_ml
"""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.spark_ml.classifier import Classifier, PARADIGM


class SparkMLConsumer:
    """Direct Kafka consumer (no Flink): MiniLM embeddings -> RandomForest (ONNX)."""

    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    SparkMLConsumer().run()


if __name__ == "__main__":
    main()
