"""Deep-learning ONNX CPV classifier — job entrypoint.

Wires the shared consumer loop to the MLP classifier (MiniLM embeddings -> MLP,
run via onnxruntime). ``run()`` drives Kafka -> classify -> PostgreSQL via
``consumers/consumer_loop``; ``process_stream`` is the placeholder for a future
PyFlink pipeline (this paradigm is meant to run through Flink, unlike spark_ml).

Usage::

    python main.py --paradigm deeplearning_onnx
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings
from consumers.consumer_loop import run_consumer
from consumers.deeplearning_onnx.classifier import Classifier, PARADIGM


class DeepLearningJob(BaseFlinkJob):
    """Deep-learning CPV classifier (MiniLM embeddings -> MLP)."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-deeplearning-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_deeplearning,
            group_id=settings.kafka.group_deeplearning,
        )

    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)

    def process_stream(self, stream):
        """ONNX inference per record — exposed for a future PyFlink pipeline."""
        raise NotImplementedError


def main() -> None:
    DeepLearningJob().run()


if __name__ == "__main__":
    main()
