"""ONNX runtime CPV classifier — PyFlink job.

Paradigm: a neural-network model (e.g. fine-tuned transformer) exported to
ONNX format, executed via onnxruntime for low-latency inference inside the
Flink pipeline.

Note: in production this could also run as a separate JVM Kafka Streams app;
here we wrap it in PyFlink for benchmark consistency.

Usage::

    python main.py --paradigm kstreams_onnx

TODO:
    - Load an ONNX model from disk (``onnxruntime.InferenceSession``).
    - Tokenise / vectorise contract text to match model input schema.
    - Implement ``process_stream`` to run ONNX inference per record.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class OnnxJob(BaseFlinkJob):
    """ONNX runtime CPV classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-onnx-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_onnx,
            group_id=settings.kafka.group_onnx,
        )

    def process_stream(self, stream: object) -> object:
        """Run ONNX inference on each contract record.

        TODO: implement ONNX-based classification.
        """
        raise NotImplementedError


def main() -> None:
    OnnxJob().run()


if __name__ == "__main__":
    main()
