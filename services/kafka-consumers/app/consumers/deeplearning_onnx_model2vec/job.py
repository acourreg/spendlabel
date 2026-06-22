"""Deep-learning CPV classifier (Model2Vec embeddings -> MLP, ONNX) — Kafka consumer entrypoint.

Consumes ``cpv-raw`` via the shared consumer loop; inference runs on onnxruntime.
"""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.deeplearning_onnx_model2vec.classifier import Classifier, PARADIGM


class Job:
    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    Job().run()


if __name__ == "__main__":
    main()
