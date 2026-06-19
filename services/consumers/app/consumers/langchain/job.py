"""LangChain LLM agent CPV classifier — PyFlink job.

Paradigm: a simple LangChain agent backed by an LLM (e.g. GPT-4o-mini)
receives the contract description in a prompt and returns the predicted
CPV 2-digit category. Zero-shot or few-shot prompting.

Usage::

    python main.py --paradigm langchain

TODO:
    - Build a LangChain chain (prompt template → LLM → output parser).
    - Implement ``process_stream`` to invoke the chain per record.
    - Handle rate-limiting / retries for the LLM API.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class LangChainJob(BaseFlinkJob):
    """LangChain LLM agent CPV classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-langchain-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_langchain,
            group_id=settings.kafka.group_langchain,
        )

    def process_stream(self, stream: object) -> object:
        """Invoke LangChain LLM agent for each contract record.

        TODO: implement LLM-based classification.
        """
        raise NotImplementedError


def main() -> None:
    LangChainJob().run()


if __name__ == "__main__":
    main()
