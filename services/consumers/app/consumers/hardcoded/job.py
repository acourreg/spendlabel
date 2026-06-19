"""Hardcoded rule-based CPV classifier — PyFlink job.

Paradigm: deterministic keyword/regex rules mapping contract descriptions
to one of 15 CPV 2-digit categories. No ML involved.

Usage::

    python main.py --paradigm hardcoded

TODO:
    - Define a keyword → CPV-category lookup table (dict or decision tree).
    - Implement ``process_stream`` to apply rules to each record.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class HardcodedJob(BaseFlinkJob):
    """Rule-based CPV classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-hardcoded-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_hardcoded,
            group_id=settings.kafka.group_hardcoded,
        )

    def process_stream(self, stream: object) -> object:
        """Apply keyword/regex rules to classify each contract.

        TODO: implement rule-based classification logic.
        """
        raise NotImplementedError


def main() -> None:
    HardcodedJob().run()


if __name__ == "__main__":
    main()
