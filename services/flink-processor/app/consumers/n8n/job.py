"""n8n webhook CPV classifier — PyFlink job.

Paradigm: an n8n workflow exposes a webhook endpoint that receives a contract
record, runs an internal classification flow (potentially involving an LLM,
lookup tables, or other n8n nodes), and returns the predicted CPV category.
This PyFlink job calls that webhook for each record.

Usage::

    python main.py --paradigm n8n

TODO:
    - Configure the n8n webhook URL (see ``n8n/README.md``).
    - Implement ``process_stream`` to POST each record to the webhook
      and parse the CPV prediction from the response.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class N8nJob(BaseFlinkJob):
    """n8n webhook-triggered CPV classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-n8n-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_n8n,
            group_id=settings.kafka.group_n8n,
        )

    def process_stream(self, stream: object) -> object:
        """Call n8n webhook for each contract record.

        TODO: implement webhook-based classification.
        """
        raise NotImplementedError


def main() -> None:
    N8nJob().run()


if __name__ == "__main__":
    main()
