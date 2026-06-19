"""MCP (Model Context Protocol) + Claude tools CPV classifier — PyFlink job.

Paradigm: uses the Model Context Protocol to give Claude access to
domain-specific tools (CPV lookup, keyword extraction, budget analysis)
so it can classify contracts via structured tool use rather than raw
prompting.

Usage::

    python main.py --paradigm mcp

TODO:
    - Define MCP tools (cpv_lookup, extract_keywords, etc.).
    - Create an MCP server or connect to an existing one.
    - Implement ``process_stream`` to invoke Claude with tools per record.
    - Emit (contract_id, predicted_cpv, ground_truth_cpv, latency_ms).
"""

from __future__ import annotations

from consumers.base_job import BaseFlinkJob
from config import settings


class McpJob(BaseFlinkJob):
    """MCP + Claude tools CPV classifier."""

    def __init__(self) -> None:
        super().__init__(
            job_name="cpv-mcp-classifier",
            source_topic=settings.kafka.topic_raw,
            sink_topic=settings.kafka.topic_mcp,
            group_id=settings.kafka.group_mcp,
        )

    def process_stream(self, stream: object) -> object:
        """Invoke Claude via MCP tools for each contract record.

        TODO: implement MCP-based classification.
        """
        raise NotImplementedError


def main() -> None:
    McpJob().run()


if __name__ == "__main__":
    main()
