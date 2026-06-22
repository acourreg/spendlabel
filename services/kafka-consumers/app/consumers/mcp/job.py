"""mcp paradigm entrypoint — agentic Gemini + MCP tools, via the shared loop."""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.mcp.classifier import Classifier, PARADIGM


class McpJob:
    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    McpJob().run()


if __name__ == "__main__":
    main()
