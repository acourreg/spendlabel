"""flow paradigm entrypoint — n8n webhook (Gemini inside), via the shared loop."""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.flow.classifier import Classifier, PARADIGM


class FlowJob:
    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    FlowJob().run()


if __name__ == "__main__":
    main()
