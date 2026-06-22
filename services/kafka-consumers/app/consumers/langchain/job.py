"""langchain paradigm entrypoint — Gemini via LangChain (plain prompt), via the shared loop."""

from __future__ import annotations

from consumers.consumer_loop import run_consumer
from consumers.langchain.classifier import Classifier, PARADIGM


class LangChainJob:
    def run(self, idle_timeout: float = 15.0) -> None:
        run_consumer(PARADIGM, Classifier(), idle_timeout=idle_timeout)


def main() -> None:
    LangChainJob().run()


if __name__ == "__main__":
    main()
