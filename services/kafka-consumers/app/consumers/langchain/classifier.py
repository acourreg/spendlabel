"""langchain classifier — Gemini via LangChain (plain prompt, no tools).

The framework paradigm: a LangChain prompt -> Gemini (OpenAI-compatible endpoint)
-> output parser. Zero-shot, no tools and no training — the contrast point against
`mcp` (agentic tools) and `flow` (n8n), all on the same model. Env config:
GEMINI_API_KEY (required), GEMINI_BASE_URL, GEMINI_MODEL, LANGCHAIN_CONCURRENCY.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from consumers.cpv_labels import FALLBACK_CPV, coerce_cpv, render_catalogue

PARADIGM = "langchain"
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL = "gemini-2.5-flash-lite"

_SYSTEM = (
    "Classify the UK public contract into exactly ONE 2-digit CPV division from the "
    "list. Reply with ONLY the 2-digit code (e.g. 45); always commit, never refuse.\n\n"
    "Allowed divisions:\n" + render_catalogue()
)


class Classifier:
    """LangChain chain (prompt -> Gemini) mapping each contract to a CPV division."""

    def __init__(self):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (source openclaw/load-secrets.sh or .env.docker).")
        llm = ChatOpenAI(model=os.getenv("GEMINI_MODEL", _MODEL),
                         base_url=os.getenv("GEMINI_BASE_URL", _BASE_URL),
                         api_key=key, temperature=0, max_tokens=8)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("user", "Title: {title}\nDescription: {desc}")])
        self.chain = prompt | llm
        self.concurrency = int(os.getenv("LANGCHAIN_CONCURRENCY", "8"))
        self._lat: list[float] = []

    def classify(self, items):  # items: (title, description, value); value unused
        def timed(it):
            t0 = time.perf_counter()
            try:
                out = self.chain.invoke({"title": it[0] or "", "desc": it[1] or ""})
                code = coerce_cpv(out.content)
            except Exception as e:  # network / rate-limit — never crash the run
                print(f"[langchain] {e}", flush=True)
                code = FALLBACK_CPV
            return code, (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            pairs = list(pool.map(timed, items))  # map preserves input order
        self._lat = [lat for _, lat in pairs]
        return [c for c, _ in pairs]

    def pop_latencies_ms(self):  # true per-request latencies (audit fix B6)
        lat, self._lat = self._lat, []
        return lat


if __name__ == "__main__":
    print(Classifier().classify([("road construction works", "new carriageway", None)]))
