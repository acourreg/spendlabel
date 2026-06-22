"""mcp classifier — Gemini (OpenAI-compatible endpoint) + in-process MCP tools.

The agentic paradigm: the model calls the MCP tools in ``server.py`` before
committing to a CPV division. Zero-shot, no training. Env config: GEMINI_API_KEY
(required), GEMINI_BASE_URL, GEMINI_MODEL, MCP_MAX_TOOL_ROUNDS, MCP_CONCURRENCY.
"""

from __future__ import annotations

import asyncio
import os
import time

from consumers.cpv_labels import FALLBACK_CPV, coerce_cpv

PARADIGM = "mcp"
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL = "gemini-2.5-flash-lite"

_SYSTEM = (
    "You classify a UK public contract into exactly ONE 2-digit CPV division. "
    "First call list_cpv_divisions to get the allowed codes (and keyword_candidates "
    "for hints if useful), then reply with ONLY the 2-digit code (e.g. 45). Always "
    "commit to the single best division; never refuse or ask for clarification."
)


def _to_openai_tools(tools):
    return [{"type": "function", "function": {
        "name": t.name, "description": (t.description or "").strip(),
        "parameters": t.inputSchema or {"type": "object", "properties": {}}}} for t in tools]


def _tool_text(result):
    # Prefer the text content blocks (already JSON); fall back to structured output.
    blocks = getattr(result, "content", None) or []
    text = "\n".join(getattr(b, "text", "") for b in blocks if getattr(b, "text", "")).strip()
    if text:
        return text
    import json
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        try:
            return json.dumps(sc)
        except TypeError:
            return str(sc)
    return ""


class Classifier:
    """Gemini drives MCP tools to pick a CPV division; the batch runs concurrently."""

    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", _MODEL)
        self.base_url = os.getenv("GEMINI_BASE_URL", _BASE_URL)
        self.max_rounds = int(os.getenv("MCP_MAX_TOOL_ROUNDS", "3"))
        self.concurrency = int(os.getenv("MCP_CONCURRENCY", "8"))
        self._lat: list[float] = []

    def classify(self, items):  # items: (title, description, value); value unused
        codes, self._lat = asyncio.run(self._batch(items))
        return codes

    def pop_latencies_ms(self):  # true per-request latencies (audit fix B6)
        lat, self._lat = self._lat, []
        return lat

    async def _batch(self, items):
        from fastmcp import Client
        from openai import AsyncOpenAI

        from consumers.mcp.server import mcp as server

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (source openclaw/load-secrets.sh or .env.docker).")
        sem = asyncio.Semaphore(self.concurrency)

        # async with closes the HTTP client before the event loop ends (avoids the
        # "Event loop is closed" cleanup warning when asyncio.run is called per batch).
        async with AsyncOpenAI(api_key=key, base_url=self.base_url) as client, \
                Client(server) as mcp:
            tools = _to_openai_tools(await mcp.list_tools())

            async def one(it):
                async with sem:
                    t0 = time.perf_counter()
                    code = await self._one(client, mcp, tools, it[0], it[1])
                    return code, (time.perf_counter() - t0) * 1000.0

            pairs = await asyncio.gather(*(one(it) for it in items))  # order preserved
            return [c for c, _ in pairs], [lat for _, lat in pairs]

    async def _one(self, client, mcp, tools, title, desc):
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Title: {title or ''}\nDescription: {desc or ''}".strip()}]
        try:
            for r in range(self.max_rounds):
                use_tools = r < self.max_rounds - 1  # last round forces an answer
                if not use_tools:
                    msgs.append({"role": "user", "content":
                                 "Reply with ONLY the 2-digit CPV code now. Pick the single best division; do not refuse."})
                resp = await client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=0, max_tokens=64,
                    tools=tools if use_tools else None,
                    tool_choice="auto" if use_tools else None)
                m = resp.choices[0].message
                if use_tools and m.tool_calls:
                    msgs.append({"role": "assistant", "content": m.content or "",
                                 "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
                    for tc in m.tool_calls:
                        msgs.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": await self._call(mcp, tc)})
                    continue
                return coerce_cpv(m.content)
        except Exception as e:  # network / rate-limit — never crash the run
            print(f"[mcp] {e}", flush=True)
        return FALLBACK_CPV

    async def _call(self, mcp, tc):
        import json
        try:
            args = json.loads(tc.function.arguments or "{}")
        except ValueError:
            args = {}
        try:
            return _tool_text(await mcp.call_tool(tc.function.name, args))
        except Exception as e:
            return f"tool error: {e}"


if __name__ == "__main__":
    print(Classifier().classify([("road construction works", "new carriageway", None)]))
