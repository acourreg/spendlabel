"""flow classifier — POST each contract to an n8n workflow (Gemini inside).

The workflow-automation-platform paradigm: classification is delegated to n8n
(Webhook -> Gemini -> Respond); this consumer just calls the webhook. Env config:
FLOW_WEBHOOK_URL (required), FLOW_CONCURRENCY, FLOW_TIMEOUT. No LLM dep here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from consumers.cpv_labels import FALLBACK_CPV, coerce_cpv

PARADIGM = "flow"


class Classifier:
    """Call the n8n webhook per contract; parse a 2-digit CPV from the reply."""

    def __init__(self):
        self.url = os.getenv("FLOW_WEBHOOK_URL", "")
        self.concurrency = int(os.getenv("FLOW_CONCURRENCY", "8"))
        self.timeout = float(os.getenv("FLOW_TIMEOUT", "30"))
        self._lat: list[float] = []

    def classify(self, items):  # items: (title, description, value); value unused
        if not self.url:
            raise RuntimeError("FLOW_WEBHOOK_URL not set (see flow/README.md).")

        def timed(it):
            t0 = time.perf_counter()
            code = self._predict(it[0], it[1])
            return code, (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            pairs = list(pool.map(timed, items))  # map preserves input order
        self._lat = [lat for _, lat in pairs]
        return [c for c, _ in pairs]

    def pop_latencies_ms(self):  # true per-request latencies (audit fix B6)
        lat, self._lat = self._lat, []
        return lat

    def _predict(self, title, description):
        body = json.dumps({"title": title or "", "description": description or ""}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[flow] {e}", flush=True)
            return FALLBACK_CPV
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return coerce_cpv(str(data.get("cpv") or data.get("predicted_cpv") or text))
            if isinstance(data, list) and data:
                d0 = data[0]
                return coerce_cpv(str(d0.get("cpv") if isinstance(d0, dict) else d0))
        except ValueError:
            pass
        return coerce_cpv(text)


if __name__ == "__main__":
    print(Classifier().classify([("road construction works", "new carriageway", None)]))
