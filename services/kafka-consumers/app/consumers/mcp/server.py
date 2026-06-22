"""MCP tool server for the agentic CPV classifier (FastMCP).

Tools the model may call while classifying: the CPV catalogue and a keyword hint.
Consumed in-process by classifier.py; also runnable over stdio via
``python -m consumers.mcp.server``.
"""

from __future__ import annotations

from fastmcp import FastMCP

from consumers.cpv_labels import CPV_DIVISIONS

mcp = FastMCP("spendlabel-cpv")

_HINTS = {
    "45": ("construction", "building", "refurbishment", "highway", "demolition"),
    "44": ("materials", "timber", "steel", "aggregate", "fencing"),
    "48": ("software", "licence", "saas", "application", "system"),
    "72": ("it support", "hosting", "cyber", "data centre", "integration"),
    "71": ("engineering", "architect", "surveying", "design", "feasibility"),
    "60": ("transport", "haulage", "freight", "bus", "rail"),
    "34": ("vehicle", "fleet", "van", "hgv", "minibus"),
    "79": ("consultancy", "recruitment", "audit", "marketing", "legal"),
    "80": ("school", "training", "education", "apprenticeship", "curriculum"),
    "85": ("healthcare", "nhs", "nursing", "care home", "mental health"),
    "90": ("waste", "recycling", "cleaning", "grounds maintenance", "environmental"),
    "50": ("repair", "maintenance", "servicing", "overhaul", "breakdown"),
    "33": ("medical equipment", "surgical", "diagnostic", "ppe", "pharmaceutical"),
    "30": ("computer", "laptop", "printer", "stationery", "office equipment"),
    "55": ("catering", "hospitality", "accommodation", "meals", "canteen"),
    "66": ("insurance", "banking", "pension", "financial", "actuarial"),
}


@mcp.tool
def list_cpv_divisions() -> dict[str, str]:
    """Valid CPV divisions (2-digit code -> label); the prediction MUST be one key."""
    return dict(CPV_DIVISIONS)


@mcp.tool
def keyword_candidates(text: str) -> list[dict]:
    """Heuristic shortlist of likely divisions for ``text`` (a hint, not the answer)."""
    low = (text or "").lower()
    scored = sorted(((sum(kw in low for kw in kws), c) for c, kws in _HINTS.items()), reverse=True)
    return [{"code": c, "label": CPV_DIVISIONS[c]} for n, c in scored if n][:5]


if __name__ == "__main__":
    mcp.run()
