"""solver classifier — classification only (constraint-solver paradigm).

No ML, no training. Each contract is classified by compiling the declarative rules in
``dsl.py`` into a Z3 weighted-MaxSMT and solving it per record:

  - one Bool per CPV category, with a one-hot constraint (exactly one chosen);
  - HARD constraints from the DSL prune infeasible categories (term exclusions,
    requirements, value bands);
  - the per-category evidence score combines flat keyword hits (title ×3, desc ×1),
    **conjunction combos** (e.g. (vehicle|fleet) AND (maintenance|repair) -> Repair),
    **negative penalties** (anti-evidence that kills over-firing), and value bonuses
    — these last two are what a plain keyword-argmax cannot express;
  - the objective maximizes evidence×100 + prior, so a frequency prior only breaks
    ties; no-evidence falls back to the majority sector.

Reads (title, description, value_gbp) per item — value feeds the arithmetic bands.
"""

from __future__ import annotations

import re

from z3 import Optimize, Bool, Not, If, Sum, sat

from consumers.solver.dsl import RULES, FALLBACK

PARADIGM = "solver"


def _all_terms() -> set[str]:
    terms: set[str] = set()
    for r in RULES.values():
        terms |= set(r.get("soft", {}))
        terms |= set(r.get("penalty", {}))
        terms |= set(r.get("forbid_terms", []))
        terms |= set(r.get("require_terms", []))
        for combo in r.get("combo", []):
            for group in combo["when"]:
                terms |= set(group)
    return terms


_PATTERNS = {t: re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in _all_terms()}


def _cmp(v: float, op: str, thr: float) -> bool:
    return {">": v > thr, ">=": v >= thr, "<": v < thr, "<=": v <= thr}[op]


class Classifier:
    """Per-record Z3 MaxSMT over the DSL rules -> 2-digit CPV code."""

    def classify(self, items) -> list[str]:
        return [self._solve(it[0], it[1], it[2] if len(it) > 2 else None) for it in items]

    @staticmethod
    def _solve(title, description, value) -> str:
        title_hits = {t for t, p in _PATTERNS.items() if p.search((title or "").lower())}
        desc_hits = {t for t, p in _PATTERNS.items() if p.search((description or "").lower())}
        present = title_hits | desc_hits
        v = value if isinstance(value, (int, float)) else None

        ev: dict[str, int] = {}      # evidence score (drives the fallback decision)
        for c, r in RULES.items():
            s = 0
            for term, w in r.get("soft", {}).items():
                if term in title_hits:
                    s += w * 3
                elif term in desc_hits:
                    s += w
            for combo in r.get("combo", []):                      # conjunction bonus
                if all(any(t in present for t in group) for group in combo["when"]):
                    s += combo["bonus"]
            for term, w in r.get("penalty", {}).items():          # negative evidence
                if term in present:
                    s -= w
            for op, thr, bonus in r.get("value_soft", []):
                if v is not None and _cmp(v, op, thr):
                    s += bonus
            ev[c] = s

        opt = Optimize()
        x = {c: Bool(f"x_{c}") for c in RULES}
        opt.add(Sum([If(x[c], 1, 0) for c in RULES]) == 1)        # exactly one

        for c, r in RULES.items():
            forbidden = any(t in present for t in r.get("forbid_terms", []))
            req = r.get("require_terms")
            if req and not any(t in present for t in req):
                forbidden = True
            fv = r.get("forbid_value")
            if fv and v is not None and _cmp(v, fv[0], fv[1]):
                forbidden = True
            if forbidden:
                opt.add(Not(x[c]))

        # evidence dominates; prior (≈ frequency) only breaks ties
        opt.maximize(Sum([If(x[c], ev[c] * 100 + RULES[c].get("prior", 0), 0) for c in RULES]))

        if opt.check() == sat:
            m = opt.model()
            for c in RULES:
                if m[x[c]]:
                    return c if ev[c] > 0 else FALLBACK
        return FALLBACK


if __name__ == "__main__":
    clf = Classifier()
    print(clf.classify([
        ("fleet vehicle maintenance contract", "servicing and repair of council vans", 600_000),  # 50 not 34
        ("bespoke software development", "system integration and managed support", 900_000),       # 72 not 48
        ("highway design consultancy", "civil and structural engineering services", 1_200_000),    # 71 not 79
        ("supply of medical equipment", "surgical instruments and x-ray devices", 500_000),        # 33 not 85
    ]))
