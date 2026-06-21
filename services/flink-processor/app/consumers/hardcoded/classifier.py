"""hardcoded classifier — classification only.

The simplest paradigm: a keyword dictionary over ``title`` + ``description`` maps
each contract to a 2-digit CPV sector. No ML, no training step. Maps a batch of
texts to CPV codes (or None when nothing matches) — no Kafka, no DB (the shared
``consumers/consumer_loop`` drives I/O).
"""

from __future__ import annotations

import re

PARADIGM = "hardcoded"

KEYWORDS = {
    "45": ["construction", "building", "civil", "refurbishment", "infrastructure", "demolition", "roofing", "drainage"],
    "48": ["software", "licence", "saas", "platform", "application", "erp", "crm", "database"],
    "60": ["transport", "logistics", "haulage", "freight", "courier", "fleet", "vehicle", "rail"],
    "71": ["engineering", "surveying", "architectural", "design", "planning", "geotechnical"],
    "72": ["IT", "helpdesk", "network", "cyber", "cloud", "digital", "support", "datacentre", "infrastructure IT"],
    "73": ["research", "R&D", "laboratory", "testing", "analysis", "scientific"],
    "75": ["government", "public administration", "regulatory", "legal", "policy", "compliance"],
    "79": ["consultancy", "advisory", "business services", "recruitment", "HR", "training", "audit"],
    "80": ["education", "learning", "school", "university", "teaching", "curriculum"],
    "85": ["health", "care", "medical", "nursing", "clinical", "NHS", "pharmacy", "dental"],
    "90": ["waste", "environmental", "recycling", "cleaning", "facilities", "grounds"],
    "98": ["community", "social", "charity", "voluntary", "cultural", "sport"],
    "34": [
        "vehicle", "fleet", "bus", "coach", "van", "truck", "car",
        "automobile", "transport vehicle", "road transport equipment",
        "rail vehicle", "tram", "locomotive", "motor",
    ],
    "35": [
        "security", "defence", "police", "military", "surveillance",
        "body armour", "protective equipment", "alarm system",
        "fire fighting", "safety equipment", "emergency", "detection",
    ],
}

# Pre-compiled word-boundary patterns (lowercased). Whole-word matching avoids
# short keywords like "it" matching inside unrelated words (e.g. "site").
_PATTERNS = {
    code: [re.compile(r"\b" + re.escape(kw.lower()) + r"\b") for kw in kws]
    for code, kws in KEYWORDS.items()
}


class Classifier:
    """Keyword rules over title+description -> 2-digit CPV code (or None)."""

    def classify(self, texts: list[str]) -> list[str | None]:
        return [self._classify_one(t) for t in texts]

    @staticmethod
    def _classify_one(text: str | None) -> str | None:
        text = (text or "").lower()
        best_code: str | None = None
        best_score = 0
        for code, patterns in _PATTERNS.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > best_score:  # strict > keeps the earliest code on ties
                best_score = score
                best_code = code
        return best_code


if __name__ == "__main__":
    clf = Classifier()
    print(clf.classify(["road resurfacing and drainage works", "NHS community nursing"]))
