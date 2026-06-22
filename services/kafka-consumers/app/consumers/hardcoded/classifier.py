"""hardcoded classifier — classification only.

The simplest paradigm: keyword rules over ``title`` + ``description`` map each
contract to a 2-digit CPV sector. No ML, no training step. ``classify`` takes a
batch of ``(title, description)`` pairs and returns CPV codes.

Three deliberate bits of added complexity (vs a flat keyword bag):
  - Broad CPV coverage: rules for 25 sectors, incl. the equipment/materials/
    services divisions that were previously unwinnable (30/31/32/33/38/39/44/50/
    51/55/66).
  - Title weighting: a keyword hit in the title counts 3×, in the description 1×
    (the title carries the strongest signal).
  - Fallback: when nothing matches, predict the global majority sector (CPV 45,
    Construction) instead of abstaining — so every record gets a prediction and
    the paradigm is comparable to the always-predicting ML heads.
"""

from __future__ import annotations

import re

PARADIGM = "hardcoded"

# Global majority CPV in the data (Construction, ~14%) — used when no keyword hits.
FALLBACK_CPV = "45"

# Title hits weigh more than description hits.
TITLE_WEIGHT = 3
DESC_WEIGHT = 1

KEYWORDS = {
    "45": ["construction", "building", "civil", "refurbishment", "demolition", "roofing", "highway", "carriageway", "groundworks"],
    "48": ["software", "licence", "saas", "platform", "application", "erp", "crm", "database", "software licence"],
    "60": ["transport", "logistics", "haulage", "freight", "courier", "bus service", "rail", "passenger transport", "taxi"],
    "71": ["engineering", "surveying", "architectural", "architect", "design services", "planning", "geotechnical", "feasibility", "structural design"],
    "72": ["helpdesk", "managed service", "cyber security", "data centre", "hosting", "it support", "software development", "system integration"],
    "73": ["research", "r&d", "laboratory study", "clinical trial", "scientific research", "feasibility study"],
    "75": ["public administration", "regulatory", "government service", "policy", "statutory", "registration service"],
    "79": ["consultancy", "advisory", "recruitment", "human resources", "audit", "marketing", "legal services", "business support"],
    "80": ["education", "school", "university", "teaching", "curriculum", "training course", "apprenticeship", "tuition"],
    "85": ["healthcare", "nursing", "medical care", "clinical", "nhs", "dental", "care home", "mental health", "social care"],
    "90": ["waste", "recycling", "street cleaning", "grounds maintenance", "environmental", "refuse", "sewerage", "sweeping"],
    "98": ["community", "charity", "voluntary", "cultural", "sport", "leisure"],
    "34": ["vehicle", "fleet", "bus", "coach", "van", "truck", "hgv", "minibus", "automobile", "rolling stock", "locomotive"],
    "35": ["security equipment", "defence", "police", "military", "surveillance", "body armour", "fire fighting", "alarm", "smoke alarm", "cctv"],
    # --- previously uncovered divisions (coverage win) ---
    "30": ["computer", "laptop", "desktop", "printer", "toner", "stationery", "office equipment", "photocopier", "monitor"],
    "31": ["electrical", "cabling", "lighting", "luminaire", "transformer", "switchgear", "battery", "generator", "lightning conductor"],
    "32": ["telecom", "telecommunications", "radio equipment", "antenna", "broadcast", "telephony", "two-way radio", "network equipment"],
    "33": ["medical equipment", "surgical", "diagnostic", "x-ray", "wheelchair", "defibrillator", "prosthetic", "pharmaceutical", "ppe"],
    "38": ["laboratory", "microscope", "measuring instrument", "calibration", "optical", "scientific instrument", "analyser", "sensor"],
    "39": ["furniture", "seating", "desk", "chair", "bedding", "mattress", "shelving", "storage unit"],
    "44": ["building materials", "aggregate", "timber", "steel", "piping", "fittings", "fencing", "scaffolding", "plumbing", "tactile paving"],
    "50": ["repair", "maintenance", "servicing", "overhaul", "spare parts", "breakdown", "planned maintenance", "reactive maintenance"],
    "51": ["installation", "commissioning", "fit-out", "fitting out"],
    "55": ["catering", "hospitality", "accommodation", "hotel", "meals", "canteen", "restaurant"],
    "66": ["insurance", "banking", "pension", "financial services", "actuarial", "investment management", "brokerage"],
}

# Pre-compiled word-boundary patterns (lowercased). Whole-word matching avoids
# short keywords like "it" matching inside unrelated words (e.g. "site").
_PATTERNS = {
    code: [re.compile(r"\b" + re.escape(kw.lower()) + r"\b") for kw in kws]
    for code, kws in KEYWORDS.items()
}


class Classifier:
    """Keyword rules over title (×3) + description (×1) -> 2-digit CPV code."""

    def classify(self, items: list[tuple[str | None, str | None]]) -> list[str]:
        return [self._classify_one(it[0], it[1]) for it in items]

    @staticmethod
    def _classify_one(title: str | None, description: str | None) -> str:
        title_l = (title or "").lower()
        desc_l = (description or "").lower()
        best_code: str | None = None
        best_score = 0
        for code, patterns in _PATTERNS.items():
            score = 0
            for p in patterns:
                if p.search(title_l):
                    score += TITLE_WEIGHT
                if p.search(desc_l):
                    score += DESC_WEIGHT
            if score > best_score:  # strict > keeps the earliest code on ties
                best_score = score
                best_code = code
        return best_code if best_code is not None else FALLBACK_CPV


if __name__ == "__main__":
    clf = Classifier()
    print(clf.classify([("road construction works", "new carriageway"),
                        ("annual boiler maintenance", "reactive repair of heating"),
                        ("supply of laptops", "office computers and monitors")]))
