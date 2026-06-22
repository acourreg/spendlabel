"""Canonical CPV division catalogue — the ONE label space for every paradigm.

The 2-digit CPV division ("spend sector") is the closed set every consumer
classifies into. Single source of truth (audit fix B1) so paradigms are
comparable instead of each using a divergent set. Labels mirror the ui-chart.
"""

from __future__ import annotations

import re

# 2-digit CPV division -> short label. The closed universe of valid predictions.
CPV_DIVISIONS: dict[str, str] = {
    "03": "Agriculture, farming, fishing, forestry products",
    "09": "Petroleum, fuels, electricity, energy",
    "14": "Mining, metals, minerals",
    "15": "Food, beverages, tobacco",
    "18": "Clothing, footwear, luggage",
    "22": "Printed matter, publications",
    "24": "Chemical products",
    "30": "Office & computing machinery, equipment, supplies",
    "31": "Electrical machinery, apparatus, equipment",
    "32": "Radio, TV, communication, telecom equipment",
    "33": "Medical equipment, pharmaceuticals, personal care",
    "34": "Transport equipment & vehicles",
    "35": "Security, fire-fighting, police, defence equipment",
    "37": "Musical instruments, sport goods, games",
    "38": "Laboratory, optical & precision instruments",
    "39": "Furniture, furnishings, cleaning products",
    "41": "Collected & purified water",
    "42": "Industrial machinery",
    "43": "Mining & construction machinery",
    "44": "Construction structures & materials",
    "45": "Construction work",
    "48": "Software packages & information systems",
    "50": "Repair & maintenance services",
    "51": "Installation services",
    "55": "Hotel, restaurant, retail trade services",
    "60": "Transport services (road, rail, air, water)",
    "63": "Supporting transport services, travel agencies",
    "64": "Postal & telecommunications services",
    "65": "Public utilities (water, gas, electricity distribution)",
    "66": "Financial & insurance services",
    "70": "Real estate services",
    "71": "Architectural, engineering, surveying, inspection",
    "72": "IT services: consulting, software dev, internet",
    "73": "Research & development services",
    "75": "Administration, defence, social security services",
    "77": "Agricultural, horticultural, forestry services",
    "79": "Business services: legal, marketing, consulting, HR, security",
    "80": "Education & training services",
    "85": "Health & social work services",
    "90": "Sewage, refuse, cleaning, environmental services",
    "92": "Recreational, cultural, sporting services",
    "98": "Other community, social & personal services",
}

# Global majority CPV in the data (Construction). Every paradigm falls back to
# this when it cannot decide, so all paradigms ALWAYS predict — keeping the
# scoring denominator identical (correct / all), see audit fix B4.
FALLBACK_CPV = "45"

VALID_CODES = frozenset(CPV_DIVISIONS)


def render_catalogue() -> str:
    """Render the catalogue as a compact ``NN - label`` list for prompts/tools."""
    return "\n".join(f"{code} - {label}" for code, label in CPV_DIVISIONS.items())


def coerce_cpv(text: str | None) -> str:
    """Extract a valid 2-digit CPV division from free-form text.

    Returns the first two-digit token that is a known division, else the
    majority-class fallback. Used by every paradigm that parses a code out of a
    model/tool/webhook reply, so they all resolve into the same closed set.
    """
    if text:
        for tok in re.findall(r"\d{2}", text):
            if tok in VALID_CODES:
                return tok
    return FALLBACK_CPV
