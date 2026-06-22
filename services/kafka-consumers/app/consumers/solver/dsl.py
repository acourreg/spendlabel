"""Domain-definition rules for the constraint-solver paradigm.

Each CPV category is a declarative rule (a tiny DSL):
  soft         : {term: weight}    flat evidence (title hit ×3, desc ×1)
  value_soft   : [(op, thr, bonus)] arithmetic bonus on value_gbp
  forbid_terms : [term]            HARD exclusion (infeasible if any present)
  require_terms: [term]            HARD requirement (infeasible unless >=1 present)
  forbid_value : (op, thr)         HARD exclusion on value_gbp
  prior        : int               lexicographic tie-break (≈ CPV frequency %)

classifier.py compiles these to a Z3 MaxSMT: one-hot choice, hard constraints prune
infeasible categories, objective = evidence×100 + prior (prior only breaks ties).

NB: an earlier version added conjunction "combo" bonuses and negative penalties. An
ablation on the 5000-record eval showed they did NOT help (29.4% vs 29.9% without —
within noise): hand-written compositional rules route the textbook cases but don't
generalise on messy real text. They were removed; the symbolic ceiling here is ~30%,
statistically tied with the keyword baseline. Breaking it needs the ML embedders.
"""

FALLBACK = "45"

RULES = {
    "45": {"label": "Construction", "prior": 14,
           "soft": {"construction": 4, "building": 3, "civil": 3, "refurbishment": 3,
                    "demolition": 3, "roofing": 2, "highway": 3, "carriageway": 3, "groundworks": 3},
           "value_soft": [(">=", 250_000, 1)],
           "forbid_terms": ["software", "licence"]},
    "48": {"label": "Software", "prior": 9,
           "soft": {"software": 4, "saas": 3, "platform": 2, "application": 2, "licence": 3,
                    "erp": 3, "crm": 3, "database": 2, "software licence": 4},
           "forbid_terms": ["construction", "catering"]},
    "72": {"label": "IT services", "prior": 7,
           "soft": {"helpdesk": 4, "managed service": 3, "cyber security": 4, "data centre": 3,
                    "hosting": 3, "it support": 4, "software development": 3, "system integration": 3},
           "forbid_terms": ["construction", "catering"]},
    "71": {"label": "Engineering", "prior": 7,
           "soft": {"engineering": 3, "surveying": 3, "architectural": 3, "architect": 3,
                    "design services": 3, "geotechnical": 3, "structural design": 3, "feasibility": 2}},
    "79": {"label": "Business services", "prior": 12,
           "soft": {"consultancy": 3, "advisory": 3, "recruitment": 3, "human resources": 3,
                    "audit": 3, "marketing": 3, "legal services": 3, "business support": 2}},
    "80": {"label": "Education", "prior": 4,
           "soft": {"education": 3, "school": 3, "university": 3, "teaching": 3, "curriculum": 3,
                    "training course": 3, "apprenticeship": 3, "tuition": 3}},
    "85": {"label": "Health", "prior": 4,
           "soft": {"healthcare": 3, "nursing": 3, "medical care": 3, "clinical": 3, "nhs": 3,
                    "dental": 3, "care home": 3, "mental health": 3, "social care": 3},
           "forbid_terms": ["medical equipment", "surgical", "x-ray"]},
    "60": {"label": "Transport", "prior": 8,
           "soft": {"transport": 3, "logistics": 3, "haulage": 3, "freight": 3, "courier": 3,
                    "bus service": 3, "rail": 2, "passenger transport": 3, "taxi": 3}},
    "34": {"label": "Vehicles", "prior": 4,
           "soft": {"vehicle": 3, "fleet": 3, "hgv": 3, "minibus": 3, "automobile": 3,
                    "rolling stock": 3, "locomotive": 3, "coach": 2, "van": 2, "truck": 2}},
    "90": {"label": "Environmental", "prior": 3,
           "soft": {"waste": 3, "recycling": 3, "street cleaning": 3, "grounds maintenance": 3,
                    "environmental": 3, "refuse": 3, "sewerage": 3, "sweeping": 3}},
    "30": {"label": "Office supplies", "prior": 2,
           "soft": {"laptop": 3, "desktop": 3, "printer": 3, "toner": 3, "stationery": 3,
                    "office equipment": 3, "photocopier": 3, "computer": 2, "monitor": 2},
           "forbid_value": (">", 1_000_000)},
    "33": {"label": "Medical equipment", "prior": 2,
           "soft": {"medical equipment": 4, "surgical": 3, "diagnostic": 3, "x-ray": 3,
                    "wheelchair": 3, "defibrillator": 3, "prosthetic": 3, "ppe": 2}},
    "44": {"label": "Building materials", "prior": 2,
           "soft": {"building materials": 4, "aggregate": 3, "timber": 3, "steel": 3, "piping": 3,
                    "fittings": 2, "fencing": 3, "scaffolding": 3, "plumbing": 2, "tactile paving": 3}},
    "50": {"label": "Repair & maintenance", "prior": 4,
           "soft": {"overhaul": 3, "spare parts": 3, "breakdown": 3, "maintenance": 3, "repair": 3,
                    "servicing": 3, "planned maintenance": 4, "reactive maintenance": 4}},
    "51": {"label": "Installation", "prior": 1,
           "soft": {"installation": 2, "commissioning": 2, "fit-out": 3, "fitting out": 3}},
    "55": {"label": "Catering", "prior": 1,
           "soft": {"catering": 4, "hospitality": 3, "accommodation": 3, "hotel": 3,
                    "meals": 3, "canteen": 3, "restaurant": 3}},
    "66": {"label": "Financial", "prior": 1,
           "soft": {"insurance": 4, "banking": 3, "pension": 3, "financial services": 4,
                    "actuarial": 3, "investment management": 3, "brokerage": 3}},
    "73": {"label": "Research", "prior": 3,
           "soft": {"research": 3, "r&d": 3, "clinical trial": 3, "scientific research": 3,
                    "feasibility study": 2}},
    "75": {"label": "Public administration", "prior": 2,
           "soft": {"public administration": 3, "regulatory": 3, "government service": 3,
                    "policy": 2, "statutory": 3, "registration service": 3}},
    "31": {"label": "Electrical", "prior": 2,
           "soft": {"electrical": 3, "cabling": 3, "lighting": 3, "luminaire": 3, "transformer": 3,
                    "switchgear": 3, "lightning conductor": 3, "battery": 2, "generator": 2}},
    "32": {"label": "Telecoms", "prior": 1,
           "soft": {"telecom": 3, "telecommunications": 3, "radio equipment": 3, "antenna": 3,
                    "broadcast": 3, "telephony": 3, "two-way radio": 3}},
    "35": {"label": "Security & defence", "prior": 1,
           "soft": {"security equipment": 3, "defence": 3, "police": 2, "military": 3,
                    "surveillance": 3, "body armour": 3, "fire fighting": 3, "alarm": 2, "cctv": 3}},
    "38": {"label": "Lab & instruments", "prior": 1,
           "soft": {"laboratory": 3, "microscope": 3, "measuring instrument": 3, "calibration": 3,
                    "optical": 2, "scientific instrument": 3, "analyser": 3, "sensor": 2}},
    "39": {"label": "Furniture", "prior": 1,
           "soft": {"furniture": 3, "seating": 3, "desk": 3, "chair": 2, "bedding": 3,
                    "mattress": 3, "shelving": 3, "storage unit": 3}},
    "98": {"label": "Community services", "prior": 2,
           "soft": {"community": 2, "charity": 3, "voluntary": 3, "cultural": 3,
                    "sport": 2, "leisure": 2}},
}
