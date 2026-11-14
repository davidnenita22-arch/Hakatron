import re
from collections import Counter

# slot -> value -> regex patterns (E2E-style venue facts)
SLOTS = {
    "price": {
        "low": [r"\bcheap", r"\blow[- ]?(cost|price|priced)", r"less than £ ?20", r"\binexpensive", r"\baffordabl", r"low price range", r"\bbudget"],
        "mid": [r"\bmoderate", r"\baverage[- ]priced", r"\bmid[- ]?(range|priced)", r"£ ?20 ?- ?(£ ?)?25", r"\breasonabl", r"average price"],
        "high": [r"\bexpensive", r"\bhigh[- ]?(price|priced|cost)", r"more than £ ?30", r"\bpricey", r"high price range", r"\bcostly"],
    },
    "rating": {
        "low": [r"\b1 out of 5", r"\bone star", r"\b1[- ]star", r"low (customer )?rat", r"\bpoor(ly)? rat", r"rated low", r"low rated", r"lowly"],
        "mid": [r"\b3 out of 5", r"\bthree star", r"\b3[- ]star", r"average (customer )?rat", r"rated average", r"\bmediocre"],
        "high": [r"\b5 out of 5", r"\bfive star", r"\b5[- ]star", r"high(ly)? (customer )?rat", r"rated high", r"\bexcellent rat", r"\btop rated"],
    },
    "food": {
        "japanese": [r"\bjapanese", r"\bsushi"], "italian": [r"\bitalian", r"\bpasta", r"\bpizza", r"\bspaghetti"],
        "french": [r"\bfrench"], "chinese": [r"\bchinese"], "indian": [r"\bindian", r"\bcurr(y|ies)\b"],
        "english": [r"\benglish", r"\bbritish"], "fast": [r"\bfast[- ]food", r"\bburger(s)?\b(?! king)", r"\bfries"],
    },
    "area": {"riverside": [r"\briver"], "centre": [r"city cent(re|er)", r"\bcent(re|er) of (the )?city", r"\bdowntown"]},
    "family": {"yes": [r"(family|kid|child|children)[- ]friendly", r"\bfamilies\b", r"\bkids?\b(?!.*not)"],
               "no": [r"\bnot (family|kid|child|children)", r"\bno (kids|children|families)", r"\badults? only", r"\bnon[- ]family", r"isn'?t (family|kid|child)"]},
}


def values(text):
    t = text.lower()
    found = {}
    for slot, vals in SLOTS.items():
        hit = {v for v, pats in vals.items() if any(re.search(p, t) for p in pats)}
        if slot == "family" and "no" in hit:
            hit = {"no"}
        if hit:
            found[slot] = hit
    return found


def card_facts(refs):
    """Majority value per slot over the references (slot kept if >=2 refs mention it)."""
    facts = {}
    for slot in SLOTS:
        c = Counter(v for r in refs for v in values(r).get(slot, ()))
        if c and sum(c.values()) >= 2:
            facts[slot] = {v for v, n in c.items() if n >= max(2, max(c.values()) * 0.5)}
    return facts


def contradictions(text, facts):
    out = []
    for slot, vals in values(text).items():
        if slot in facts and not (vals & facts[slot]):
            out.append(f"{slot}:{'/'.join(sorted(vals))}!={'/'.join(sorted(facts[slot]))}")
    return out
