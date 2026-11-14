import re
from collections import Counter
import facts as F

CAP = re.compile(r"\b(?:The |the )?(?:[A-Z][\w'’é-]*)(?: (?:[A-Z][\w'’é-]*|by|of|Indian|Plaza))*")
GENERIC = {"There", "It", "This", "They", "Near", "Located", "A", "An", "The", "In", "If", "For", "With", "Its", "Customers",
           "Japanese", "Italian", "French", "Chinese", "Indian", "English", "British", "Fast", "Riverside", "City", "Children",
           "Family", "Kids", "Average", "High", "Low", "Moderate", "Cheap", "Expensive", "Rated", "Customer", "Families", "Yes"}

def proper(text):
    out = set()
    for m in CAP.findall(text):
        m = re.sub(r"^(the|The) ", "", m).strip()
        if m and m.split()[0] not in GENERIC and len(m) > 2:
            out.add(m.lower())
    return out

def card_slots(refs):
    """Facts a listener expects, read only from the references."""
    names = Counter(n for r in refs for n in proper(r))
    ents = {n for n, c in names.items() if c >= 3}
    return ents, F.card_facts(refs)

def coverage(text, slots):
    ents, facts = slots
    t = text.lower(); vals = F.values(text)
    hits = [e in t for e in ents] + [s in vals and bool(vals[s] & v) for s, v in facts.items()]
    return sum(hits) / max(1, len(hits))
