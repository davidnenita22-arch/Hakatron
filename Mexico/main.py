import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

# Street-type abbreviations -> one canonical word
ABBR = {
    "av": "avenida", "ave": "avenida", "blvd": "boulevard", "bulevar": "boulevard",
    "c": "calle", "cjon": "callejon", "cda": "cerrada", "cerr": "cerrada",
    "priv": "privada", "prol": "prolongacion", "ret": "retorno", "peat": "peatonal",
    "cor": "corredor", "circ": "circuito", "calz": "calzada", "diag": "diagonal",
    "and": "andador", "int": "interior", "depto": "departamento", "edif": "edificio",
    "amp": "ampliacion", "cont": "continuacion", "pje": "pasaje", "psje": "pasaje",
    "cto": "circuito", "mz": "manzana", "lt": "lote",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def addr_tokens(a):
    a = re.sub(r"(\d{5})-\d{4}", r"\1", a)  # drop zip+4
    return frozenset(ABBR.get(t, t) for t in norm(a).split())


def name_tokens(first, last):
    return [t for t in norm(first + " " + last).split() if t]


def tok_sim(a, b):
    if a == b:
        return 1.0
    # initial like "c" vs "cesar"
    if len(a) == 1 or len(b) == 1:
        return 0.9 if a[0] == b[0] else 0.0
    return SequenceMatcher(None, a, b).ratio()


def name_sim(x, y):
    """Best order-free token alignment (handles first/last swap, typos, initials)."""
    if len(x) > len(y):
        x, y = y, x
    used, total = set(), 0.0
    for t in x:
        best, bj = 0.0, None
        for j, u in enumerate(y):
            if j in used:
                continue
            s = tok_sim(t, u)
            if s > best:
                best, bj = s, j
        if bj is not None:
            used.add(bj)
        total += best
    return total / max(len(x), 1)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def score(r1, r2):
    """Address is the strong signal (same zip + same street tokens). Names are noisy
    (typos, swaps, initials, truncation), so they only break ties and rescue borderline cases."""
    ad = jaccard(r1["addr"], r2["addr"])
    nm = name_sim(r1["name"], r2["name"])
    return 0.7 * ad + 0.3 * nm


def load(path):
    recs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.search(r"(\d{5})(?:-\d{4})?\s*$", r["address"])
            recs.append({
                "id": r["record_id"],
                "src": r["source_table"],
                "zip": m.group(1) if m else norm(r["address"])[-12:],
                "addr": addr_tokens(r["address"]),
                "name": name_tokens(r["first_name"], r["last_name"]),
            })
    return recs


def match(recs, threshold=0.5):
    blocks = defaultdict(list)
    for r in recs:
        blocks[r["zip"]].append(r)

    cands = []
    for group in blocks.values():
        for a, b in combinations(group, 2):
            if a["src"] == b["src"]:
                continue  # dups live in different tables
            s = score(a, b)
            if s >= threshold:
                cands.append((s, a, b))

    # Greedy 1-to-1 per table pair: each record links to at most one record of each other table
    cands.sort(key=lambda c: -c[0])
    linked = set()  # (record_id, other_table)
    parent = {}
    seen = set()

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for s, a, b in cands:
        if (a["id"], b["src"]) in linked or (b["id"], a["src"]) in linked:
            continue
        linked.add((a["id"], b["src"]))
        linked.add((b["id"], a["src"]))
        parent[find(a["id"])] = find(b["id"])
        seen.update((a["id"], b["id"]))

    # Emit all pairs inside each cluster (transitive closure)
    clusters = defaultdict(list)
    for x in seen:
        clusters[find(x)].append(x)
    pairs = []
    for members in clusters.values():
        srcs = [m[0] for m in members]
        if len(set(srcs)) != len(srcs):
            continue  # shouldn't happen given 1-to-1 rule
        for a, b in combinations(sorted(members), 2):
            pairs.append([a, b])
    return sorted(pairs)


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = next((p for p in ("test-input.csv", os.path.join(here, "test-input.csv"))
                       if os.path.exists(p)), "train-input.csv")
    inp = sys.argv[1] if len(sys.argv) > 1 else default_in
    out = sys.argv[2] if len(sys.argv) > 2 else "test-output.json"
    pairs = match(load(inp))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"matches": pairs}, f)
    print(f"{len(pairs)} matches -> {out}")


if __name__ == "__main__":
    main()
