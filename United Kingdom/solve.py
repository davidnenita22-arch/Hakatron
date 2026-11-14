"""Call My BLEUff — catch the bluffers BLEU crowns, name the honest panellist, spell the flag.

Usage:  python3 solve.py
Needs:  references.csv, panel.csv, bleu.py, facts.py, coverage.py (this folder)
        data/e2e/{trainset,devset,testset_w_refs}.csv  (public E2E NLG dataset,
        https://github.com/tuetschek/e2e-dataset) — used for provenance labels.
Writes: decks.csv, cards.csv, bluff_vs_listener.png
"""
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coverage
from bleu import corpus_bleu, ngrams, tokenize

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
refs = pd.read_csv(HERE / "references.csv")
panel = pd.read_csv(HERE / "panel.csv")
R = refs.groupby(["round", "card"]).reference.apply(list).to_dict()


def decode(seats):
    return "".join(chr(65 + 9 * (seats[i] - 1) + 3 * (seats[i + 1] - 1) + (seats[i + 2] - 1)) for i in range(0, 24, 3))


# ---------------------------------------------------------------- 1. house BLEU
decks = []
for (rnd, seat), deck in panel.groupby(["round", "seat"]):
    res = corpus_bleu(list(deck.text), [R[(rnd, c)] for c in deck.card])
    decks.append({"round": rnd, "seat": seat, "bleu": res["bleu"], "bp": res["brevity_penalty"]})
decks = pd.DataFrame(decks)
bleu_pick = decks.loc[decks.groupby("round").bleu.idxmax()].sort_values("round").seat.tolist()
print("House BLEU crowns:", decode(bleu_pick), "(not the flag)")

# ---------------------------------------------------------------- 2. listener-check features per card
STOP = set("a an the is are and of in near with it its to that for at by has have this there on as be . ,".split())
CONS_STOP = STOP | set("serves serving located area food shop place".split())
END_DANGLE = set("a an the of and is in with out at to near by for has rated rating customer high low average "
                 "moderately moderate check range price".split())

# fluency: Kneser-Ney trigram LM on references of the OTHER rounds
def train_lm(texts, D=0.75):
    c3, c2 = Counter(), Counter()
    for s in texts:
        t = ["<s>", "<s>"] + tokenize(s) + ["</s>"]
        for i in range(2, len(t)):
            c3[(t[i - 2], t[i - 1], t[i])] += 1
            c2[(t[i - 1], t[i])] += 1
    kn1 = Counter(b for (_, b) in c2); tot1 = sum(kn1.values()); V = len(kn1) + 1
    ctx2, n2, ctx1, n1 = Counter(), Counter(), Counter(), Counter()
    for (a, b, _), v in c3.items(): ctx2[(a, b)] += v; n2[(a, b)] += 1
    for (a, _), v in c2.items(): ctx1[a] += v; n1[a] += 1
    p1 = lambda w: (kn1[w] + 0.5) / (tot1 + 0.5 * V)
    p2 = lambda a, w: p1(w) if not ctx1[a] else max(c2[(a, w)] - D, 0) / ctx1[a] + D * n1[a] / ctx1[a] * p1(w)
    p3 = lambda a, b, w: p2(b, w) if not ctx2[(a, b)] else max(c3[(a, b, w)] - D, 0) / ctx2[(a, b)] + D * n2[(a, b)] / ctx2[(a, b)] * p2(b, w)
    return p3

uni, bi = Counter(), Counter()
for s in refs.reference:
    t = ["<s>"] + tokenize(s) + ["</s>"]; uni.update(t); bi.update(zip(t, t[1:]))


def longest_common_run(a, b):
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1; best = max(best, cur[j])
        prev = cur
    return best


slots = {k: coverage.card_slots(v) for k, v in R.items()}
rows = []
lms = {}
for i, x in panel.iterrows():
    rnd, card, text = x["round"], x.card, x.text
    if rnd not in lms:
        lms[rnd] = train_lm(refs[refs["round"] != rnd].reference)
    t = tokenize(text); rt = [tokenize(s) for s in R[(rnd, card)]]
    g4 = ngrams(t, 4); rg4 = set().union(*[set(ngrams(q, 4)) for q in rt])
    words = Counter(w for q in rt for w in set(q))
    cons = {w for w, c in words.items() if c >= 3 and w not in CONS_STOP and w.isalnum()}
    lt = ["<s>", "<s>"] + t + ["</s>"]
    pairs = Counter(zip(t, t[1:]))
    rows.append(dict(
        round=rnd, card=card, seat=x.seat,
        lenratio=len(t) / np.mean([len(q) for q in rt]),                       # brevity / padding
        copy4=sum(1 for g in g4 if g in rg4) / max(1, sum(g4.values())),       # share of 4-grams lifted from refs
        span=max(longest_common_run(t, q) for q in rt) / len(t),               # longest verbatim run
        dup4=sum(c - 1 for c in g4.values()) / max(1, sum(g4.values())),       # repeated 4-grams
        repw=sum(1 for a, b in zip(t, t[1:]) if a == b),                       # "more than more than"
        dup2=sum(v - 1 for (a, b), v in pairs.items() if v > 1 and not (a in STOP and b in STOP)),  # restating
        splice=sum(1 for a, b in zip(["<s>"] + t, t + ["</s>"]) if bi[(a, b)] == 0 and uni[a] >= 20 and uni[b] >= 20),
        midcap=len(re.findall(r"[a-z] (It|The|There|They|This|Its|A|An|Near|Located)\b", text)),  # deleted words
        nsent=len(re.findall(r"[.!?](\s|$)", text)),
        dangle=int((re.findall(r"(\w+)\W*$", text.lower()) or [""])[0] in END_DANGLE),      # truncated mid-phrase
        recall=len(cons & set(t)) / max(1, len(cons)),                          # consensus content kept
        facts=coverage.coverage(text, slots[(rnd, card)]),                      # venue facts kept
        oov=sum(1 for w in t if w not in set().union(*map(set, rt))) / len(t),
        lm=np.mean([math.log(lms[rnd](lt[j - 2], lt[j - 1], lt[j])) for j in range(2, len(lt))]),
    ))
cards = pd.DataFrame(rows)
cards["text"] = panel.text.values
FEATS = ["lenratio", "copy4", "span", "dup4", "repw", "dup2", "splice", "midcap", "nsent", "dangle",
         "recall", "facts", "oov", "lm"]

# ---------------------------------------------------------------- 3. provenance labels (E2E originals)
E = pd.concat([pd.read_csv(HERE / "data" / "e2e" / f) for f in ["trainset.csv", "devset.csv", "testset_w_refs.csv"]])
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
originals = set(E.ref.map(norm))
cards["original"] = cards.text.map(norm).isin(originals)
prov = cards.groupby(["round", "seat"]).original.sum().unstack()
prov_pick = prov.idxmax(axis=1).tolist()
print("\nUnchanged E2E originals per deck (40 = every card untouched):")
print(prov.to_string())

# ---------------------------------------------------------------- 4. listener check (leave-one-round-out)
X = cards[FEATS].assign(lr2=(cards.lenratio - 1) ** 2)
y = (~cards.original).astype(int)
p = np.zeros(len(cards))
for rnd in range(1, 25):
    tr = cards["round"] != rnd
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
    p[~tr] = model.predict_proba(X[~tr])[:, 1]
cards["p_bluff"] = p
print(f"\nCard-level bluff AUC (leave-one-round-out): {roc_auc_score(y, p):.3f}")

# honest = every card good  ->  judge a deck by its two worst cards
listener = cards.groupby(["round", "seat"]).p_bluff.apply(lambda s: s.nlargest(2).mean()).rename("listener")
decks = decks.merge(listener.reset_index(), on=["round", "seat"])
decks["originals"] = [prov.loc[r, s] for r, s in zip(decks["round"], decks.seat)]
lis_pick = decks.loc[decks.groupby("round").listener.idxmin()].sort_values("round").seat.tolist()
agree = sum(a == b for a, b in zip(lis_pick, prov_pick))
print(f"Listener check names the same honest seat as provenance in {agree}/24 rounds")

cards.drop(columns="text").to_csv(HERE / "cards.csv", index=False)
decks.to_csv(HERE / "decks.csv", index=False)

# ---------------------------------------------------------------- 5. figure
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for seat, mk in zip([1, 2, 3], ["o", "s", "^"]):
    g = decks[decks.seat == seat]
    axs[0].scatter(g["round"], g.bleu, marker=mk, label=f"seat {seat}")
    axs[1].scatter(g["round"], g.listener, marker=mk)
    axs[2].scatter(g["round"], g.originals, marker=mk)
hon = decks[[s == prov_pick[r - 1] for r, s in zip(decks["round"], decks.seat)]]
for ax, col in zip(axs, ["bleu", "listener", "originals"]):
    ax.scatter(hon["round"], hon[col], s=160, facecolors="none", edgecolors="red", lw=1.5)
axs[0].set_ylabel("house BLEU\n(crowns a bluffer)")
axs[1].set_ylabel("listener check\n(worst-2 bluff prob, low = honest)")
axs[2].set_ylabel("E2E originals\n(of 40 cards)")
axs[0].legend(ncol=3, frameon=False)
axs[2].set_xticks(range(1, 25)); axs[2].set_xlabel("round  (red ring = honest panellist)")
fig.suptitle(f"BLEU picks {decode(bleu_pick)};  honesty spells {decode(lis_pick)}", fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "bluff_vs_listener.png", dpi=120)

word = decode(lis_pick)
print("\nHonest seats:", "".join(map(str, lis_pick)))
print("WORD:", word, " (provenance gives", decode(prov_pick) + ")")
print("FLAG:", f"SIGMOID_{word}")
