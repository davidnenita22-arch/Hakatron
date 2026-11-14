"""Usage: python score.py output.json train-output.json"""
import json
import sys


def load(p):
    return {frozenset(m) for m in json.load(open(p))["matches"]}


pred, gold = load(sys.argv[1]), load(sys.argv[2])
tp = len(pred & gold)
p = tp / len(pred) if pred else 0
r = tp / len(gold) if gold else 0
f1 = 2 * p * r / (p + r) if p + r else 0
print(f"pred={len(pred)} gold={len(gold)} tp={tp} precision={p:.4f} recall={r:.4f} F1={f1:.4f}")
