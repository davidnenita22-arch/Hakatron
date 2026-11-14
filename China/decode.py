"""Turn counts.csv into the flag.

Each crate's five set counts are either low (stray detections) or high (the glyph was
deliberately carved many times). High = 1, low = 0, sets 1..5 = most..least significant bit,
and the 5-bit number is a letter with A = 1.

    python3 decode.py counts.csv
"""
import sys

import numpy as np
import pandas as pd

counts = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "counts.csv")
word = ""
for crate, g in counts.groupby("crate", sort=True):
    c = g.sort_values("set")["count"].to_numpy().astype(int)
    s = np.sort(c)
    cut = np.argmax(np.diff(s))                       # split at the largest gap (2 clusters)
    thr = (s[cut] + s[cut + 1]) / 2
    bits = (c > thr).astype(int)
    value = int("".join(map(str, bits)), 2)
    letter = chr(64 + value) if 1 <= value <= 26 else "?"
    print(f"{crate}  {g.glyph_id.iloc[0]}  counts {c.tolist()}  split>{thr:.1f}  bits {''.join(map(str, bits))}  "
          f"= {value:2d} -> {letter}")
    word += letter
print("WORD:", word)
print("FLAG:", f"SIGMOID_{word}")
