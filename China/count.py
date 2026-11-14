"""Count each crate's glyph in every set with the trained CenterNet.

    python3 count.py --data data --model model.pt --thr 0.3
Writes counts.csv (crate, set, glyph_id, count) and prints the per-crate count vectors.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from centernet import Net, heatmap, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", type=Path, default=Path("model.pt"))
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("counts.csv"))
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.model, map_location="cpu")
    classes = ck["classes"]
    model = Net(len(classes) + 1); model.load_state_dict(ck["state"]); model.to(dev).eval()

    crates = pd.read_csv(a.data / "crates.csv")
    rows = []
    for crate, gid in zip(crates.crate, crates.glyph_id):
        k = classes.index(gid)
        vec = []
        for set_dir in sorted((a.data / "crates" / crate).iterdir()):
            n = 0
            for f in sorted(set_dir.glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                n += len(peaks(heatmap(model, img, dev)[k], a.thr, *img.shape))
            rows.append([crate, set_dir.name, gid, n]); vec.append(n)
        print(crate, gid, vec, flush=True)
    pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv(a.out, index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
