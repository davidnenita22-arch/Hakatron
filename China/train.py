"""Train the glyph-counting CenterNet on the labelled rubbings.

    python3 train.py --data data --epochs 60
Writes model.pt (and prints per-glyph count error on held-out rubbings).
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from centernet import Net, TrainSet, focal_loss, heatmap, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val", type=int, default=200, help="held-out rubbings (0 = train on all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("model.pt"))
    a = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    labels = pd.read_csv(a.data / "train_labels.csv")
    classes = pd.read_csv(a.data / "crates.csv").glyph_id.tolist()
    imgs = sorted(labels.image.unique()); random.Random(0).shuffle(imgs)
    val_imgs = set(imgs[:a.val]); tr = labels[~labels.image.isin(val_imgs)]
    ds = TrainSet(a.data / "train" / "images", tr, classes)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=a.workers > 0)

    model = Net(len(classes) + 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * len(dl))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = focal_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"epoch {ep + 1:3d}  loss {tot / len(dl):.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "classes": classes}, a.out)

    if a.val:
        model.eval(); va = labels[labels.image.isin(val_imgs)]
        res = {c: {thr: [0, 0, 0] for thr in (0.2, 0.3, 0.4, 0.5)} for c in classes}
        for name, g in va.groupby("image"):
            img = cv2.imread(str(a.data / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
            hm = heatmap(model, img, dev)
            for k, c in enumerate(classes):
                true = int((g.glyph_id == c).sum())
                for thr in res[c]:
                    pred = len(peaks(hm[k], thr, *img.shape))
                    res[c][thr][0] += pred; res[c][thr][1] += true; res[c][thr][2] += abs(pred - true)
        print("held-out:  predicted/true (sum |per-image error|)  at thr 0.2 0.3 0.4 0.5")
        for c in classes:
            print(c, "  ".join(f"{p}/{t}({e})" for p, t, e in res[c].values()))


if __name__ == "__main__":
    main()
