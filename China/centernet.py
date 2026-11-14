"""Tiny CenterNet for counting oracle-bone glyphs.

Heatmap head: one channel per crate glyph + one channel for "any other glyph".
Counting = number of local maxima above a threshold in a glyph's channel.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4


def conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.e1 = nn.Sequential(conv(1, 32, 2), conv(32, 32))          # /2
        self.e2 = nn.Sequential(conv(32, 64, 2), conv(64, 64))         # /4
        self.e3 = nn.Sequential(conv(64, 128, 2), conv(128, 128))      # /8
        self.e4 = nn.Sequential(conv(128, 192, 2), conv(192, 192))     # /16
        self.e5 = nn.Sequential(conv(192, 256, 2), conv(256, 256))     # /32
        self.u4 = conv(256 + 192, 192)
        self.u3 = conv(192 + 128, 128)
        self.u2 = conv(128 + 64, 96)
        self.head = nn.Sequential(conv(96, 96), nn.Conv2d(96, n_out, 1))
        self.head[-1].bias.data.fill_(-2.2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3); e5 = self.e5(e4)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        d4 = self.u4(up(e5, e4)); d3 = self.u3(up(d4, e3)); d2 = self.u2(up(d3, e2))
        return self.head(d2)  # logits at stride 4


def draw_gaussian(hm, cx, cy, w, h):
    r = max(1.0, 0.15 * math.sqrt(w * h) / STRIDE)
    H, W = hm.shape
    cx, cy = int(cx), int(cy)  # peak exactly 1.0 on a whole cell (CenterNet positive)
    x0, x1 = max(0, int(cx - 3 * r)), min(W, int(cx + 3 * r) + 1)
    y0, y1 = max(0, int(cy - 3 * r)), min(H, int(cy + 3 * r) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * r * r))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def normalise(img):
    img = img.astype(np.float32) / 255.0
    return (img - img.mean()) / (img.std() + 1e-6)


def pad32(a):
    H, W = a.shape[:2]
    return np.pad(a, ((0, (32 - H % 32) % 32), (0, (32 - W % 32) % 32)))


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, img_dir: Path, labels: pd.DataFrame, classes: list[str], crop=384, aug=True):
        self.img_dir, self.classes, self.crop, self.aug = img_dir, classes, crop, aug
        self.groups = list(labels.groupby("image"))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        name, g = self.groups[i]
        img = cv2.imread(str(self.img_dir / name), cv2.IMREAD_GRAYSCALE)
        boxes = g[["x", "y", "w", "h"]].values.astype(np.float32)
        cls = [self.classes.index(c) if c in self.classes else len(self.classes) for c in g.glyph_id]
        if self.aug:
            s = random.uniform(0.7, 1.3)
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            boxes *= s
            a = random.uniform(0.7, 1.3); b = random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        H, W = img.shape
        c = self.crop
        canvas = np.full((c, c), int(np.median(img)), np.uint8)
        ox = random.randint(min(0, W - c), max(0, W - c)) if self.aug else 0
        oy = random.randint(min(0, H - c), max(0, H - c)) if self.aug else 0
        sx0, sy0 = max(0, ox), max(0, oy); dx0, dy0 = max(0, -ox), max(0, -oy)
        cw, ch = min(W - sx0, c - dx0), min(H - sy0, c - dy0)
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]
        hm = np.zeros((len(self.classes) + 1, c // STRIDE, c // STRIDE), np.float32)
        for (x, y, w, h), k in zip(boxes, cls):
            cx, cy = (x + w / 2 - ox) / STRIDE, (y + h / 2 - oy) / STRIDE
            if 0 <= cx < c / STRIDE and 0 <= cy < c / STRIDE:
                draw_gaussian(hm[k], cx, cy, w, h)
        return torch.from_numpy(normalise(canvas))[None], torch.from_numpy(hm)


def focal_loss(logits, gt):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


@torch.no_grad()
def heatmap(model, img, device):
    x = torch.from_numpy(pad32(normalise(img)))[None, None].to(device)
    return torch.sigmoid(model(x))[0].cpu()


def peaks(hm_channel: torch.Tensor, thr: float, H: int, W: int):
    """Local maxima (3x3) above thr, inside the original image area."""
    h = hm_channel[None, None]
    keep = (h == F.max_pool2d(h, 5, 1, 2)) & (h >= thr)
    ys, xs = torch.nonzero(keep[0, 0], as_tuple=True)
    m = (ys * STRIDE < H) & (xs * STRIDE < W)
    return list(zip((xs[m] * STRIDE).tolist(), (ys[m] * STRIDE).tolist(), h[0, 0, ys[m], xs[m]].tolist()))
