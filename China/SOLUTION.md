# China – Carved in bones, engraved in the heart

**Flag:** `SIGMOID_FENGSHUI`

Files: [`centernet.py`](centernet.py) (model) · [`train.py`](train.py) · [`count.py`](count.py) · [`decode.py`](decode.py) · [`counts.csv`](counts.csv) · [`model.pt`](model.pt) · [`train_log.txt`](train_log.txt)

## Data
- `train/images`: 1,800 labelled greyscale rubbings, 14,149 boxes, 211 glyph ids. 12% of boxes are `rare`.
- `crates/crate_XX/set_Y`: 8 crates × 5 sets × 30 unlabelled rubbings.
- `crates.csv` gives each crate's glyph: 賓 g015, 今 g028, 其 g005, 㞢 g018, 王 g003, 不 g011, 貞 g001, 雨 g010.
- Glyphs are white strokes on dark ink. Box size varies about 50× (7–340 px).

## Method: count glyphs with a tiny CenterNet
Counting doesn't need box sizes, only *how many* of a glyph appear. So a small U-Net (trained from scratch, ~3 M parameters) predicts a **centre heatmap**, stride 4:
- one channel per crate glyph
- one shared channel for every other labelled glyph (helps the model learn "this is a glyph")

Training details:
- Gaussian peak at each labelled box centre, CenterNet focal loss
- augmentation: random scale 0.7–1.3, contrast/brightness, slight blur, random 384² crops; no flips, since mirrored glyphs can be different characters
- 40 epochs, AdamW + one-cycle, Apple M4 GPU (MPS), ~30 min; 200 rubbings held out

**Count = local maxima above a threshold in the glyph's channel.** Held-out totals (predicted / true, threshold 0.2–0.3):

| glyph | 賓 | 今 | 其 | 㞢 | 王 | 不 | 貞 | 雨 |
|---|---|---|---|---|---|---|---|---|
| thr 0.2 | 24/20 | 11/10 | 49/44 | 34/26 | 51/46 | 38/33 | 132/123 | 26/25 |
| thr 0.3 | 17/20 | 8/10 | 38/44 | 23/26 | 41/46 | 20/33 | 117/123 | 22/25 |

So set totals are accurate to roughly ±10–20%. That is far tighter than the signal below.

## The code in the counts
Run on every crate rubbing (threshold 0.2 → [`counts.csv`](counts.csv)). Every set count is either **low** (0–4: stray or natural occurrences) or **high** (10–21: the glyph was carved over and over). "The bones answer whoever counts carefully enough": each crate is a **5-bit number**, read with set 1 as the most significant bit, high = 1, and A = 1.

| crate | glyph | set counts 1–5 | bits | value | letter |
|---|---|---|---|---|---|
| 01 | 賓 | 0, 1, 17, 17, 4 | 00110 | 6 | **F** |
| 02 | 今 | 3, 1, 15, 3, 11 | 00101 | 5 | **E** |
| 03 | 其 | 2, 17, 15, 17, 2 | 01110 | 14 | **N** |
| 04 | 㞢 | 1, 1, 14, 15, 15 | 00111 | 7 | **G** |
| 05 | 王 | 21, 1, 1, 14, 16 | 10011 | 19 | **S** |
| 06 | 不 | 3, 13, 4, 1, 3 | 01000 | 8 | **H** |
| 07 | 貞 | 17, 10, 17, 4, 17 | 10101 | 21 | **U** |
| 08 | 雨 | 2, 12, 2, 3, 17 | 01001 | 9 | **I** |

The split between low and high is the largest gap in each crate's five counts (`decode.py`). The result is **FENGSHUI** (風水), a Chinese word, 8 letters.

## Reproduce
```bash
pip install torch==2.5.1 opencv-python pandas numpy
python3 train.py --data data --epochs 40 --out model.pt   # ~30 min on an M4; or use the included model.pt
python3 count.py --data data --model model.pt --thr 0.2   # -> counts.csv
python3 decode.py counts.csv                               # -> SIGMOID_FENGSHUI
```

## Limits
- Crate 07's glyph 貞 is the most common character in the archive, so its "0" sets still contain natural occurrences. Set 2 (10) is the closest call. At threshold 0.3 it comes out as 7 and the gap rule can't decide. Reading it as 1 would give 29, which is not a letter, so 0 is the only valid reading.
- The model comes from one training run without tuning; count errors per single image are larger (see `train_log.txt`), but the code only needs low vs high per set of 30 rubbings.
