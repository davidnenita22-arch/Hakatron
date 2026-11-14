# Albania streaming anomaly detector

Submit **`detector.py`**. It exposes `Detector()` and `update(reading: dict) -> float` and needs only NumPy, which is available in the stated evaluator. The CSV files, tests, and local environment are for reproducibility and are not needed at inference.

Each bunker zone learns its own rolling median and robust deviation from past readings. A maximum of 256 accepted values per sensor is retained; unusual values do not redefine the normal baseline. The score combines individual spikes, joint deviations across sensors, and smoothed persistent shifts. Persistence requires evidence in the current reading, so alerts can stop promptly after recovery. The first 23 usable readings in each new zone initialize its baseline. Missing and non-finite sensor values are ignored. Timestamps are not used to identify events.

The detector reads no files, labels, or network resources. Every returned score uses only readings already passed to that instance. Scores are finite floats in `[0, 1]`; the alert threshold is `>= 0.5`. They are detection scores, not calibrated probabilities.

## Public-data validation

The supplied [public Google Drive folder](https://drive.google.com/drive/folders/14AMHetQATgeB-wSm0-Da10NffyxOGQVo?usp=sharing) contained `train-input.csv` and `train-output.csv`; no organizer evaluator was supplied. `validate.py` is a local replay using the stated formula:

`100 * (0.8 * F1 + 0.05 * point_recall + 0.075 * drift_recall + 0.075 * collective_recall)`

Verified on Python **3.11.17**, NumPy **2.2.4**, at threshold **0.5**:

| Measure | Result |
| --- | ---: |
| Rows | 3,600 |
| True positives / false positives / false negatives | 277 / 6 / 10 |
| F1 | 0.9719298246 |
| Point recall | 7/7 = 1.0 |
| Drift recall | 170/180 = 0.9444444444 |
| Collective recall | 100/100 = 1.0 |
| Weighted local score | **97.33771930 / 100** |
| Replay time | approximately 0.15–0.21 seconds |

Chronological checks with a fresh detector at the start of each slice also passed the stated 70-point target: first 1,800 readings **96.6126**, last 1,800 **96.8612**, last 900 **97.6000**. These are checks on the same public dataset, not an independent hidden evaluation. Thresholds were developed using the public data. Hidden-data performance and acceptance are unverified, and no platform flag has been obtained.

## Reproduce

From the `Albania` folder, using the prepared Python 3.11 environment:

```sh
.venv/bin/python validate.py --predictions predictions.csv
.venv/bin/python -m unittest -v test_detector.py
```

Four regression tests cover deterministic causal prefixes, unchanged inputs, finite bounded outputs including invalid/extreme values, independent zones, and synthetic point/drift/persistent changes. All pass. `predictions.csv` records the public replay scores and labels for inspection.
