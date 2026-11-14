# Shaken, Still Standing — Romania

**Flag: `SIGMOID_CONVOLUTION`**

The supplied evaluator returned this flag for `predictions.csv`, which contains all 5,000 sealed records with exactly the columns `record_id,vrancea`. Its original output is preserved in `outputs/evaluator_model.txt`.

## Model results

All five-fold comparisons use `StratifiedKFold(5, shuffle=True, random_state=0)` on the 7,500 labelled records.

| Method | Five-fold accuracy |
|---|---:|
| Scaled logistic regression, C=1 | 60.08% |
| Scaled RBF SVM, C=3 | 67.55% |
| Initial two-regime mixture | 92.91% |
| **Final conditional specialists** | **94.93%** |

Final fold accuracies were **94.80%, 95.73%, 94.67%, 94.80%, 94.67%**. Out-of-fold ROC AUC was **0.97423**, log loss **0.19248**, with 380 errors among 7,500 records. `outputs/validation.json` contains baseline scores, final fold metrics, learned channels, and input hashes.

Exploratory models used a stratified 75%/25% split with seed 42 (5,625 training and 1,875 validation records):

| Model | Holdout accuracy |
|---|---:|
| Logistic regression | 59.73% |
| Linear discriminant analysis | 59.63% |
| Quadratic discriminant analysis | 64.96% |
| RBF SVM, C=3 | 68.21% |
| RBF SVM, C=30 | 67.41% |
| ExtraTrees, 500 trees | 69.81% |
| Histogram gradient boosting | 86.77% |
| CatBoost, depth 6 | 88.80% |
| k-nearest neighbours | 52.43% |

Restricting a standalone model to the 12 approximately Gaussian channels gave only about 55%–60%. A first mixture combined a variance gate, narrow-regime quadratic/SVM models, and broad-regime CatBoost models. It achieved 93.71% on the exploratory holdout and 92.91% in five-fold validation, but its sealed submission did not unlock the flag. The final solution replaces the broad-regime classifier with conditional specialists.

Exploratory scripts, temporary logs, datasets, local environments, archives, and post-unlock audit helpers are intentionally excluded from this minimal submission. The final method and the two five-fold baselines are reproducible from the included code.

## How the models are combined

`record_id` is used only to align output rows. The model uses the 48 channel amplitudes.

1. Standardize the channels using training statistics. Select channels whose absolute excess kurtosis is below 0.3. A two-component diagonal Gaussian mixture on the remaining channels identifies narrow and broad variance regimes.
2. In the narrow regime, average probabilities from quadratic-feature logistic regression (C=0.1) and an RBF SVM (C=3, gamma=0.03). Every outer fold independently selects `c03,c09,c12,c14,c15,c19,c27,c28,c30,c36,c39,c40` for these models.
3. In the broad regime, rank channels with CatBoost and test each of the top four as a switch using an internal 80%/20% split with seed 73. Every fold selects `c13` as the switch and `c35,c38,c43` as the interaction channels. A decision stump learns the threshold from agreement with a preliminary three-channel model; the final raw threshold is approximately `c13 = -0.5885`.
4. Below the switch, a depth-4 CatBoost model captures a parity/XOR-like interaction among the three channels. Above it, logistic regression (C=1) uses all 48 standardized channels.
5. Combine the narrow and broad probabilities using the Gaussian mixture's posterior regime probabilities. Threshold the result at 0.5 for the final verdict.

Scaling, channel screening, mixture fitting, branch selection, and model fitting are repeated on each outer fold's training rows. No channel indices or switch threshold are hardcoded into selection. No vault information is used during model training or validation. Because architecture development used the labelled dataset, CV is a development estimate; the successful sealed evaluation supplies the independent challenge outcome.

## Reproduce

Python **3.12.15** was used. Obtain the original `dataset/train.csv`, `dataset/sealed.csv`, `dataset/vault.npz`, and `evaluation_script.py` from the [challenge source folder](https://drive.google.com/drive/folders/1dCTKtIKpRqyJs57fN6nqIR2RuYuwt-Hd), and place them under `Romania/` in their original layout.

From `Romania/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python solve.py
```

This reruns the final five-fold comparison and the logistic/SVM baselines, fits all training rows, writes `predictions.csv`, and invokes the original evaluator. To fit and predict without rerunning cross-validation:

```bash
.venv/bin/python solve.py --skip-cv
```

To evaluate the included predictions directly:

```bash
.venv/bin/python evaluation_script.py predictions.csv
```

Expected output:

```text
SIGMOID_CONVOLUTION
```

`solve.py` is the entry point, and `structural_model.py` implements the final combination. Additional per-record outputs generated during reproduction are not required for submission.
