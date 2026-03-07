# NoiseCliff

> How much label noise can an LLM safety classifier tolerate before it becomes unreliable?  
> And can you detect it _before_ you retrain on bad data?

This project does two things:

1. **Benchmarks the exact point** where label noise breaks a toxicity classifier — not just "noise hurts" but where, how fast, and why standard metrics hide it
2. **Ships a pre-training quality gate CLI** that estimates noise rate from any new label batch before retraining, with no external ground truth required

The research was validated on SST-2 first, then applied to ToxicChat — real user prompts from a deployed LLM with real annotation pipeline noise.

---

## The core finding

There is a cliff at ~20% label noise. Below it, degradation is gradual (~0.02 PR-AUC drop per 5% interval). Above it, degradation accelerates 4.5× (~0.09 per interval). Standard metrics like F1 macro do not show this — F1 can rise while PR-AUC collapses because the model is quietly learning to predict the majority class.

In practice, it is easy to drift into this regime without measuring it.

---

## Quickstart

```bash
git clone https://github.com/yourname/noise-cliff
cd noise-cliff
pip install -r requirements.txt

# one-time setup: generate ToxicChat baseline from existing results (~5 sec)
python gate/bootstrap_toxicchat_baseline.py

# check a label batch
python gate/check.py --batch your_labels.csv --baseline toxicchat
```

CSV format: two columns required — `text` (string) and `label` (0 or 1).

---

## The quality gate

Teams retrain safety classifiers on new labeled batches weekly. If that batch is noisy, the retrain degrades production. Nobody measures this before training. The gate is a go/no-go check that runs before any training job starts.

The gate requires **no trusted external ground truth** — only the batch's own labels. It trains a lightweight TF-IDF + LogReg probe on the batch itself via 3-fold cross-validation to avoid training-set leakage and to approximate out-of-sample uncertainty signals. Noisy labels produce uncertain, boundary-hugging predictions; clean labels produce confident, well-separated ones.

```bash
python gate/check.py --batch week_47_labels.csv --baseline toxicchat
```

Output:

```
────────────────────────────────────────────────────────────
  NoiseCliff Quality Gate
  2026-03-07 12:48:26
────────────────────────────────────────────────────────────

  Batch:    week_47_labels.csv
  Baseline: toxicchat
  Samples:  2,847
  Toxic:    9.4%  (baseline toxicchat rate,  drift 2.1%)

  Estimated noise rate         24.3%
  Estimated PR-AUC if trained  0.481  (clean baseline 0.628)
  Collapse risk                DANGER
  Safe to retrain              ✗  NO

  Signal breakdown  (thresholds at toxicchat tipping point)
  ────────────────────────────────────────────────────────
  Mean entropy                 0.985   higher = more uncertain
  Mean margin |p−0.5|          0.061   lower = closer to boundary
  Near-threshold (40-60%)      0.712   higher = mass near boundary
  Toxic rate drift             0.021   higher = label dist shifted

  Recommendation
  ────────────────────────────────────────────────────────
  Near or past the cliff. Apply loss filtering.
  1. Apply loss filtering before retraining
  2. Send 847 near-boundary examples (29.7%) to adjudication
  3. Toxic rate drifted 2.1% — check label source composition
  4. Do NOT use confidence filtering
────────────────────────────────────────────────────────────
```

### Gate accuracy

Validated across 9 noise levels × 3 seeds on ToxicChat:

| Injected | Estimated (mean ± std) | Error  | Band     | Correct                 |
| -------- | ---------------------- | ------ | -------- | ----------------------- |
| 0%       | 0.000 ± 0.000          | +0.000 | CLEAN    | ✓                       |
| 5%       | 0.050 ± 0.002          | -0.000 | CLEAN    | ✓                       |
| 10%      | 0.104 ± 0.007          | +0.004 | WATCH    | ✓                       |
| 15%      | 0.154 ± 0.008          | +0.004 | WATCH    | ✓                       |
| 20%      | 0.195 ± 0.004          | -0.005 | DANGER   | ✓                       |
| 25%      | 0.242 ± 0.015          | -0.008 | DANGER   | ✗ (0.8pp from boundary) |
| 30%      | 0.311 ± 0.033          | +0.011 | CRITICAL | ✓                       |
| 35%      | 0.326 ± 0.020          | -0.024 | CRITICAL | ✓                       |
| 40%      | 0.357 ± 0.029          | -0.043 | CRITICAL | ✓                       |

**MAE: 1.1 percentage points. Max error: 4.3pp. Band accuracy: 8/9 (89%).**

Safe zone (<18% noise): all correctly cleared. Danger zone (≥20%): all correctly blocked. Cliff at 20% correctly flagged as DANGER.

The one miss is 25% landing in DANGER instead of CRITICAL — estimate (24.2%) is 0.8pp from the zone boundary. Operationally not meaningful: DANGER already says do not retrain without cleaning.

Reproduce:

```bash
python gate/validate.py --baseline toxicchat --save
```

### All gate flags

```bash
# save report to results/ as JSON
python gate/check.py --batch labels.csv --baseline toxicchat --save

# CI mode: exits with code 1 if unsafe
python gate/check.py --batch labels.csv --baseline toxicchat --ci

# structured JSON output (pipe into other tools or CI logs)
python gate/check.py --batch labels.csv --baseline toxicchat --json > report.json

# show top 10 most suspicious examples (highest adjudication priority)
python gate/check.py --batch labels.csv --baseline toxicchat --explain

# print calibration curve for a baseline
python gate/check.py --calibration-curve --baseline toxicchat
```

### Exit codes

| Code | Meaning                                                          |
| ---- | ---------------------------------------------------------------- |
| `0`  | Safe to retrain (CLEAN or WATCH)                                 |
| `1`  | Unsafe to retrain (DANGER or CRITICAL) — only raised with `--ci` |
| `2`  | Invalid input (missing columns, bad labels, file not found)      |

### Use in CI

```yaml
# .github/workflows/retrain.yml
- name: NoiseCliff quality gate
  run: |
    python gate/check.py \
      --batch data/new_labels.csv \
      --baseline toxicchat \
      --ci \
      --save
```

The job fails and blocks the retrain if the batch is in DANGER or CRITICAL. The saved JSON report goes into the workflow artifacts for review.

---

## Using a custom baseline

The gate ships calibrated for ToxicChat (7.3% toxic, English safety data). If your dataset has a different toxic rate, language, or domain, calibrate a new baseline first.

```bash
# calibrate on your own labeled CSV (~10-15 min on CPU)
python gate/calibrate.py --csv my_data.csv --name my_dataset

# then use it
python gate/check.py --batch new_labels.csv --baseline my_dataset
python gate/validate.py --baseline my_dataset
```

### What calibration does

1. Runs a noise sweep (LogReg at 0–40% injected noise) on your data to measure PR-AUC degradation
2. Measures signal curves (entropy / margin / near-threshold) at each noise level via cross-val probe
3. Finds your tipping point via piecewise linear fit
4. Saves everything to `results/<name>_baseline.json`

The gate then interpolates against your actual curves rather than ToxicChat's. This matters if your toxic rate, text distribution, or label schema differs substantially.

```bash
# recalibrate ToxicChat from scratch
python gate/calibrate.py --dataset toxicchat --name toxicchat
```

---

## Key takeaways

- **Noise cliff at ~20%**: degradation accelerates ~4.5× past this point
- **F1 macro is misleading** on imbalanced toxicity data — it can rise while PR-AUC collapses
- **Loss filtering** is the only cleaning strategy with consistent recovery; confidence filtering can destroy toxic class prediction entirely
- **Weak labels can help** when they come from a correlated source — quantity beat quality on ToxicChat because OpenAI moderation outputs carry real signal
- **DeBERTa dominates under moderate noise** but becomes highly unstable at 40% (std 0.149 vs LogReg 0.010) — unpredictable failure is operationally worse than a lower stable floor
- **Correlated noise vs random noise behave differently** — the SST-2 finding (clean beats noisy) reverses on ToxicChat

---

## Dataset

ToxicChat (`toxicchat0124`) — 10,165 real user prompts from the Vicuna LLM demo. Labeled by 4 researchers using strict majority vote. 7.33% toxic, 2.01% jailbreaking. Human-AI collaborative annotation — not all examples have full human review.

Label corrections between versions `1123` and `0124` changed 1.28% of toxicity labels and 0.34% of jailbreaking labels. That correction delta is itself a noise signal and the motivation for the gate.

License: CC-BY-NC-4.0 — non-commercial use only.

---

## Models

**Logistic Regression with TF-IDF** — fast, interpretable, stable under noise. The tipping point analysis runs on LogReg because its stability makes the cliff visible — DeBERTa's variance at high noise obscures the breakpoint.

**DeBERTa-v3-base** — the modern encoder baseline. Dominates under clean and moderate noise. Becomes unpredictably unstable at 40% noise.

Both models expose identical interfaces. Everything else — optimizer, epochs, batch size, eval split — is held constant across all experiments.

---

## Noise methodology

Label noise is injected using class-conditional flipping: toxic labels (1) and non-toxic labels (0) are each flipped independently at the specified rate. This preserves approximate class prevalence better than uniform random flipping, which would disproportionately corrupt the already-rare toxic signal.

Full implementation in `noise/injector.py`.

---

## Phase 1 results (SST-2 synthetic baseline)

Validated the experimental framework on SST-2 sentiment classification. 3 seeds per experiment.

**Degradation curves — F1 mean across 3 seeds:**

| Model      | 0% noise | 10% noise | 20% noise | 40% noise | Total drop |
| ---------- | -------- | --------- | --------- | --------- | ---------- |
| LogReg     | 0.9149   | 0.9042    | 0.8829    | 0.7172    | -0.1977    |
| DistilBERT | 0.9524   | 0.9418    | 0.9250    | 0.8340    | -0.1184    |

Note: Phase 1 used DistilBERT. Phase 2 uses DeBERTa-v3-base. SST-2 numbers kept as-is for reproducibility.

**Quantity vs quality (SST-2):**

| Scenario               | LogReg F1 | DistilBERT F1 |
| ---------------------- | --------- | ------------- |
| 50k samples, 30% noise | 0.8190    | 0.8974        |
| 20k samples, clean     | 0.8823    | 0.9364        |
| Delta                  | +0.0633   | +0.0390       |

On SST-2 with synthetic random noise, clean beats noisy. This reverses on ToxicChat.

---

## Phase 2 results (ToxicChat)

Primary metric: PR-AUC. Standard for heavily imbalanced classification and what ToxicChat's own paper reports. All experiments: 3 seeds, mean ± std. LogReg ~1 min per noise level. DeBERTa ~6 min per noise level on A100.

### Experiment 1 — Degradation curves

**LogReg PR-AUC:**

| Noise | PR-AUC | Δ from clean |
| ----- | ------ | ------------ |
| 0%    | 0.628  | —            |
| 10%   | 0.605  | -0.023       |
| 20%   | 0.532  | -0.096       |
| 40%   | 0.185  | -0.443       |

**DeBERTa-v3-base PR-AUC:**

| Noise | PR-AUC | Δ from clean |
| ----- | ------ | ------------ |
| 0%    | 0.845  | —            |
| 10%   | 0.804  | -0.041       |
| 20%   | 0.750  | -0.095       |
| 40%   | 0.243  | -0.602       |

DeBERTa starts 21.7 PR-AUC points higher and holds that advantage through 20%. At 40%, DeBERTa std = 0.149 vs LogReg std = 0.010. Unpredictable failure is operationally worse than a lower stable floor.

### Tipping point experiment

Fine-grained sweep at 5% intervals, LogReg only, 3 seeds:

| Noise | PR-AUC | Drop per step |
| ----- | ------ | ------------- |
| 5%    | 0.627  | —             |
| 10%   | 0.605  | -0.022        |
| 15%   | 0.585  | -0.020        |
| 20%   | 0.532  | -0.053        |
| 25%   | 0.470  | -0.062        |
| 30%   | 0.382  | -0.088        |
| 35%   | 0.256  | -0.126        |
| 40%   | 0.185  | -0.071        |

Breakpoint at 20% noise (std ±0.008 across seeds). Before: ~0.02 PR-AUC drop per interval. After: ~0.09. Acceleration: ~4.5×.

### Experiment 2 — Cleaning recovery

**LogReg:**

| Strategy          | 10% noise | 20% noise | 40% noise |
| ----------------- | --------- | --------- | --------- |
| Noisy baseline    | 0.605     | 0.532     | 0.185     |
| Loss filter       | 0.518     | 0.534 ✓   | 0.233 ✓   |
| Heuristic filter  | 0.604     | 0.529     | 0.175     |
| Confidence filter | 0.083 ✗✗✗ | 0.080 ✗✗✗ | 0.211     |

**DeBERTa:**

| Strategy          | 10% noise | 20% noise | 40% noise |
| ----------------- | --------- | --------- | --------- |
| Noisy baseline    | 0.804     | 0.750     | 0.243     |
| Loss filter       | 0.769     | 0.765 ✓   | 0.185     |
| Heuristic filter  | 0.799     | 0.743     | 0.181     |
| Confidence filter | 0.775     | 0.498 ✗✗✗ | 0.057 ✗✗✗ |

Loss filtering is the only strategy with consistent recovery. Confidence filtering is actively dangerous on imbalanced data. No strategy recovers from 40% noise — the damage is structural.

### Experiment 3 — Quantity vs quality

| Scenario                                        | LogReg PR-AUC | DeBERTa PR-AUC |
| ----------------------------------------------- | ------------- | -------------- |
| All data (~4,300 samples, weak labels included) | 0.628 ± 0.004 | 0.842 ± 0.010  |
| Human annotation only (~2,380 samples)          | 0.616 ± 0.011 | 0.829 ± 0.020  |
| Delta                                           | -0.011        | -0.013         |

More data wins even with weak labels — because ToxicChat's weak labels come from OpenAI moderation outputs, which are correlated with true toxicity. The type of noise matters as much as the amount.

---

## Metric trap: F1 macro can improve while the classifier gets worse

On imbalanced data, label noise causes the model to predict the majority class more aggressively. This improves majority-class F1 and pulls up macro F1 — even as PR-AUC collapses.

In the tipping point experiment: f1_macro rises from 0.606 at 5% noise to 0.656 at 30% before collapsing. PR-AUC drops from 0.627 to 0.382 over the same range. Anyone monitoring only F1 sees a stable classifier while it is actively failing.

PR-AUC measures ranking quality across all thresholds and is robust to class imbalance. It is the right primary metric for toxicity classification.

---

## Business translation

At 100,000 daily interactions with a 7.3% base toxic rate: ~7,300 toxic prompts to catch per day. Moving from clean training to 20% pipeline noise drops LogReg PR-AUC from 0.628 to 0.532 — a 15% relative decline in ranking separability. At 40% noise, PR-AUC hits 0.185 — near-random ranking.

A missed toxic prompt is a safety incident. A false block is a degraded user experience. The 20% threshold is where both risks start compounding.

---

## Project structure

```
noise-cliff/
├── requirements.txt
├── config.py
├── data/
│   └── loader.py
├── noise/
│   └── injector.py
├── models/
│   ├── logreg.py
│   ├── distilbert.py                      -- Phase 1 only
│   └── deberta.py
├── training/
│   └── trainer.py
├── evaluation/
│   └── evaluator.py
├── cleaning/
│   └── strategies.py
├── experiments/
│   ├── run_noise_sweep.py
│   ├── run_cleaning.py
│   ├── run_quantity_vs_quality.py
│   └── find_tipping_point.py
├── gate/
│   ├── check.py                           -- quality gate CLI
│   ├── noise_estimator.py                 -- signal extraction + noise estimation
│   ├── validate.py                        -- gate accuracy validation
│   ├── calibrate.py                       -- calibrate gate for a new dataset
│   └── bootstrap_toxicchat_baseline.py    -- one-time setup from existing results
├── notebooks/
│   └── plots.ipynb
└── results/
    ├── tipping_point_toxicchat.json
    ├── toxicchat_baseline.json            -- generated by bootstrap script
    ├── gate_validation.json
    └── gate_validation.png
```

---

## Running everything

```bash
pip install -r requirements.txt

# ── one-time setup ────────────────────────────────────────────────────────
python gate/bootstrap_toxicchat_baseline.py   # fast, uses existing results
# or recalibrate from scratch (~15 min):
python gate/calibrate.py --dataset toxicchat --name toxicchat

# ── research experiments ──────────────────────────────────────────────────
python experiments/find_tipping_point.py
python experiments/run_noise_sweep.py --dataset toxicchat --models logreg deberta
python experiments/run_cleaning.py --dataset toxicchat --models logreg deberta
python experiments/run_quantity_vs_quality.py --dataset toxicchat --models logreg deberta

# ── gate ──────────────────────────────────────────────────────────────────
python gate/check.py --batch your_labels.csv
python gate/check.py --batch your_labels.csv --save
python gate/check.py --batch your_labels.csv --ci
python gate/check.py --batch your_labels.csv --json > report.json
python gate/check.py --batch your_labels.csv --explain
python gate/validate.py --save

# ── custom baseline ───────────────────────────────────────────────────────
python gate/calibrate.py --csv my_data.csv --name my_dataset
python gate/check.py --batch labels.csv --baseline my_dataset
```

---

## Stack

Python 3.10, PyTorch, HuggingFace Transformers and Datasets, scikit-learn, pandas, matplotlib, sentencepiece, protobuf
