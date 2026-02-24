# data-quality-bench

How much label noise can an LLM safety classifier tolerate before it becomes unreliable? And does cleaning your annotation pipeline actually help, or does it make things worse?

This project benchmarks data quality degradation on real LLM safety data -- using ToxicChat, a dataset of real user prompts collected from an LLM demo and labeled by human annotators for toxicity. Two models are compared: Logistic Regression as the classical baseline and DeBERTa-v3-base as the modern encoder baseline. The experimental framework was validated on SST-2 with synthetic noise first. ToxicChat is the real-world problem.

---

## Key takeaways

- **Noise cliff at ~20%**: degradation rate accelerates ~4.5× after this point -- from ~0.02 PR-AUC drop per 5% interval to ~0.09
- **F1 macro is misleading** on imbalanced toxicity data -- it can rise while the classifier is actively getting worse
- **Loss filtering helps modestly**; confidence filtering can collapse toxic class prediction entirely
- **Weak labels can help** when they come from a correlated source -- quantity beat quality on ToxicChat because OpenAI moderation outputs carry real signal
- **DeBERTa dominates under moderate noise** but becomes highly unstable at 40% (std 0.149 vs LogReg's 0.010) -- unpredictable failure is operationally worse than a lower stable floor

---

## The problem this is solving

Every company shipping an LLM product needs a toxicity filter. That filter is trained on human-labeled examples. Human labelers miss edge cases. Automated pre-filtering introduces weak labels. Label corrections happen between dataset versions. That pipeline noise goes into training and nobody measures the cost until the classifier starts failing.

This project makes that cost measurable. Not just "noise hurts performance" -- but exactly where the degradation accelerates, how different models respond, whether cleaning recovers it, and what the business impact looks like when it does not.

The framing is accurate for ToxicChat specifically: this is not annotator disagreement analysis -- labels are already aggregated via majority vote. What we are studying is real-world annotation pipeline noise: a mix of class imbalance, weak-label filtering, and residual labeling errors measured via human_annotation coverage and label corrections between dataset versions.

---

## Dataset

**ToxicChat (toxicchat0124)** [[paper](https://arxiv.org/abs/2310.17389)] -- 10,165 real user prompts from the Vicuna LLM demo. Labeled by 4 researchers using strict majority vote. 7.33% toxic, 2.01% jailbreaking. Human-AI collaborative annotation -- not all examples have full human review. Label corrections between versions 1123 and 0124 changed 1.28% of toxicity labels and 0.34% of jailbreaking labels. That correction delta is itself a noise signal.

License: CC-BY-NC-4.0 -- non-commercial use only.

---

## Models

**Logistic Regression with TF-IDF** -- the classical baseline. Fast, interpretable, and still used in production at companies that need explainability.

**DeBERTa-v3-base** [[paper](https://arxiv.org/abs/2111.09543)] -- the modern encoder baseline. Released 2021, widely used in production text classification today. Stronger than DistilBERT on nuanced and short-text classification tasks. Uses disentangled attention and enhanced mask decoder.

Both models expose identical interfaces. Everything else -- optimizer, epochs, batch size, eval split -- is held constant across every experiment.

---

## Noise methodology

Label noise is injected using class-conditional flipping: toxic labels (1) and non-toxic labels (0) are each flipped independently at the specified noise rate. This preserves approximate class prevalence better than uniform random flipping, which would disproportionately corrupt the already-rare toxic signal. At 10% noise, roughly 10% of toxic examples get flipped to non-toxic and 10% of non-toxic examples get flipped to toxic. Full implementation in `noise/injector.py`.

---

## Phase 1 results (SST-2 synthetic noise baseline)

The first phase validated the experimental framework on SST-2 sentiment classification with synthetic label noise. These results stay in the repo permanently as the controlled baseline. 3 seeds per experiment.

**Degradation curves -- F1 mean across 3 seeds:**

| Model      | 0% noise | 10% noise | 20% noise | 40% noise | Total drop |
| ---------- | -------- | --------- | --------- | --------- | ---------- |
| LogReg     | 0.9149   | 0.9042    | 0.8829    | 0.7172    | -0.1977    |
| DistilBERT | 0.9524   | 0.9418    | 0.9250    | 0.8340    | -0.1184    |

Note: Phase 1 used DistilBERT. Phase 2 uses DeBERTa-v3-base. SST-2 numbers are kept as-is for reproducibility.

**Quantity vs quality (SST-2):**

| Scenario               | LogReg F1 | DistilBERT F1 |
| ---------------------- | --------- | ------------- |
| 50k samples, 30% noise | 0.8190    | 0.8974        |
| 20k samples, clean     | 0.8823    | 0.9364        |
| Delta                  | +0.0633   | +0.0390       |

On SST-2 with synthetic random noise, 20,000 clean samples outperformed 50,000 noisy ones. This finding reverses on ToxicChat -- see Phase 2 below.

---

## Phase 2 results (ToxicChat)

Primary metric: **PR-AUC**. Standard for heavily imbalanced classification and what ToxicChat's own paper reports. F1 macro is reported alongside but is not the primary metric -- see the metric trap section below.

All experiments: 3 seeds, results reported as mean ± std. LogReg runs in ~1 min per noise level. DeBERTa runs in ~6 min per noise level on an A100.

---

### Experiment 1 -- Degradation curves

**LogReg PR-AUC:**

| Noise | PR-AUC | Δ from clean |
| ----- | ------ | ------------ |
| 0%    | 0.628  | --           |
| 10%   | 0.605  | -0.023       |
| 20%   | 0.532  | -0.096       |
| 40%   | 0.185  | -0.443       |

**DeBERTa-v3-base PR-AUC:**

| Noise | PR-AUC | Δ from clean |
| ----- | ------ | ------------ |
| 0%    | 0.845  | --           |
| 10%   | 0.804  | -0.041       |
| 20%   | 0.750  | -0.095       |
| 40%   | 0.243  | -0.602       |

DeBERTa starts 21.7 PR-AUC points higher and holds that advantage through 20% noise. Beyond ~20% noise, both models degrade rapidly. At 40% noise, both are near-collapse in PR-AUC, with DeBERTa showing high variance across seeds (std 0.149 vs LogReg's 0.010). In a production safety system, unpredictable failure is operationally worse than a lower but stable floor.

---

### Tipping point experiment

Fine-grained sweep at 5% intervals, LogReg only, 3 seeds:

| Noise | PR-AUC | Drop per step |
| ----- | ------ | ------------- |
| 5%    | 0.627  | --            |
| 10%   | 0.605  | -0.022        |
| 15%   | 0.585  | -0.020        |
| 20%   | 0.532  | -0.053        |
| 25%   | 0.470  | -0.062        |
| 30%   | 0.382  | -0.088        |
| 35%   | 0.256  | -0.126        |
| 40%   | 0.185  | -0.071        |

**Finding:** There is a consistent breakpoint around 20% noise where degradation accelerates ~4.5×. Before the breakpoint: ~0.02 PR-AUC drop per 5% interval. After: ~0.09 per interval. The piecewise linear fit locates the breakpoint at 20% (std ±0.008 across seeds).

---

### Experiment 2 -- Cleaning recovery

**LogReg:**

| Strategy          | 10% noise | 20% noise   | 40% noise   |
| ----------------- | --------- | ----------- | ----------- |
| Noisy baseline    | 0.605     | 0.532       | 0.185       |
| Loss filter       | 0.518     | **0.534 ✓** | **0.233 ✓** |
| Heuristic filter  | 0.604     | 0.529       | 0.175       |
| Confidence filter | 0.083 ✗✗✗ | 0.080 ✗✗✗   | 0.211       |

**DeBERTa:**

| Strategy          | 10% noise | 20% noise   | 40% noise |
| ----------------- | --------- | ----------- | --------- |
| Noisy baseline    | 0.804     | 0.750       | 0.243     |
| Loss filter       | 0.769     | **0.765 ✓** | 0.185     |
| Heuristic filter  | 0.799     | 0.743       | 0.181     |
| Confidence filter | 0.775     | 0.498 ✗✗✗   | 0.057 ✗✗✗ |

Loss filtering is the only strategy with consistent recovery. DeBERTa at 20% noise recovers from 0.750 to 0.765 with tighter variance.

Confidence filtering is actively dangerous on imbalanced data. At 10-20% noise it collapses LogReg to ~0.08 PR-AUC -- the model stops predicting the toxic class entirely (f1_macro std=0.0 across all seeds confirms it). The per-class guardrail restored 14-29 samples per run but could not recover classifier behavior once the signal was corrupted. DeBERTa at 40% noise hit 0.057 PR-AUC under confidence filtering -- same failure mode.

No strategy fully recovers from 40% noise. The damage at that level is structural.

---

### Experiment 3 -- Quantity vs quality

| Scenario                                        | LogReg PR-AUC | DeBERTa PR-AUC |
| ----------------------------------------------- | ------------- | -------------- |
| All data (~4,300 samples, weak labels included) | 0.628 ± 0.004 | 0.842 ± 0.010  |
| Human annotation only (~2,380 samples)          | 0.616 ± 0.011 | 0.829 ± 0.020  |
| Delta                                           | -0.011        | -0.013         |

More data wins, even with weak labels. This reverses the SST-2 Phase 1 finding where clean 20k beat noisy 50k. The key difference: ToxicChat's weak labels come from OpenAI moderation outputs, which are correlated with true toxicity -- they are noisy but not random. When weak labels carry real signal, the quantity advantage outweighs the quality disadvantage.

Human-only training also produces higher variance across seeds for both models (LogReg std 0.011 vs 0.004, DeBERTa 0.020 vs 0.010) -- smaller dataset, less stable training.

**Contrast with SST-2:** synthetic random noise makes clean data clearly better. Real correlated weak labels make more data better. The type of noise matters as much as the amount.

---

### Metric trap: F1 macro can improve while the classifier gets worse

On imbalanced data, F1 macro is a misleading primary metric for safety classification. As label noise corrupts the toxic signal, the model learns to predict the majority (non-toxic) class more aggressively. This improves majority-class F1 and pulls up macro F1 -- even as PR-AUC collapses because the model is losing its ability to rank toxic content correctly.

In the tipping point experiment, f1_macro rises from 0.606 at 5% noise to 0.656 at 30% noise before collapsing. PR-AUC drops from 0.627 to 0.382 over the same range. Anyone monitoring only F1 would see a stable or improving classifier while it is actively failing at its core job.

PR-AUC measures ranking quality across all thresholds and is robust to class imbalance. It is the right primary metric for toxicity classification.

---

## Business translation

At 100,000 user interactions per day with a 7.3% base toxic rate, there are approximately 7,300 toxic prompts to catch. Moving from a clean training set to one with 20% pipeline noise drops LogReg PR-AUC from 0.628 to 0.532 -- a 15% relative decline in ranking separability, which translates to worse precision/recall tradeoffs at any fixed deployment threshold. At 40% noise, PR-AUC hits 0.185, indicating near-random ranking of toxic vs non-toxic content.

A missed toxic prompt is a safety incident. A false block is a degraded user experience. The 20% noise threshold is where both risks start compounding -- and where most real annotation pipelines operate without realizing it.

---

## Project structure

```
data-quality-bench/
├── config.py                              -- all hyperparameters and paths
├── data/
│   └── loader.py                          -- SST-2 and ToxicChat loaders
├── noise/
│   └── injector.py                        -- class-conditional label noise
├── models/
│   ├── logreg.py                          -- TF-IDF + logistic regression
│   ├── distilbert.py                      -- Phase 1 only, kept for reference
│   └── deberta.py                         -- DeBERTa-v3-base fine-tuning wrapper
├── training/
│   └── trainer.py                         -- seed control, model init
├── evaluation/
│   └── evaluator.py                       -- PR-AUC, macro F1, seed aggregation
├── cleaning/
│   └── strategies.py                      -- loss, confidence, heuristic filters
├── experiments/
│   ├── run_noise_sweep.py                 -- experiment 1: degradation curves
│   ├── run_cleaning.py                    -- experiment 2: recovery study
│   ├── run_quantity_vs_quality.py         -- experiment 3: quantity vs quality
│   └── find_tipping_point.py             -- exact noise threshold with std
├── scripts/
│   └── save_models.py                     -- pretrain and save for Streamlit
├── streamlit_app/
│   ├── app.py                             -- interactive demo
│   └── inference.py                       -- load saved model, predict
├── notebooks/
│   └── plots.ipynb                        -- all visualizations
└── results/                               -- JSON results and PNG plots
```

---

## Running the experiments

```bash
# install dependencies
pip install sentencepiece protobuf

# tipping point (logreg only, ~1 min)
python experiments/find_tipping_point.py

# degradation curves
python experiments/run_noise_sweep.py --dataset toxicchat --models logreg deberta

# cleaning recovery
python experiments/run_cleaning.py --dataset toxicchat --models logreg deberta

# quantity vs quality
python experiments/run_quantity_vs_quality.py --dataset toxicchat --models logreg deberta
```

All results save to `results/` as JSON. Run `notebooks/plots.ipynb` to generate all visualizations.

---

## Stack

Python 3.10, PyTorch, HuggingFace Transformers and Datasets, scikit-learn, pandas, matplotlib, seaborn, Streamlit, sentencepiece, protobuf
