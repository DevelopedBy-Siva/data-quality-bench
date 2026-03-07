from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from noise.injector import inject_label_noise_conditional
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds
from gate.noise_estimator import _batch_cross_val_proba, _binary_entropy

DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_SAMPLES = 500


def _load_builtin(name: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load a built-in dataset. Returns (train_texts, train_labels, test_texts, test_labels)."""
    if name == "toxicchat":
        from data.loader import load_toxicchat

        splits = load_toxicchat()
        return (
            splits["train"]["texts"],
            splits["train"]["labels"],
            splits["test"]["texts"],
            splits["test"]["labels"],
        )
    raise ValueError(
        f"Unknown built-in dataset: '{name}'. "
        f"Supported: toxicchat. For custom data use --csv."
    )


def _load_csv(csv_path: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load a CSV with text,label columns. Splits 80/20 train/test."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        raise ValueError(
            f"Labels must be 0 or 1. Found: {df['label'].unique().tolist()}"
        )

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tr_t, te_t, tr_l, te_l = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return tr_t, tr_l, te_t, te_l


def _run_noise_sweep(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    noise_levels: list[float],
    seeds: list[int],
    samples: int,
) -> dict:
    """
    Runs LogReg at each noise level across seeds.
    Returns dict: noise_level → {prauc_mean, prauc_std, ...}
    """
    if samples and samples < len(train_texts):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_texts), samples, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    results = {}

    for noise in noise_levels:
        seed_results = []
        for seed in seeds:
            if noise == 0.0:
                t, l = train_texts, train_labels
            else:
                t, l = inject_label_noise_conditional(
                    train_texts, train_labels, noise, seed=seed
                )

            model = train("logreg", t, l, seed=seed)
            metrics = evaluate(model, test_texts, test_labels)
            seed_results.append(metrics)

        agg = aggregate_across_seeds(seed_results)
        results[str(noise)] = agg
        print(
            f"  noise={noise:.0%}  prauc={agg['prauc_mean']:.4f} ± {agg['prauc_std']:.4f}"
        )

    return results


def _measure_signal_curves(
    train_texts: list[str],
    train_labels: list[int],
    noise_levels: list[float],
    seeds: list[int],
    samples: int,
) -> dict:
    """
    Measures entropy/margin/near-threshold at each noise level.
    These become the interpolation curves in the estimator.
    Returns dict: noise_level → {entropy, margin, near, toxic_rate}
    """
    if samples and samples < len(train_texts):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_texts), samples, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    signal_curves = {}

    for noise in noise_levels:
        seed_entropy = []
        seed_margin = []
        seed_near = []
        seed_toxic = []

        for seed in seeds:
            if noise == 0.0:
                t, l = train_texts, train_labels
            else:
                t, l = inject_label_noise_conditional(
                    train_texts, train_labels, noise, seed=seed
                )

            p = _batch_cross_val_proba(t, l, cv=3)
            ents = [_binary_entropy(float(x)) for x in p]
            margins = np.abs(p - 0.5)
            near = np.mean((p >= 0.4) & (p <= 0.6))

            seed_entropy.append(float(np.mean(ents)))
            seed_margin.append(float(np.mean(margins)))
            seed_near.append(float(near))
            seed_toxic.append(float(np.mean(l)))

        signal_curves[str(noise)] = {
            "entropy_mean": round(float(np.mean(seed_entropy)), 4),
            "entropy_std": round(float(np.std(seed_entropy)), 4),
            "margin_mean": round(float(np.mean(seed_margin)), 4),
            "margin_std": round(float(np.std(seed_margin)), 4),
            "near_mean": round(float(np.mean(seed_near)), 4),
            "near_std": round(float(np.std(seed_near)), 4),
            "toxic_rate_mean": round(float(np.mean(seed_toxic)), 4),
        }

        print(
            f"  noise={noise:.0%}  "
            f"entropy={signal_curves[str(noise)]['entropy_mean']:.4f}  "
            f"margin={signal_curves[str(noise)]['margin_mean']:.4f}  "
            f"near={signal_curves[str(noise)]['near_mean']:.4f}"
        )

    return signal_curves


def _find_tipping_point(noise_levels: list[float], prauc_means: list[float]) -> float:
    best_bp = noise_levels[1]
    best_error = float("inf")

    for i in range(1, len(noise_levels) - 1):
        left_x = np.array(noise_levels[: i + 1])
        left_y = np.array(prauc_means[: i + 1])
        right_x = np.array(noise_levels[i:])
        right_y = np.array(prauc_means[i:])

        lf = np.polyfit(left_x, left_y, 1)
        rf = np.polyfit(right_x, right_y, 1)

        error = np.sum((left_y - np.polyval(lf, left_x)) ** 2) + np.sum(
            (right_y - np.polyval(rf, right_x)) ** 2
        )
        if error < best_error:
            best_error = error
            best_bp = noise_levels[i]

    return best_bp


def run_calibration(
    name: str,
    dataset: str | None = None,
    csv_path: str | None = None,
    noise_levels: list[float] = DEFAULT_NOISE_LEVELS,
    seeds: list[int] = DEFAULT_SEEDS,
    samples: int = DEFAULT_SAMPLES,
) -> Path:
    """
    Full calibration run. Returns path to saved baseline JSON.
    """
    print(f"\nNoiseCliff Calibration — {name}")
    print(f"seeds={seeds}  samples={samples}  noise_levels={noise_levels}\n")

    if csv_path:
        print(f"Loading CSV: {csv_path}")
        tr_t, tr_l, te_t, te_l = _load_csv(csv_path)
    elif dataset:
        print(f"Loading built-in dataset: {dataset}")
        tr_t, tr_l, te_t, te_l = _load_builtin(dataset)
    else:
        raise ValueError("Must provide --csv or --dataset")

    toxic_rate = sum(tr_l) / len(tr_l)
    print(f"Train: {len(tr_t):,} samples  toxic={toxic_rate*100:.1f}%")
    print(f"Test:  {len(te_t):,} samples\n")

    print("Step 1/2: Running noise sweep (LogReg)...")
    sweep_results = _run_noise_sweep(
        tr_t,
        tr_l,
        te_t,
        te_l,
        noise_levels,
        seeds,
        samples,
    )

    prauc_means = [sweep_results[str(n)]["prauc_mean"] for n in noise_levels]
    tipping_pt = _find_tipping_point(noise_levels, prauc_means)
    clean_prauc = prauc_means[0]

    print(f"\n  Tipping point: {tipping_pt:.0%}")
    print(f"  Clean PR-AUC:  {clean_prauc:.4f}\n")

    print("Step 2/2: Measuring signal curves...")
    signal_curves = _measure_signal_curves(
        tr_t,
        tr_l,
        noise_levels,
        seeds,
        samples,
    )

    baseline = {
        "name": name,
        "toxic_rate": round(toxic_rate, 4),
        "clean_prauc": clean_prauc,
        "tipping_point": tipping_pt,
        "noise_levels": noise_levels,
        "sweep_results": sweep_results,
        "signal_curves": signal_curves,
        "curves": {
            "entropy": [signal_curves[str(n)]["entropy_mean"] for n in noise_levels],
            "margin": [signal_curves[str(n)]["margin_mean"] for n in noise_levels],
            "near": [signal_curves[str(n)]["near_mean"] for n in noise_levels],
        },
    }

    out_path = _REPO / "results" / f"{name}_baseline.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nCalibration complete.")
    print(f"  Tipping point: {tipping_pt:.0%}")
    print(f"  Clean PR-AUC:  {clean_prauc:.4f}")
    print(f"  Toxic rate:    {toxic_rate*100:.1f}%")
    print(f"  Saved → {out_path}")
    print(f"\nNow use this baseline:")
    print(f"  python gate/check.py --batch labels.csv --baseline {name}")
    print(f"  python gate/validate.py --baseline {name}\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate NoiseCliff gate for a new baseline dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # calibrate on ToxicChat (re-run to refresh)
  python gate/calibrate.py --dataset toxicchat --name toxicchat

  # calibrate on your own labeled CSV
  python gate/calibrate.py --csv my_data.csv --name my_dataset

  # then use it
  python gate/check.py --batch new_labels.csv --baseline my_dataset
        """,
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for this baseline (used in output filename)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Built-in dataset: toxicchat"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to CSV with text,label columns"
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help="Noise levels to sweep (default: 0 0.05 0.1 ... 0.4)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 43 44)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Max training samples per run (default: 500)",
    )

    args = parser.parse_args()

    if not args.dataset and not args.csv:
        parser.error("Must provide --dataset or --csv")

    run_calibration(
        name=args.name,
        dataset=args.dataset,
        csv_path=args.csv,
        noise_levels=args.noise_levels,
        seeds=args.seeds,
        samples=args.samples,
    )


if __name__ == "__main__":
    main()
