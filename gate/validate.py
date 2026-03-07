from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from data.loader import load_toxicchat
from noise.injector import inject_label_noise_conditional
from gate.noise_estimator import NoiseEstimator

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_SAMPLES = 500


def run_validation(
    seeds: list[int] = DEFAULT_SEEDS,
    samples: int = DEFAULT_SAMPLES,
    baseline: str = "toxicchat",
    save: bool = False,
) -> list[dict]:

    print(f"\nNoiseCliff Gate Validation")
    print(
        f"baseline={baseline}  seeds={seeds}  samples={samples}  noise_levels={NOISE_LEVELS}\n"
    )

    print("Loading ToxicChat...")
    splits = load_toxicchat()
    texts = splits["train"]["texts"][:samples]
    labels = splits["train"]["labels"][:samples]
    print(
        f"Loaded {len(texts)} samples  "
        f"(toxic: {sum(labels)} / {len(labels)} = {sum(labels)/len(labels)*100:.1f}%)\n"
    )

    estimator = NoiseEstimator(baseline=baseline)
    rows = []

    for noise_level in NOISE_LEVELS:
        seed_estimates = []
        seed_bands = []

        for seed in seeds:
            if noise_level == 0.0:
                batch_texts = texts
                batch_labels = labels
            else:
                batch_texts, batch_labels = inject_label_noise_conditional(
                    texts, labels, noise_level, seed=seed
                )

            signals = estimator.estimate(batch_texts, batch_labels)
            seed_estimates.append(signals.estimated_noise)
            seed_bands.append(signals.noise_band)

        mean_est = float(np.mean(seed_estimates))
        std_est = float(np.std(seed_estimates))
        error = mean_est - noise_level
        abs_err = abs(error)

        band_counts = {}
        for b in seed_bands:
            band_counts[b] = band_counts.get(b, 0) + 1
        majority_band = max(band_counts, key=band_counts.get)

        expected_band = _expected_band(noise_level)
        band_correct = majority_band == expected_band

        rows.append(
            {
                "injected_noise": noise_level,
                "estimated_mean": round(mean_est, 3),
                "estimated_std": round(std_est, 3),
                "error": round(error, 3),
                "abs_error": round(abs_err, 3),
                "majority_band": majority_band,
                "expected_band": expected_band,
                "band_correct": band_correct,
                "per_seed": seed_estimates,
            }
        )

    _print_table(rows)

    if save:
        _save_results(rows)
        _save_plot(rows)

    return rows


def _expected_band(noise: float) -> str:
    if noise < 0.10:
        return "CLEAN"
    if noise < 0.18:
        return "WATCH"
    if noise < 0.25:
        return "DANGER"
    return "CRITICAL"


def _print_table(rows: list[dict]) -> None:
    width = 72

    print("─" * width)
    print(
        f"  {'Injected':>9}  {'Estimated':>14}  {'Error':>8}  {'Band':>10}  {'OK':>4}"
    )
    print(f"  {'noise':>9}  {'mean ± std':>14}  {'':>8}  {'(majority)':>10}  {'':>4}")
    print("─" * width)

    for r in rows:
        ok = "✓" if r["band_correct"] else "✗"
        error_str = f"{r['error']:+.3f}"
        est_str = f"{r['estimated_mean']:.3f} ± {r['estimated_std']:.3f}"

        if r["abs_error"] <= 0.05:
            err_display = f" {error_str}"
        elif r["error"] > 0:
            err_display = f"↑{error_str}"
        else:
            err_display = f"↓{error_str}"

        print(
            f"  {r['injected_noise']:>8.0%}  "
            f"{est_str:>14}  "
            f"{err_display:>8}  "
            f"{r['majority_band']:>10}  "
            f"{ok:>4}"
        )

    print("─" * width)

    abs_errors = [r["abs_error"] for r in rows]
    band_correct = sum(1 for r in rows if r["band_correct"])
    mae = float(np.mean(abs_errors))
    max_err = float(np.max(abs_errors))

    print(
        f"\n  MAE (mean absolute error):  {mae:.3f}  ({mae*100:.1f} percentage points)"
    )
    print(
        f"  Max absolute error:         {max_err:.3f}  ({max_err*100:.1f} percentage points)"
    )
    print(
        f"  Band accuracy:              {band_correct}/{len(rows)}  "
        f"({band_correct/len(rows)*100:.0f}%)"
    )

    cliff_row = next(r for r in rows if r["injected_noise"] == 0.20)
    prepost = [r for r in rows if r["injected_noise"] in (0.15, 0.25)]
    cliff_ok = cliff_row["majority_band"] in ("DANGER", "CRITICAL")

    print(
        f"\n  Cliff detection (20% noise): "
        f"{'✓ correctly flagged as ' + cliff_row['majority_band'] if cliff_ok else '✗ missed'}"
    )

    safe_correct = all(
        r["majority_band"] in ("CLEAN", "WATCH")
        for r in rows
        if r["injected_noise"] < 0.18
    )
    unsafe_correct = all(
        r["majority_band"] in ("DANGER", "CRITICAL")
        for r in rows
        if r["injected_noise"] >= 0.20
    )
    print(
        f"  Safe zone (<18% noise):      {'✓ all correctly cleared' if safe_correct else '✗ false positives present'}"
    )
    print(
        f"  Danger zone (≥20% noise):    {'✓ all correctly blocked' if unsafe_correct else '✗ false negatives present'}"
    )
    print()


def _save_results(rows: list[dict]) -> None:
    out_path = _REPO / "results" / "gate_validation.json"
    out_path.parent.mkdir(exist_ok=True)

    abs_errors = [r["abs_error"] for r in rows]
    band_correct = sum(1 for r in rows if r["band_correct"])

    output = {
        "summary": {
            "mae": round(float(np.mean(abs_errors)), 3),
            "max_abs_error": round(float(np.max(abs_errors)), 3),
            "band_accuracy": round(band_correct / len(rows), 3),
            "n_noise_levels": len(rows),
        },
        "rows": rows,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {out_path}")


def _save_plot(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    injected = [r["injected_noise"] for r in rows]
    estimated = [r["estimated_mean"] for r in rows]
    stds = [r["estimated_std"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NoiseCliff Gate Validation", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(
        [0, 0.4], [0, 0.4], "--", color="#aaa", linewidth=1, label="perfect calibration"
    )
    ax.errorbar(
        injected,
        estimated,
        yerr=stds,
        fmt="o",
        color="#3b82f6",
        ecolor="#93c5fd",
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label="gate estimate",
    )

    ax.axhspan(0, 0.10, alpha=0.08, color="green", label="CLEAN")
    ax.axhspan(0.10, 0.18, alpha=0.08, color="yellow", label="WATCH")
    ax.axhspan(0.18, 0.25, alpha=0.08, color="orange", label="DANGER")
    ax.axhspan(0.25, 0.45, alpha=0.08, color="red", label="CRITICAL")
    ax.axvline(0.20, color="#ef4444", linestyle=":", linewidth=1.2, label="cliff (20%)")

    ax.set_xlabel("Injected noise rate")
    ax.set_ylabel("Estimated noise rate")
    ax.set_title("Estimated vs Injected Noise")
    ax.set_xlim(-0.01, 0.43)
    ax.set_ylim(-0.01, 0.43)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = [
        (
            "#22c55e"
            if r["abs_error"] <= 0.05
            else "#f59e0b" if r["abs_error"] <= 0.10 else "#ef4444"
        )
        for r in rows
    ]
    bars = ax2.bar(
        [f"{n:.0%}" for n in injected],
        [r["abs_error"] for r in rows],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axhline(
        0.05, color="#22c55e", linestyle="--", linewidth=1, label="±5pp threshold"
    )
    ax2.axhline(
        0.10, color="#f59e0b", linestyle="--", linewidth=1, label="±10pp threshold"
    )
    ax2.axvline(3.5, color="#ef4444", linestyle=":", linewidth=1.2, label="cliff (20%)")
    ax2.set_xlabel("Injected noise rate")
    ax2.set_ylabel("Absolute error (noise rate)")
    ax2.set_title("Estimation Error by Noise Level")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    band_acc = sum(1 for r in rows if r["band_correct"]) / len(rows)
    ax2.text(
        0.97,
        0.97,
        f"Band accuracy: {band_acc:.0%}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1"),
    )

    plt.tight_layout()

    out_path = _REPO / "results" / "gate_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NoiseCliff quality gate against known noise levels"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="toxicchat",
        help="Baseline name to validate against (default: toxicchat)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Batch size to test (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to results/gate_validation.json and .png",
    )
    args = parser.parse_args()

    run_validation(
        seeds=args.seeds,
        samples=args.samples,
        baseline=args.baseline,
        save=args.save,
    )


if __name__ == "__main__":
    main()
