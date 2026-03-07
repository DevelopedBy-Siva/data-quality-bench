from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from gate.noise_estimator import (
    NoiseEstimator,
    BatchSignals,
    print_calibration_curve,
    list_baselines,
    _batch_cross_val_proba,
)

SAFE_BANDS = {"CLEAN", "WATCH"}
UNSAFE_BANDS = {"DANGER", "CRITICAL"}

BAND_COLORS = {
    "CLEAN": "\033[92m",
    "WATCH": "\033[93m",
    "DANGER": "\033[91m",
    "CRITICAL": "\033[91m",
}
RESET = "\033[0m"
BOLD = "\033[1m"

EXIT_SAFE = 0
EXIT_UNSAFE = 1
EXIT_INVALID = 2


def _color(text: str, band: str) -> str:
    return f"{BAND_COLORS.get(band, '')}{text}{RESET}"


def _bar(value: float, width: int = 24, threshold: float | None = None) -> str:
    filled = int(min(value, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if threshold is not None:
        pos = int(threshold * width)
        lst = list(bar)
        if 0 <= pos < width:
            lst[pos] = "│"
        bar = "".join(lst)
    return f"[{bar}]"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _get_suspicious_examples(
    texts: list[str],
    labels: list[int],
    p_toxic: np.ndarray,
    n: int = 10,
) -> list[dict]:
    """
    Returns the n examples closest to the decision boundary (|p - 0.5| smallest).
    These are the most ambiguous predictions — highest adjudication priority.
    """
    margins = np.abs(p_toxic - 0.5)
    top_idx = np.argsort(margins)[:n]
    examples = []
    for idx in top_idx:
        examples.append(
            {
                "index": int(idx),
                "text": texts[idx][:120] + ("…" if len(texts[idx]) > 120 else ""),
                "label": int(labels[idx]),
                "p_toxic": round(float(p_toxic[idx]), 4),
                "margin": round(float(margins[idx]), 4),
            }
        )
    return examples


def print_explain(examples: list[dict]) -> None:
    width = 60
    print(f"\n  {'─' * (width - 4)}")
    print(f"  Top {len(examples)} most suspicious examples (closest to boundary)")
    print(f"  {'─' * (width - 4)}")
    print(f"  {'#':<4}  {'p_toxic':>7}  {'label':>5}  {'margin':>7}  text")
    print(f"  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*30}")
    for ex in examples:
        print(
            f"  {ex['index']:<4}  "
            f"{ex['p_toxic']:>7.4f}  "
            f"{ex['label']:>5}  "
            f"{ex['margin']:>7.4f}  "
            f"{ex['text']}"
        )
    print()


def _build_report(
    batch_path: str,
    baseline: str,
    signals: BatchSignals,
    safe: bool,
    extra_stats: dict,
    actions: list[str],
    timestamp: str,
) -> dict:
    """Builds the full structured report dict (used for --json and --save)."""
    return {
        "timestamp": timestamp,
        "batch": batch_path,
        "baseline": baseline,
        "safe_to_retrain": safe,
        "signals": {
            "n_samples": signals.n_samples,
            "toxic_rate": signals.toxic_rate,
            "toxic_rate_drift": signals.toxic_rate_drift,
            "mean_entropy": signals.mean_entropy,
            "mean_margin": signals.mean_margin,
            "near_threshold": signals.near_threshold,
            "estimated_prauc": signals.estimated_prauc,
            "estimated_noise": signals.estimated_noise,
            "noise_band": signals.noise_band,
        },
        "batch_stats": extra_stats,
        "recommendation": signals.recommended_action,
        "actions": actions,
    }


def print_report(
    batch_path: str,
    signals: BatchSignals,
    safe: bool,
    extra_stats: dict,
    actions: list[str],
    baseline: str,
    timestamp: str,
) -> None:
    band = signals.noise_band
    c = lambda t: _color(t, band)
    width = 60

    print()
    print("─" * width)
    print(f"  {BOLD}NoiseCliff Quality Gate{RESET}")
    print(f"  {timestamp}")
    print("─" * width)

    print(f"\n  Batch:    {batch_path}")
    print(f"  Baseline: {signals.baseline_name}")
    print(f"  Samples:  {signals.n_samples:,}")
    print(
        f"  Toxic:    {_pct(signals.toxic_rate)}  "
        f"(baseline {signals.baseline_name} rate,  "
        f"drift {_pct(signals.toxic_rate_drift)})"
    )

    print(
        f"\n  {'Estimated noise rate':<28} "
        f"{c(BOLD + _pct(signals.estimated_noise) + RESET)}"
    )
    print(
        f"  {'Estimated PR-AUC if trained':<28} "
        f"{c(BOLD + str(signals.estimated_prauc) + RESET)}  "
        f"(clean baseline {signals.estimated_prauc:.4f})"
    )
    print(f"  {'Collapse risk':<28} " f"{c(BOLD + band + RESET)}")
    print(f"\n  {'Safe to retrain':<28} " f"{'✓  YES' if safe else c('✗  NO')}")

    print(f"\n  {'─' * (width - 4)}")
    print(f"  Signal breakdown  (thresholds at {signals.baseline_name} tipping point)")
    print(f"  {'─' * (width - 4)}")

    sig_rows = [
        ("Mean entropy", signals.mean_entropy, None, "higher = more uncertain"),
        (
            "Mean margin |p−0.5|",
            signals.mean_margin,
            None,
            "lower = closer to boundary",
        ),
        (
            "Near-threshold (40-60%)",
            signals.near_threshold,
            None,
            "higher = mass near boundary",
        ),
        (
            "Toxic rate drift",
            signals.toxic_rate_drift,
            None,
            "higher = label dist shifted",
        ),
    ]
    for label, value, _, note in sig_rows:
        bar = _bar(min(value, 1.0), width=24)
        print(f"  {label:<28} {value:.4f}  {bar}  {note}")

    if extra_stats:
        print(f"\n  {'─' * (width - 4)}")
        print(f"  Batch statistics")
        print(f"  {'─' * (width - 4)}")
        for k, v in extra_stats.items():
            print(f"  {k:<28} {v}")

    print(f"\n  {'─' * (width - 4)}")
    print(f"  Recommendation")
    print(f"  {'─' * (width - 4)}")
    print(f"  {signals.recommended_action}")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action}")

    print()
    print("─" * width)
    print()


def _get_actions(signals: BatchSignals) -> list[str]:
    """Derive specific recommended actions from signal values."""
    actions = []
    near_count = int(signals.near_threshold * signals.n_samples)

    if signals.noise_band in ("DANGER", "CRITICAL"):
        actions.append(
            "Apply loss filtering before retraining "
            "(recovers ~0.015 PR-AUC at this noise level — see cleaning study)"
        )

    if signals.near_threshold >= 0.25:
        actions.append(
            f"Send {near_count:,} near-boundary examples "
            f"({_pct(signals.near_threshold)} of batch) to adjudication"
        )

    if signals.toxic_rate_drift >= 0.05:
        actions.append(
            f"Toxic rate drifted {_pct(signals.toxic_rate_drift)} from baseline — "
            f"check if label source composition changed"
        )

    if signals.noise_band in ("DANGER", "CRITICAL"):
        actions.append(
            "Do NOT use confidence filtering — "
            "collapses toxic class prediction on imbalanced data (see cleaning study)"
        )

    if signals.noise_band == "CRITICAL":
        actions.append(
            "Audit individual label sources before proceeding. "
            "At this noise level, no cleaning strategy fully recovers PR-AUC."
        )

    if not actions:
        actions.append("No corrective actions required.")

    return actions


def load_batch(csv_path: str) -> tuple[list[str], list[int], dict]:
    """
    Loads batch CSV. Exits with code 2 on invalid input.
    Returns (texts, labels, extra_stats).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n  Error reading CSV: {e}\n")
        sys.exit(EXIT_INVALID)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"\n  Error: CSV missing required columns: {missing}\n"
            f"  Expected: text, label\n"
            f"  Found:    {list(df.columns)}\n"
        )
        sys.exit(EXIT_INVALID)

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    if before != len(df):
        print(f"  Warning: dropped {before - len(df)} rows with null values")

    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        print(
            f"\n  Error: labels must be 0 or 1. "
            f"Found {len(invalid)} invalid values: {df['label'].unique().tolist()}\n"
        )
        sys.exit(EXIT_INVALID)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    extra_stats = {
        "Total rows": f"{len(df):,}",
        "Toxic (label=1)": f"{labels.count(1):,}  ({labels.count(1)/len(labels)*100:.1f}%)",
        "Non-toxic (label=0)": f"{labels.count(0):,}  ({labels.count(0)/len(labels)*100:.1f}%)",
        "Avg text length": f"{df['text'].str.len().mean():.0f} chars",
    }

    if "labeler_id" in df.columns:
        extra_stats["Labelers"] = str(df["labeler_id"].nunique())
        majority = (
            df.groupby("text")["label"].agg(lambda x: x.mode()[0]).to_dict()
            if "text" in df.columns
            else {}
        )
        if majority:
            df["majority_label"] = df["text"].map(majority)
            per_labeler = df.groupby("labeler_id").apply(
                lambda g: (g["label"] != g["majority_label"]).mean()
            )
            worst = per_labeler.idxmax()
            extra_stats["Labeler disagreement (max)"] = (
                f"labeler {worst}: {per_labeler[worst]*100:.1f}% flip rate"
            )

    return texts, labels, extra_stats


def run_gate(
    batch_path: str,
    baseline: str = "toxicchat",
    save: bool = False,
    ci: bool = False,
    as_json: bool = False,
    explain: bool = False,
) -> BatchSignals:

    print(f"\n  Loading batch: {batch_path}")
    batch_texts, batch_labels, extra_stats = load_batch(batch_path)

    try:
        estimator = NoiseEstimator(baseline=baseline)
    except FileNotFoundError as e:
        print(f"\n  Error: {e}\n")
        sys.exit(EXIT_INVALID)

    signals = estimator.estimate(batch_texts, batch_labels)
    safe = signals.noise_band in SAFE_BANDS
    actions = _get_actions(signals)
    timestamp = datetime.now().isoformat()

    suspicious = []
    if explain:
        print(f"  Running probe for --explain...")
        p_toxic = _batch_cross_val_proba(batch_texts, batch_labels, cv=3)
        suspicious = _get_suspicious_examples(batch_texts, batch_labels, p_toxic)

    report = _build_report(
        batch_path, baseline, signals, safe, extra_stats, actions, timestamp
    )
    if suspicious:
        report["suspicious_examples"] = suspicious

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print_report(
            batch_path,
            signals,
            safe,
            extra_stats,
            actions,
            baseline,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if explain and suspicious:
            print_explain(suspicious)

    if save:
        stem = Path(batch_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"gate_{stem}_{ts}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        if not as_json:
            print(f"  Report saved → {out_path}\n")

    if ci and not safe:
        if not as_json:
            print(f"  CI: exiting with code {EXIT_UNSAFE} (unsafe to retrain)\n")
        sys.exit(EXIT_UNSAFE)

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python gate/check.py",
        description="NoiseCliff pre-training quality gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  safe to retrain
  1  unsafe to retrain  (only raised with --ci)
  2  invalid input

Examples:
  python gate/check.py --batch new_labels.csv
  python gate/check.py --batch new_labels.csv --baseline my_dataset
  python gate/check.py --batch new_labels.csv --ci --save
  python gate/check.py --batch new_labels.csv --json > report.json
  python gate/check.py --batch new_labels.csv --explain
  python gate/check.py --calibration-curve --baseline toxicchat
        """,
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to label batch CSV (columns: text, label)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="toxicchat",
        help="Baseline name (default: toxicchat). Run gate/calibrate.py to add new ones.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save full report to results/ as JSON",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with code 1 if unsafe to retrain (use in CI pipelines)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print structured JSON report to stdout instead of human-readable output",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show top 10 most suspicious examples (closest to decision boundary)",
    )
    parser.add_argument(
        "--calibration-curve",
        action="store_true",
        help="Print calibration curve for the selected baseline and exit",
    )

    args = parser.parse_args()

    if args.calibration_curve:
        try:
            print_calibration_curve(args.baseline)
        except FileNotFoundError as e:
            print(f"\n  Error: {e}\n")
            sys.exit(EXIT_INVALID)
        sys.exit(EXIT_SAFE)

    if not args.batch:
        parser.error("--batch is required (unless --calibration-curve is set)")

    if not Path(args.batch).exists():
        print(f"\n  Error: file not found: {args.batch}\n")
        sys.exit(EXIT_INVALID)

    run_gate(
        batch_path=args.batch,
        baseline=args.baseline,
        save=args.save,
        ci=args.ci,
        as_json=args.as_json,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
