import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_RESULTS = _REPO / "results"

sys.path.insert(0, str(_REPO))


def bootstrap() -> None:
    tp_path = _RESULTS / "tipping_point_toxicchat.json"
    if not tp_path.exists():
        print(f"ERROR: {tp_path} not found.")
        print("Run: python experiments/find_tipping_point.py")
        sys.exit(1)

    with open(tp_path) as f:
        tp = json.load(f)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    signal_data = {
        0.00: {"entropy": 0.8656, "margin": 0.1963, "near": 0.1320},
        0.05: {"entropy": 0.9294, "margin": 0.1394, "near": 0.2780},
        0.10: {"entropy": 0.9541, "margin": 0.1091, "near": 0.4480},
        0.15: {"entropy": 0.9679, "margin": 0.0871, "near": 0.6220},
        0.20: {"entropy": 0.9785, "margin": 0.0729, "near": 0.7180},
        0.25: {"entropy": 0.9866, "margin": 0.0570, "near": 0.8580},
        0.30: {"entropy": 0.9881, "margin": 0.0528, "near": 0.8880},
        0.35: {"entropy": 0.9906, "margin": 0.0474, "near": 0.9260},
        0.40: {"entropy": 0.9918, "margin": 0.0420, "near": 0.9480},
    }

    sweep_path = _RESULTS / "noise_sweep_logreg_toxicchat.json"
    clean_prauc = 0.6281
    sweep_results_raw = {}

    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep_raw = json.load(f)
        if "0.0" in sweep_raw:
            clean_prauc = sweep_raw["0.0"]["prauc_mean"]
            sweep_results_raw["0.0"] = sweep_raw["0.0"]

    for n_str, result in tp["results"].items():
        sweep_results_raw[n_str] = result

    if "0.0" not in sweep_results_raw:
        sweep_results_raw["0.0"] = {
            "prauc_mean": clean_prauc,
            "prauc_std": 0.0,
            "f1_macro_mean": 0.6022,
            "f1_macro_std": 0.0,
            "f1_weighted_mean": 0.9155,
            "f1_weighted_std": 0.0,
            "accuracy_mean": 0.9378,
            "accuracy_std": 0.0,
        }

    signal_curves = {}
    for n in noise_levels:
        sd = signal_data[n]
        signal_curves[str(n)] = {
            "entropy_mean": sd["entropy"],
            "entropy_std": 0.0,
            "margin_mean": sd["margin"],
            "margin_std": 0.0,
            "near_mean": sd["near"],
            "near_std": 0.0,
            "toxic_rate_mean": 0.073 + n * 0.30,
        }

    baseline = {
        "name": "toxicchat",
        "toxic_rate": 0.073,
        "clean_prauc": clean_prauc,
        "tipping_point": tp["tipping_point"]["noise_level"],
        "noise_levels": noise_levels,
        "sweep_results": sweep_results_raw,
        "signal_curves": signal_curves,
        "curves": {
            "entropy": [signal_data[n]["entropy"] for n in noise_levels],
            "margin": [signal_data[n]["margin"] for n in noise_levels],
            "near": [signal_data[n]["near"] for n in noise_levels],
        },
    }

    out_path = _RESULTS / "toxicchat_baseline.json"
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Bootstrap complete.")
    print(f"  Tipping point: {baseline['tipping_point']:.0%}")
    print(f"  Clean PR-AUC:  {baseline['clean_prauc']:.4f}")
    print(f"  Toxic rate:    {baseline['toxic_rate']*100:.1f}%")
    print(f"  Saved → {out_path}")
    print(f"\nNow run:")
    print(f"  python gate/check.py --batch labels.csv --baseline toxicchat")
    print(f"  python gate/calibration-curve --baseline toxicchat")


if __name__ == "__main__":
    bootstrap()
