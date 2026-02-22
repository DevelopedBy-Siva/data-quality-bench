import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SEEDS, QUANTITY_VS_QUALITY
from data.loader import load_sst2_subset
from noise.injector import inject_label_noise
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results


def run_quantity_vs_quality(model_name):
    print(f"\nrunning quantity vs quality — {model_name}")
    print(
        f"  scenario A: {QUANTITY_VS_QUALITY['noisy_size']} samples @ {QUANTITY_VS_QUALITY['noisy_noise_level']:.0%} label noise"
    )
    print(f"  scenario B: {QUANTITY_VS_QUALITY['clean_size']} samples @ 0% label noise")
    print(f"  seeds: {SEEDS}\n")

    noisy_results = []
    clean_results = []

    for seed in SEEDS:
        noisy_splits = load_sst2_subset(QUANTITY_VS_QUALITY["noisy_size"], seed=seed)
        test_texts = noisy_splits["test"]["texts"]
        test_labels = noisy_splits["test"]["labels"]

        noisy_texts, noisy_labels = inject_label_noise(
            noisy_splits["train"]["texts"],
            noisy_splits["train"]["labels"],
            QUANTITY_VS_QUALITY["noisy_noise_level"],
            seed=seed,
        )

        noisy_model = train(model_name, noisy_texts, noisy_labels, seed=seed)
        noisy_metrics = evaluate(noisy_model, test_texts, test_labels)
        noisy_results.append(noisy_metrics)

        clean_splits = load_sst2_subset(QUANTITY_VS_QUALITY["clean_size"], seed=seed)

        clean_model = train(
            model_name,
            clean_splits["train"]["texts"],
            clean_splits["train"]["labels"],
            seed=seed,
        )
        clean_metrics = evaluate(clean_model, test_texts, test_labels)
        clean_results.append(clean_metrics)

        print(
            f"  seed={seed} | "
            f"noisy(50k,30%) acc={noisy_metrics['accuracy']:.4f} | "
            f"clean(20k) acc={clean_metrics['accuracy']:.4f}"
        )

    noisy_aggregated = aggregate_across_seeds(noisy_results)
    clean_aggregated = aggregate_across_seeds(clean_results)

    f1_delta = round(clean_aggregated["f1_mean"] - noisy_aggregated["f1_mean"], 4)
    acc_delta = round(
        clean_aggregated["accuracy_mean"] - noisy_aggregated["accuracy_mean"], 4
    )

    results = {
        "noisy_50k_30pct": noisy_aggregated,
        "clean_20k": clean_aggregated,
        "delta": {
            "f1": f1_delta,
            "accuracy": acc_delta,
            "clean_wins": f1_delta > 0,
        },
    }

    save_results(results, f"quantity_vs_quality_{model_name}")

    print(f"\n{'—' * 50}")
    print(
        f"  noisy 50k  | acc={noisy_aggregated['accuracy_mean']:.4f} ± {noisy_aggregated['accuracy_std']:.4f} | f1={noisy_aggregated['f1_mean']:.4f} ± {noisy_aggregated['f1_std']:.4f}"
    )
    print(
        f"  clean 20k  | acc={clean_aggregated['accuracy_mean']:.4f} ± {clean_aggregated['accuracy_std']:.4f} | f1={clean_aggregated['f1_mean']:.4f} ± {clean_aggregated['f1_std']:.4f}"
    )
    print(
        f"  delta      | acc={acc_delta:+.4f} | f1={f1_delta:+.4f} | clean wins: {results['delta']['clean_wins']}"
    )
    print(f"{'—' * 50}")
    print(f"\ndone. results saved to results/quantity_vs_quality_{model_name}.json")

    return results


if __name__ == "__main__":
    for model_name in ["logreg", "distilbert"]:
        run_quantity_vs_quality(model_name)
