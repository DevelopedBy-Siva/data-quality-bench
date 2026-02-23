import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from tqdm import tqdm
from config import SEEDS, NOISE_LEVELS
from data.loader import load_sst2, load_toxicchat
from noise.injector import inject_label_noise, inject_label_noise_conditional
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results


def run_noise_sweep(model_name, dataset):
    print(f"\nrunning noise sweep -- {model_name} on {dataset}")
    print(f"noise levels: {NOISE_LEVELS}")
    print(f"seeds: {SEEDS}\n")

    if dataset == "sst2":
        splits = load_sst2()
        inject_fn = inject_label_noise
    elif dataset == "toxicchat":
        splits = load_toxicchat()
        inject_fn = inject_label_noise_conditional
    else:
        raise ValueError(f"unknown dataset: {dataset}. choose 'sst2' or 'toxicchat'.")

    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    results = {}

    for noise_level in tqdm(NOISE_LEVELS, desc="noise levels"):
        seed_results = []

        for seed in SEEDS:
            noisy_texts, noisy_labels = inject_fn(
                train_texts, train_labels, noise_level, seed=seed
            )

            model = train(model_name, noisy_texts, noisy_labels, seed=seed)
            metrics = evaluate(model, test_texts, test_labels)
            seed_results.append(metrics)

        aggregated = aggregate_across_seeds(seed_results)
        results[str(noise_level)] = aggregated

        _print_row(noise_level, aggregated, dataset)

    save_results(results, f"noise_sweep_{model_name}_{dataset}")
    print(f"\ndone. results saved to results/noise_sweep_{model_name}_{dataset}.json")
    return results


def _print_row(noise_level, aggregated, dataset):
    if dataset == "toxicchat":
        print(
            f"  noise={noise_level:.0%} | "
            f"prauc={aggregated['prauc_mean']:.4f} ± {aggregated['prauc_std']:.4f} | "
            f"f1_macro={aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}"
        )
    else:
        print(
            f"  noise={noise_level:.0%} | "
            f"acc={aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f} | "
            f"f1_macro={aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sst2", "toxicchat"],
        default="toxicchat",
        help="which dataset to run the sweep on",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "deberta"],
        default=["logreg", "deberta"],
        help="which models to run",
    )
    args = parser.parse_args()

    for model_name in args.models:
        run_noise_sweep(model_name, args.dataset)
