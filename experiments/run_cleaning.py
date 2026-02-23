import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from tqdm import tqdm
from config import SEEDS, NOISE_LEVELS, TOXICCHAT_CONFIG
from data.loader import load_sst2, load_toxicchat
from noise.injector import inject_label_noise, inject_label_noise_conditional
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results
from cleaning.strategies import confidence_filter, loss_filter, heuristic_filter


def run_cleaning(model_name, dataset):
    print(f"\nrunning cleaning experiment -- {model_name} on {dataset}")

    noise_levels = [nl for nl in NOISE_LEVELS if nl > 0.0]
    print(f"noise levels: {noise_levels}")
    print(f"seeds: {SEEDS}\n")

    if dataset == "sst2":
        splits = load_sst2()
        inject_fn = inject_label_noise
        min_toxic = None
    elif dataset == "toxicchat":
        splits = load_toxicchat()
        inject_fn = inject_label_noise_conditional
        min_toxic = TOXICCHAT_CONFIG["min_toxic_samples"]
    else:
        raise ValueError(f"unknown dataset: {dataset}. choose 'sst2' or 'toxicchat'.")

    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    cleaning_strategies = {
        "confidence": lambda texts, labels, model: confidence_filter(
            texts, labels, model, min_toxic_samples=min_toxic
        ),
        "loss": lambda texts, labels, model: loss_filter(
            texts, labels, model, min_toxic_samples=min_toxic
        ),
        "heuristic": lambda texts, labels, _model: heuristic_filter(
            texts, labels, min_toxic_samples=min_toxic
        ),
    }

    results = {}

    for noise_level in tqdm(noise_levels, desc="noise levels"):
        results[str(noise_level)] = {}

        noisy_baseline_results = []
        cleaned_results = {strategy: [] for strategy in cleaning_strategies}

        for seed in SEEDS:
            noisy_texts, noisy_labels = inject_fn(
                train_texts, train_labels, noise_level, seed=seed
            )

            noisy_model = train(model_name, noisy_texts, noisy_labels, seed=seed)
            noisy_metrics = evaluate(noisy_model, test_texts, test_labels)
            noisy_baseline_results.append(noisy_metrics)

            for strategy_name, strategy_fn in cleaning_strategies.items():
                cleaned_texts, cleaned_labels = strategy_fn(
                    noisy_texts, noisy_labels, noisy_model
                )

                cleaned_model = train(
                    model_name, cleaned_texts, cleaned_labels, seed=seed
                )
                cleaned_metrics = evaluate(cleaned_model, test_texts, test_labels)
                cleaned_results[strategy_name].append(cleaned_metrics)

        results[str(noise_level)]["noisy_baseline"] = aggregate_across_seeds(
            noisy_baseline_results
        )

        for strategy_name, seed_results in cleaned_results.items():
            results[str(noise_level)][strategy_name] = aggregate_across_seeds(
                seed_results
            )

        _print_noise_level_summary(noise_level, results[str(noise_level)], dataset)

    save_results(results, f"cleaning_{model_name}_{dataset}")
    print(f"\ndone. results saved to results/cleaning_{model_name}_{dataset}.json")
    return results


def _print_noise_level_summary(noise_level, level_results, dataset):
    print(f"\n  noise={noise_level:.0%}")
    primary = "prauc" if dataset == "toxicchat" else "accuracy"
    for strategy, metrics in level_results.items():
        print(
            f"    {strategy:<20} | "
            f"{primary}={metrics[f'{primary}_mean']:.4f} ± {metrics[f'{primary}_std']:.4f} | "
            f"f1_macro={metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sst2", "toxicchat"],
        default="toxicchat",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "deberta"],
        default=["logreg", "deberta"],
    )
    args = parser.parse_args()

    for model_name in args.models:
        run_cleaning(model_name, args.dataset)
