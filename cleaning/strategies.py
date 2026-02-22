import numpy as np
from config import CLEANING_CONFIG


def confidence_filter(texts, labels, model):
    """
    Removes samples where the model's confidence is below the threshold.
    Low confidence usually means the sample is ambiguous or mislabeled.
    Keeps only the samples the model feels sure about.
    """
    proba = model.predict_proba(texts)
    max_confidence = np.max(proba, axis=1)
    threshold = CLEANING_CONFIG["confidence_threshold"]

    keep = [i for i, conf in enumerate(max_confidence) if conf >= threshold]

    return _subset(texts, labels, keep)


def loss_filter(texts, labels, model):
    """
    Removes the highest-loss samples — the ones the model struggles with most.
    High loss strongly correlates with mislabeled examples.
    Cuts everything above the configured loss percentile.
    """
    losses = model.get_loss_per_sample(texts, labels)
    cutoff = np.percentile(losses, CLEANING_CONFIG["loss_percentile"])

    keep = [i for i, loss in enumerate(losses) if loss <= cutoff]

    return _subset(texts, labels, keep)


def heuristic_filter(texts, labels):
    """
    No model needed here — just removes obvious garbage.
    Drops duplicates and samples that are too short to carry any signal.
    """
    min_tokens = CLEANING_CONFIG["min_token_length"]
    seen = set()
    keep = []

    for i, text in enumerate(texts):
        if len(text.split()) < min_tokens:
            continue
        if text in seen:
            continue
        seen.add(text)
        keep.append(i)

    return _subset(texts, labels, keep)


def apply_all(texts, labels, model):
    """
    Runs all three strategies in sequence.
    Heuristic first since it doesn't need the model and is fast.
    Then confidence and loss filtering on what's left.
    Returns cleaned texts and labels plus a small summary of what got removed.
    """
    original_size = len(texts)

    texts, labels = heuristic_filter(texts, labels)
    after_heuristic = len(texts)

    texts, labels = confidence_filter(texts, labels, model)
    after_confidence = len(texts)

    texts, labels = loss_filter(texts, labels, model)
    after_loss = len(texts)

    summary = {
        "original": original_size,
        "after_heuristic": after_heuristic,
        "after_confidence": after_confidence,
        "after_loss": after_loss,
        "removed_total": original_size - after_loss,
        "removed_pct": round((original_size - after_loss) / original_size * 100, 1),
    }

    return texts, labels, summary


def _subset(texts, labels, indices):
    return (
        [texts[i] for i in indices],
        [labels[i] for i in indices],
    )
