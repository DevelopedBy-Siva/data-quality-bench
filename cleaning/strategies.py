import numpy as np
from config import CLEANING_CONFIG, TOXICCHAT_CONFIG


def _apply_guardrail(texts, labels, model, min_toxic_samples):
    """
    After filtering, if toxic samples drop below the minimum threshold,
    adds back the highest-confidence toxic samples until the floor is met.
    This prevents cleaning from effectively destroying the minority class.
    """
    toxic_indices = [i for i, l in enumerate(labels) if l == 1]
    n_toxic = len(toxic_indices)

    if n_toxic >= min_toxic_samples:
        return texts, labels

    n_to_restore = min_toxic_samples - n_toxic
    print(
        f"    guardrail triggered: only {n_toxic} toxic samples remain, restoring {n_to_restore}"
    )

    proba = model.predict_proba(texts)
    non_toxic_indices = [i for i, l in enumerate(labels) if l == 0]

    scored = sorted(non_toxic_indices, key=lambda i: proba[i][1], reverse=True)
    to_restore = scored[:n_to_restore]

    restored_texts = [texts[i] for i in to_restore]
    restored_labels = [1] * len(to_restore)

    return list(texts) + restored_texts, list(labels) + restored_labels


def confidence_filter(texts, labels, model, min_toxic_samples=None):
    """
    Removes samples where the model's confidence is below the threshold.
    Low confidence usually means the sample is ambiguous or mislabeled.
    Falls back to top 50% most confident if the threshold is too aggressive.
    Applies the per-class guardrail when min_toxic_samples is set.
    """
    if min_toxic_samples is None:
        min_toxic_samples = TOXICCHAT_CONFIG["min_toxic_samples"]

    proba = model.predict_proba(texts)
    max_confidence = np.max(proba, axis=1)
    threshold = CLEANING_CONFIG["confidence_threshold"]

    keep = [i for i, conf in enumerate(max_confidence) if conf >= threshold]

    if len(keep) == 0 or len(set(labels[i] for i in keep)) < 2:
        cutoff = np.median(max_confidence)
        keep = [i for i, conf in enumerate(max_confidence) if conf >= cutoff]

    if len(set(labels[i] for i in keep)) < 2:
        return texts.copy(), labels.copy()

    filtered_texts, filtered_labels = _subset(texts, labels, keep)
    return _apply_guardrail(filtered_texts, filtered_labels, model, min_toxic_samples)


def loss_filter(texts, labels, model, min_toxic_samples=None):
    """
    Removes the highest-loss samples -- the ones the model struggles with most.
    High loss correlates strongly with mislabeled examples.
    Applies the per-class guardrail when min_toxic_samples is set.
    """
    if min_toxic_samples is None:
        min_toxic_samples = TOXICCHAT_CONFIG["min_toxic_samples"]

    losses = model.get_loss_per_sample(texts, labels)
    cutoff = np.percentile(losses, CLEANING_CONFIG["loss_percentile"])

    keep = [i for i, loss in enumerate(losses) if loss <= cutoff]

    if len(set(labels[i] for i in keep)) < 2:
        return texts.copy(), labels.copy()

    filtered_texts, filtered_labels = _subset(texts, labels, keep)
    return _apply_guardrail(filtered_texts, filtered_labels, model, min_toxic_samples)


def heuristic_filter(texts, labels, min_toxic_samples=None):
    """
    No model needed -- drops duplicates and samples too short to carry signal.
    min_toxic_samples is accepted for interface consistency but guardrail here
    is just a hard count check since we have no model to score with.
    """
    if min_toxic_samples is None:
        min_toxic_samples = TOXICCHAT_CONFIG["min_toxic_samples"]

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

    if len(keep) == 0 or len(set(labels[i] for i in keep)) < 2:
        return texts.copy(), labels.copy()

    filtered_texts, filtered_labels = _subset(texts, labels, keep)

    n_toxic = sum(1 for l in filtered_labels if l == 1)
    if n_toxic < min_toxic_samples:
        print(
            f"    heuristic guardrail: {n_toxic} toxic samples after filter, returning original"
        )
        return texts.copy(), labels.copy()

    return filtered_texts, filtered_labels


def apply_all(texts, labels, model, min_toxic_samples=None):
    """
    Runs all three strategies in sequence.
    Heuristic first since it doesn't need the model and is fast.
    Each step passes the guardrail floor through so the minority class survives.
    Returns cleaned texts and labels plus a summary of what got removed.
    """
    if min_toxic_samples is None:
        min_toxic_samples = TOXICCHAT_CONFIG["min_toxic_samples"]

    original_size = len(texts)

    texts, labels = heuristic_filter(texts, labels, min_toxic_samples)
    after_heuristic = len(texts)

    if len(texts) > 0:
        texts, labels = confidence_filter(texts, labels, model, min_toxic_samples)
    after_confidence = len(texts)

    if len(texts) > 0:
        texts, labels = loss_filter(texts, labels, model, min_toxic_samples)
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
