from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
except ImportError as e:
    raise ImportError(
        "gate/noise_estimator.py requires scikit-learn. "
        "Run: pip install scikit-learn"
    ) from e

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_DEFAULT_BASELINE = "toxicchat"


@dataclass
class BatchSignals:
    n_samples: int
    toxic_rate: float
    toxic_rate_drift: float
    mean_entropy: float
    mean_margin: float
    near_threshold: float
    estimated_prauc: float
    estimated_noise: float
    noise_band: str
    recommended_action: str
    baseline_name: str


def _binary_entropy(p: float) -> float:
    p = max(min(p, 1 - 1e-9), 1e-9)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class _Baseline:
    """
    Loads a baseline JSON from results/<name>_baseline.json.
    Exposes signal curves for interpolation and PR-AUC curve for estimation.
    """

    def __init__(self, name: str):
        path = _RESULTS_DIR / f"{name}_baseline.json"
        if not path.exists():
            available = list_baselines()
            hint = (
                f"Available baselines: {', '.join(available)}"
                if available
                else "No baselines found. Run gate/calibrate.py first."
            )
            raise FileNotFoundError(
                f"Baseline not found: {path}\n"
                f"{hint}\n\n"
                f"To calibrate a new baseline:\n"
                f"  python gate/calibrate.py --dataset toxicchat --name toxicchat\n"
                f"  python gate/calibrate.py --csv my_data.csv --name my_dataset"
            )

        with open(path) as f:
            data = json.load(f)

        self.name = data["name"]
        self.toxic_rate = data["toxic_rate"]
        self.tipping_point = data["tipping_point"]
        self.noise_levels = np.array(data["noise_levels"])
        self.entropy_curve = np.array(data["curves"]["entropy"])
        self.margin_curve = np.array(data["curves"]["margin"])
        self.near_curve = np.array(data["curves"]["near"])

        prauc_means = [
            data["sweep_results"][str(n)]["prauc_mean"] for n in data["noise_levels"]
        ]
        self.prauc_means = np.array(prauc_means)
        self.clean_prauc = float(self.prauc_means[0])

        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(-self.prauc_means, self.noise_levels)

    def signal_votes(
        self,
        mean_entropy: float,
        mean_margin: float,
        near_threshold: float,
    ) -> tuple[float, float, float]:
        """Interpolate each signal against its curve → noise vote."""
        e = float(np.interp(mean_entropy, self.entropy_curve, self.noise_levels))
        m = float(
            np.interp(mean_margin, self.margin_curve[::-1], self.noise_levels[::-1])
        )
        n = float(np.interp(near_threshold, self.near_curve, self.noise_levels))
        return e, m, n

    def noise_to_prauc(self, noise: float) -> float:
        return round(float(np.interp(noise, self.noise_levels, self.prauc_means)), 4)

    def cliff_signals(self) -> tuple[float, float, float]:
        """Return (entropy, margin, near) at the tipping point noise level."""
        levels = list(self.noise_levels)
        idx = (
            levels.index(self.tipping_point)
            if self.tipping_point in levels
            else len(levels) // 2
        )
        return (
            float(self.entropy_curve[idx]),
            float(self.margin_curve[idx]),
            float(self.near_curve[idx]),
        )


def _batch_cross_val_proba(
    texts: list[str],
    labels: list[int],
    cv: int = 3,
) -> np.ndarray:
    labels_arr = np.array(labels)
    n_pos = int(labels_arr.sum())
    n_cv = cv

    if n_pos < cv * 2:
        if n_pos >= 4:
            n_cv = 2
        else:
            warnings.warn(
                f"Only {n_pos} positive examples — entropy/margin signals unreliable. "
                f"Drift signal will carry full weight.",
                UserWarning,
                stacklevel=3,
            )
            return np.full(len(texts), 0.5)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced"
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(
            pipe, texts, labels_arr, cv=n_cv, method="predict_proba"
        )

    return proba[:, 1]


class NoiseEstimator:
    """
    Estimates label noise rate of a new batch from observable statistics.

    Usage:
        # default: uses toxicchat baseline
        estimator = NoiseEstimator()

        # custom baseline (run gate/calibrate.py first)
        estimator = NoiseEstimator(baseline="my_dataset")

        signals = estimator.estimate(batch_texts, batch_labels)
        print(signals.estimated_noise)
        print(signals.noise_band)
        print(signals.recommended_action)
    """

    _BANDS = [
        (0.00, 0.10, "CLEAN", "Safe to retrain."),
        (
            0.10,
            0.18,
            "WATCH",
            "Mild noise detected. Consider loss filtering before retraining.",
        ),
        (
            0.18,
            0.25,
            "DANGER",
            "Near or past the cliff. Apply loss filtering. Send boundary examples to adjudication.",
        ),
        (
            0.25,
            1.00,
            "CRITICAL",
            "Severe noise. Do not retrain. Audit label sources before proceeding.",
        ),
    ]

    def __init__(self, baseline: str = _DEFAULT_BASELINE):
        self._bl = _Baseline(baseline)

    def estimate(
        self,
        batch_texts: list[str],
        batch_labels: list[int],
        cv: int = 3,
    ) -> BatchSignals:
        n = len(batch_texts)
        if n < 50:
            raise ValueError(f"Batch too small ({n}). Need at least 50 samples.")

        print(f"  Running {cv}-fold cross-val probe on batch ({n} samples)...")
        p_toxic = _batch_cross_val_proba(batch_texts, batch_labels, cv=cv)

        entropies = np.array([_binary_entropy(float(p)) for p in p_toxic])
        margins = np.abs(p_toxic - 0.5)
        near_mask = (p_toxic >= 0.4) & (p_toxic <= 0.6)

        toxic_rate = float(np.mean(np.array(batch_labels)))
        toxic_rate_drift = abs(toxic_rate - self._bl.toxic_rate)
        mean_entropy = float(np.mean(entropies))
        mean_margin = float(np.mean(margins))
        near_threshold = float(np.mean(near_mask))

        e_vote, m_vote, n_vote = self._bl.signal_votes(
            mean_entropy, mean_margin, near_threshold
        )

        d_vote = float(
            np.clip(toxic_rate_drift / max(self._bl.toxic_rate, 0.01) * 0.10, 0, 0.40)
        )

        signal_noise = float(
            np.clip(
                0.35 * e_vote + 0.35 * m_vote + 0.20 * n_vote + 0.10 * d_vote, 0.0, 0.40
            )
        )

        estimated_noise = round(signal_noise, 3)
        estimated_prauc = self._bl.noise_to_prauc(signal_noise)
        band, action = self._get_band(estimated_noise)

        return BatchSignals(
            n_samples=n,
            toxic_rate=round(toxic_rate, 4),
            toxic_rate_drift=round(toxic_rate_drift, 4),
            mean_entropy=round(mean_entropy, 4),
            mean_margin=round(mean_margin, 4),
            near_threshold=round(near_threshold, 4),
            estimated_prauc=estimated_prauc,
            estimated_noise=estimated_noise,
            noise_band=band,
            recommended_action=action,
            baseline_name=self._bl.name,
        )

    def _get_band(self, noise: float) -> tuple[str, str]:
        for lo, hi, band, action in self._BANDS:
            if lo <= noise < hi:
                return band, action
        return "CRITICAL", self._BANDS[-1][3]

    @property
    def baseline(self) -> _Baseline:
        return self._bl


def list_baselines() -> list[str]:
    if not _RESULTS_DIR.exists():
        return []
    return [
        p.stem.replace("_baseline", "") for p in _RESULTS_DIR.glob("*_baseline.json")
    ]


def print_calibration_curve(baseline: str = _DEFAULT_BASELINE) -> None:
    b = _Baseline(baseline)
    cliff_e, cliff_m, cliff_n = b.cliff_signals()

    print(f"\nCalibration curve — {b.name}")
    print(
        f"  Toxic rate: {b.toxic_rate*100:.1f}%  |  "
        f"Tipping point: {b.tipping_point:.0%}  |  "
        f"Clean PR-AUC: {b.clean_prauc:.4f}"
    )
    print(
        f"\n  {'Noise':>8}  {'PR-AUC':>8}  {'Entropy':>9}  {'Margin':>8}  {'Near':>8}"
    )
    print(f"  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*8}")
    for i, noise in enumerate(b.noise_levels):
        marker = "  ← cliff" if noise == b.tipping_point else ""
        print(
            f"  {noise:>8.0%}  {b.prauc_means[i]:>8.4f}  "
            f"{b.entropy_curve[i]:>9.4f}  {b.margin_curve[i]:>8.4f}  "
            f"{b.near_curve[i]:>8.4f}{marker}"
        )

    available = list_baselines()
    print(f"\n  Available baselines: {', '.join(available) if available else 'none'}\n")


if __name__ == "__main__":
    print_calibration_curve()
