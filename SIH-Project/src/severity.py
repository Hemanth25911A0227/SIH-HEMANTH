"""Phase 9a — severity tiers + calibrated confidence (pure post-processing,
no retraining, no new dependencies).

Severity bands are anchored to the calibrated detector threshold T (2.76σ):
  T .. 1.5T LOW · .. 3T MEDIUM · .. 5T HIGH · else CRITICAL.
Rule-only catches (score NaN — stuck / missing / impossible-value) get
severity RULE: a physics violation is deterministic, not statistical.

Confidence = P(defect | score) from a logistic least-squares fit (on the
logit) through the empirical spike detection-rate curve in
data/processed/audit_results.csv (hardcoded fallback if missing).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# Ensure Unicode output on Windows (σ, · etc.) without requiring PYTHONUTF8=1
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"

BANDS = [(1.5, "LOW", "#2a9d8f", "#ffffff"),
         (3.0, "MEDIUM", "#ffd166", "#10202b"),
         (5.0, "HIGH", "#f4a261", "#10202b"),
         (float("inf"), "CRITICAL", "#e63946", "#ffffff")]
RULE = ("RULE", "#8ecae6", "#10202b")

# spike detection rates from the Phase 6 audit (magnitude σ -> detection rate)
_FALLBACK = np.array([[1.0, 0.52], [2.0, 0.96], [4.0, 0.99], [8.0, 0.99]])
_AB: tuple[float, float] | None = None


def severity_of(score: float, threshold: float) -> tuple[str, str]:
    """(label, css) for an anomaly score in σ units; NaN score -> RULE."""
    if not np.isfinite(score):
        return RULE[0], f"background:{RULE[1]};color:{RULE[2]}"
    for mult, label, bg, fg in BANDS:
        if score < mult * threshold:
            return label, f"background:{bg};color:{fg}"
    return BANDS[-1][1], f"background:{BANDS[-1][2]};color:{BANDS[-1][3]}"


def _fit() -> tuple[float, float]:
    global _AB
    if _AB is None:
        pts = _FALLBACK
        try:
            import pandas as pd
            t = pd.read_csv(PROC / "audit_results.csv")
            sp = t[t["defect"] == "spike"][["magnitude", "detection_rate"]].to_numpy(float)
            if len(sp) >= 3:
                pts = sp
        except Exception:
            pass
        y = np.clip(pts[:, 1], 0.01, 0.99)
        _AB = tuple(np.polyfit(pts[:, 0], np.log(y / (1 - y)), 1))
    return _AB


def confidence_of(score: float) -> float:
    """P(defect | score); rule-verified catches (NaN score) are ~certain."""
    if not np.isfinite(score):
        return 0.99
    a, b = _fit()
    return float(1.0 / (1.0 + np.exp(-(a * score + b))))


def main() -> None:
    from preprocess import append_metrics
    a, b = _fit()
    threshold = float(np.load(MODELS / "calibration.npz")["threshold"])
    print(f"logistic fit: confidence(x) = sigmoid({a:.3f}·x {'+' if b >= 0 else '-'} "
          f"{abs(b):.3f})   (threshold {threshold:.2f}σ)")
    lines = ["\n## Phase 9a — severity tiers + confidence fit",
             f"- logistic fit: sigmoid({a:.3f}·x {'+' if b >= 0 else '-'} {abs(b):.3f}) "
             f"through spike detection-rate curve"]
    for s in [threshold, 1.5 * threshold, 3 * threshold, 6 * threshold, 16.5]:
        lab, _ = severity_of(s, threshold)
        c = confidence_of(s)
        print(f"  {s:5.1f}σ -> severity {lab:9s} confidence {c:5.1%}")
        lines.append(f"- {s:.1f}σ -> {lab}, confidence {c:.1%}")
    append_metrics("\n".join(lines))


if __name__ == "__main__":
    main()
