"""Phase 6 — the audit. 50 events per (defect x magnitude) on the test split.
Metrics: detection rate (event + 2-day grace), latency (days), FP/day on clean
segments excluding the 14-day echo window after each event (defects inside
the input window poison forecasts => residual echoes)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import PROC, FIG, SENSORS, LOOKBACK, append_metrics
from injector import inject
from detect import Detector

GRACE, TRIALS = 2, 50
SPEC = {"spike":      dict(length=1,  mags=[1.0, 2.0, 4.0, 8.0]),
        "impossible": dict(length=5,  mags=[1.0]),
        "dropout":    dict(length=7,  mags=[1.0]),
        "stuck":      dict(length=10, mags=[1.0]),
        "drift":      dict(length=14, mags=[0.5, 1.0, 2.0]),
        "scale":      dict(length=14, mags=[0.5, 1.0, 2.0])}

def main() -> None:
    det = Detector()
    proc = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"]).reset_index(drop=True)
    test_lo = det.n_train + det.n_val
    seg = proc.iloc[test_lo - det.lookback:].reset_index(drop=True)
    raw = np.load(PROC / "processed.npz", allow_pickle=True)["raw"]
    sig = dict(zip(SENSORS, np.nanstd(raw[: det.n_train], axis=0)))
    rng = np.random.default_rng(7)

    base = det.scan(seg)                                  # clean-run FP/day
    n_clean = len(seg) - det.lookback
    base_fp = int(base["flag"].to_numpy()[det.lookback:].sum())
    print(f"clean test run: {base_fp} flags / {n_clean} days = {base_fp / n_clean:.3f} FP/day\n")

    rows = []
    for kind, spec in SPEC.items():
        for mag in spec["mags"]:
            hits, lats, fps, cds = 0, [], 0, 0
            for _ in range(TRIALS):
                length = spec["length"]
                col = SENSORS[int(rng.integers(len(SENSORS)))]
                lo, hi = det.lookback + 5, len(seg) - length - det.lookback - GRACE - 5
                start = int(rng.integers(lo, hi))
                corrupted, _ = inject(seg, kind, start, length, col, mag, sig)
                flags = det.scan(corrupted)["flag"].to_numpy()
                stop = start + length
                if flags[start: stop + GRACE].any():
                    hits += 1
                    lats.append(int(np.flatnonzero(flags[start: stop + GRACE])[0]))
                echo = np.zeros(len(seg), bool)
                echo[start: stop + GRACE + det.lookback] = True
                clean = np.zeros(len(seg), bool)
                clean[det.lookback:] = True
                clean &= ~echo
                fps += int(flags[clean].sum()); cds += int(clean.sum())
            rows.append(dict(defect=kind, magnitude=mag, n=TRIALS,
                             detection_rate=hits / TRIALS,
                             median_latency_days=(float(np.median(lats)) if lats else np.nan),
                             fp_per_day=fps / max(cds, 1)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(PROC / "audit_results.csv", index=False)

    # spike magnitude sweep on Temperature_C — where does detection hit 50%?
    sweep_mags, rates = np.arange(0.5, 8.01, 0.5), []
    for m in sweep_mags:
        hits = 0
        for _ in range(30):
            start = int(rng.integers(det.lookback + 5, len(seg) - 2 - det.lookback - GRACE))
            flags = det.scan(inject(seg, "spike", start, 1, "Temperature_C", float(m), sig)[0])["flag"].to_numpy()
            hits += int(flags[start: start + 1 + GRACE].any())
        rates.append(hits / 30)
    cross = next((m for m, r in zip(sweep_mags, rates) if r >= 0.5), None)
    cross_txt = (f"~{cross:.1f}σ (≈{cross * sig['Temperature_C']:.1f} °C)" if cross else "never (all ≥50%)")
    print(f"\nspike detection crosses 50% at {cross_txt}")

    append_metrics("\n## Phase 6 — audit (50 events/cell, random channel)\n\n"
                   + table.to_markdown(index=False)
                   + f"\n\n- clean test FP/day: {base_fp / n_clean:.3f}\n"
                   f"- spike 50% crossing: {cross_txt}\n"
                   f"- honest read: drift/stuck latency is expected; "
                   f"scale on pressure is trivially caught (30% of 850 hPa ≈ 250 hPa)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(sweep_mags, rates, "o-"); a1.axhline(0.5, ls="--", color="red")
    a1.set(xlabel="spike magnitude (σ)", ylabel="detection rate", title="spike sweep (Temperature_C)")
    labels = [f"{r.defect} m={r.magnitude:g}" for r in table.itertuples()]
    a2.bar(range(len(table)), table["detection_rate"], color="tab:blue")
    a2.set_xticks(range(len(table))); a2.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    a2.set(ylim=(0, 1.05), ylabel="detection rate", title="per-type results")
    fig.tight_layout(); fig.savefig(FIG / "audit.png", dpi=130)
    print(f"saved {FIG / 'audit.png'}")

if __name__ == "__main__":
    main()
