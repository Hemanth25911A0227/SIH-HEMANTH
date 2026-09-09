"""Phase 7 — one-command demo. Streams the test split, injects a defect
mid-stream, prints a live log + the 'money shot', and saves a figure.
    python src/demo.py --scenario spike|drift|stuck|scale|real
'real' scans YOUR full series for genuine defects (e.g. the 2019-08-08/09 gap)."""
from __future__ import annotations
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import PROC, FIG, SENSORS
from injector import inject
from detect import Detector
from severity import severity_of, confidence_of

# Ensure Unicode output on Windows (σ etc.) without requiring PYTHONUTF8=1
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SCENARIOS = {"spike": dict(kind="spike", col="Temperature_C", length=1,  magnitude=6.0),
             "drift": dict(kind="drift", col="Temperature_C", length=14, magnitude=4.0),
             "stuck": dict(kind="stuck", col="Pressure_hPa",  length=10, magnitude=1.0),
             "scale": dict(kind="scale", col="DewPoint_C",    length=21, magnitude=1.0)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="spike", choices=[*SCENARIOS, "real"])
    a = ap.parse_args()
    det = Detector()
    proc = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"]).reset_index(drop=True)

    if a.scenario == "real":
        res = det.scan(proc).iloc[det.lookback:]
        flags = res[res["flag"]]
        print(f"scanning YOUR data ({len(res)} days) — {len(flags)} flagged\n")
        for _, r in flags.iterrows():
            s = f"{r['score']:6.1f}" if pd.notna(r["score"]) else "   .  "
            print(f"  {r['Timestamp'].date()}  {s}  {r['reasons']}")
        print("\n(expect the 2019-08-08/09 gap as 'missing reading'; "
              "train-segment residuals are in-sample — trust val/test flags most)")
        return

    seg = proc.iloc[det.n_train + det.n_val - det.lookback:].reset_index(drop=True)
    sig = dict(zip(SENSORS, np.nanstd(np.load(PROC / "processed.npz", allow_pickle=True)
                                      ["raw"][: det.n_train], axis=0)))
    sc = SCENARIOS[a.scenario]
    start = int(len(seg) * 0.4)
    corrupted, labels = inject(seg, sc["kind"], start, sc["length"], sc["col"], sc["magnitude"], sig)
    res = det.scan(corrupted)

    print(f"scenario: {sc['kind']} on {sc['col']} (mag {sc['magnitude']}, {sc['length']}d) "
          f"at {res['Timestamp'].iloc[start].date()}\n\n   date        score   verdict")
    for i in range(det.lookback, len(res)):
        r = res.iloc[i]
        if r["flag"] or i % 7 == 0:
            s = f"{r['score']:6.1f}" if pd.notna(r["score"]) else "   .  "
            print(f"{'>>>' if r['flag'] else '   '} {r['Timestamp'].date()}  {s}  {r['reasons']}")
    flagged = res[res["flag"] & (res.index >= det.lookback)]
    if len(flagged):
        # Prefer the highest-scoring flag (the injected spike), fallback to first flag
        finite_sc = flagged[pd.notna(flagged["score"])]
        f0 = flagged.loc[finite_sc["score"].idxmax()] if len(finite_sc) else flagged.iloc[0]
        s0 = float(f0["score"]) if pd.notna(f0["score"]) else float("nan")
        sev, _ = severity_of(s0, det.threshold)
        print(f"\nMONEY SHOT: {f0['Timestamp'].date()} — score "
              f"{'nan' if not np.isfinite(s0) else f'{s0:.1f}'}σ "
              f"[{sev} · {confidence_of(s0):.0%} conf] — reason: {f0['reasons']}")


    dates = pd.to_datetime(res["Timestamp"])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 2]})
    for ax, c in zip(axes[:3], SENSORS):
        ax.plot(dates, res[c], lw=0.9); ax.set_ylabel(c)
        ax.axvspan(dates.iloc[start], dates.iloc[start + sc["length"] - 1], color="red", alpha=0.15)
    ax = axes[3]
    ax.plot(dates, res["score"], color="k", lw=0.9, label="max |z|")
    ax.axhline(det.threshold, color="red", ls="--", label=f"threshold {det.threshold:.1f}σ")
    fl = res["flag"].to_numpy()
    ax.scatter(dates[fl], res["score"].to_numpy()[fl], color="red", s=20, zorder=3, label="flag")
    ax.set_ylabel("anomaly score"); ax.legend()
    fig.suptitle(f"demo — {a.scenario} defect on {sc['col']}")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIG / f"demo_{a.scenario}.png", dpi=130)
    print(f"\nsaved {FIG / f'demo_{a.scenario}.png'}")

if __name__ == "__main__":
    main()
