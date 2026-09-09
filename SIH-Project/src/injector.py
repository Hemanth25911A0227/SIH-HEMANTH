"""
Phase 2 — defect injector (test harness for every later phase).
6 kinds: spike, impossible, stuck, drift, scale, dropout.
Returns (corrupted_df, boolean labels aligned to df).
df must have a RangeIndex and Timestamp column.
Note: 'impossible' pressure = 1013 hPa (sea-level value — impossible at
1587 m station level; also what a misconfigured sensor would report).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import PROC, FIG, SENSORS, LOOKBACK, append_metrics

IMPOSSIBLE_VALUES = {"Temperature_C": 60.0, "DewPoint_C": 45.0, "Pressure_hPa": 1013.0}
DEFECT_KINDS = ["spike", "impossible", "stuck", "drift", "scale", "dropout"]

def inject(df: pd.DataFrame, kind: str, start: int, length: int, col: str,
           magnitude: float = 1.0, sigma: dict | None = None):
    assert kind in DEFECT_KINDS
    assert isinstance(df.index, pd.RangeIndex), "df must be reset_index(drop=True)"
    assert 0 < start and start + length <= len(df)
    out, labels = df.copy(), np.zeros(len(df), dtype=bool)
    ci, s = out.columns.get_loc(col), (sigma or {}).get(col, 1.0)

    if kind == "spike":                      # single-day jump of magnitude*sigma
        out.iloc[start, ci] = out.iloc[start, ci] + magnitude * s
        labels[start] = True
        return out, labels
    stop = start + length
    labels[start:stop] = True
    if kind == "impossible":
        out.iloc[start:stop, ci] = IMPOSSIBLE_VALUES[col]
    elif kind == "stuck":                    # frozen at pre-event value
        out.iloc[start:stop, ci] = out.iloc[start - 1, ci]
    elif kind == "drift":                    # linear ramp reaching magnitude*sigma
        out.iloc[start:stop, ci] = out.iloc[start:stop, ci] + np.linspace(0, magnitude * s, length)
    elif kind == "scale":                    # multiplicative miscalibration
        out.iloc[start:stop, ci] = out.iloc[start:stop, ci] * (1 + 0.3 * magnitude)
    elif kind == "dropout":
        out.iloc[start:stop, ci] = np.nan
    return out, labels

def main() -> None:
    d = np.load(PROC / "processed.npz", allow_pickle=True)
    df = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"]).reset_index(drop=True)
    seg = df.iloc[int(d["n_train"]) + int(d["n_val"]) - LOOKBACK:].reset_index(drop=True)
    sig = dict(zip(SENSORS, np.nanstd(d["raw"][: int(d["n_train"])], axis=0)))

    corrupted, labels = inject(seg, "spike", start=len(seg) // 2, length=1,
                               col="Temperature_C", magnitude=6.0, sigma=sig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(corrupted["Timestamp"]), corrupted["Temperature_C"], lw=0.9)
    lab = np.flatnonzero(labels)
    if len(lab):
        ax.axvspan(pd.to_datetime(corrupted["Timestamp"]).iloc[lab[0]],
                   pd.to_datetime(corrupted["Timestamp"]).iloc[lab[-1]], color="red", alpha=0.2)
    ax.set_title("injector selftest — +6σ temperature spike (red = label window)")
    fig.tight_layout(); fig.savefig(FIG / "injector_selftest.png", dpi=120)
    print(f"saved {FIG / 'injector_selftest.png'} — eyeball jump vs label window")

    for k in DEFECT_KINDS:  # smoke-test all kinds
        c, l = inject(seg, k, len(seg) // 3, 5, "Pressure_hPa", 1.0, sig)
        print(f"{k:10s} ok, labeled days: {int(l.sum())}")

if __name__ == "__main__":
    main()
