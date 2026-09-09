"""
Phase 3 — Layer 1 (physics/climatology) + Layer 3 (flatline) + persistence
baseline. IMPORTANT: pressure is STATION-LEVEL at 1587 m (~830-860 hPa);
the roadmap's generic 900-1080 hPa sea-level rule would flag 100% of your
clean data. Bounds come from climatological.csv when SRINAGAR is present,
else from the train split (source is printed).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from preprocess import PROC, RAW, SENSORS, LOOKBACK, append_metrics
from injector import inject

FLAT_K = 5        # identical trailing days => stuck sensor
DEW_MARGIN = 0.5  # dew point may not exceed temperature (+ rounding slack)

def build_bounds(train_df: pd.DataFrame, clim: pd.DataFrame | None, station: str):
    temp, pressure, src = {}, None, "train-split fallback"
    if clim is not None:
        sub = clim[clim["station_name"].astype(str).str.upper().str.strip()
                   == station.upper().strip()]
        if len(sub) >= 12:
            for _, r in sub.iterrows():
                m = int(r["month"])
                if pd.notna(r["min_low_temp_value"]) and pd.notna(r["max_high_temp_value"]):
                    temp[m] = (float(r["min_low_temp_value"]) - 5.0,
                               float(r["max_high_temp_value"]) + 5.0)
            p = np.nanmean(sub[["mean_station_level_pressure_in_hpa_03UTC",
                                "mean_station_level_pressure_in_hpa_12UTC"]].to_numpy(float))
            if np.isfinite(p):
                pressure, src = (p - 30.0, p + 30.0), "climatological.csv"
    if len(temp) < 12 or pressure is None:
        tmp = train_df.copy(); tmp["m"] = tmp["Timestamp"].dt.month
        for m, q in tmp.groupby("m"):
            temp.setdefault(int(m), (q["Temperature_C"].min() - 5.0,
                                     q["Temperature_C"].max() + 5.0))
        if pressure is None:
            pressure = (train_df["Pressure_hPa"].min() - 12.0,
                        train_df["Pressure_hPa"].max() + 12.0)
    return {"temp": temp, "pressure": pressure, "source": src}

def check_row(values, month: int, bounds) -> list[str]:
    """Layer 1 for one day. values = [T, Td, P] (raw, may contain NaN)."""
    t, td, p = values
    if any(pd.isna(v) for v in values):
        return ["missing reading"]
    reasons = []
    lo, hi = bounds["temp"].get(int(month), (-1e9, 1e9))
    if not (lo <= t <= hi):
        reasons.append(f"temperature {t:.1f}C outside climatological range [{lo:.0f},{hi:.0f}] (month {month})")
    if td > t + DEW_MARGIN:
        reasons.append(f"dew point {td:.1f}C above temperature {t:.1f}C (RH>100%)")
    plo, phi = bounds["pressure"]
    if not (plo <= p <= phi):
        reasons.append(f"pressure {p:.0f} hPa outside station-level range [{plo:.0f},{phi:.0f}]")
    return reasons

def check_flatline(arr: np.ndarray, i: int, k: int = FLAT_K) -> list[str]:
    """Layer 3: k identical trailing values on any channel. arr: (T, 3) raw."""
    if i < k - 1:
        return []
    reasons = []
    for j, c in enumerate(SENSORS):
        w = arr[i - k + 1: i + 1, j]
        if np.isfinite(w).all() and np.ptp(w) < 1e-9:
            reasons.append(f"stuck sensor: {c} identical for {k} days")
    return reasons

def main() -> None:
    d = np.load(PROC / "processed.npz", allow_pickle=True)
    proc = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"]).reset_index(drop=True)
    clim = pd.read_csv(RAW / "climatological.csv") if (RAW / "climatological.csv").exists() else None
    try:
        station = pd.read_csv(RAW / "srinagar_daily.csv", nrows=1)["Station_Name"].iloc[0]
    except Exception:
        station = "SRINAGAR"
    bounds = build_bounds(proc.iloc[: int(d["n_train"])], clim, station)
    print(f"bounds source: {bounds['source']} | pressure: {bounds['pressure']}")

    seg = proc.iloc[int(d["n_train"]) + int(d["n_val"]) - LOOKBACK:].reset_index(drop=True)
    sig = dict(zip(SENSORS, np.nanstd(d["raw"][: int(d["n_train"])], axis=0)))

    for kind, col, ln in [("impossible", "Temperature_C", 5), ("dropout", "DewPoint_C", 7),
                          ("stuck", "Pressure_hPa", 10)]:
        start = len(seg) // 2
        corrupted, labels = inject(seg, kind, start, ln, col, 1.0, sig)
        arr = corrupted[SENSORS].to_numpy(float)
        months = corrupted["Timestamp"].dt.month.to_numpy()
        flagged = [i for i in range(start, start + int(labels.sum()))
                   if check_row(arr[i], months[i], bounds) + check_flatline(arr, i)]
        print(f"rules selftest — {kind:10s} on {col:14s}: event caught={bool(flagged)} "
              f"({len(flagged)}/{int(labels.sum())} event days flagged)")

    # persistence baseline (val split): predict tomorrow = today
    raw = d["raw"]; n_tr, n_va = int(d["n_train"]), int(d["n_val"])
    diff = raw[n_tr: n_tr + n_va][1:] - raw[n_tr: n_tr + n_va][:-1]
    mae, mse = np.nanmean(np.abs(diff), 0), np.nanmean(diff ** 2, 0)
    lines = ["\n## Phase 3 — persistence baseline (val, 1-day horizon)"]
    for c, a, m in zip(SENSORS, mae, mse):
        print(f"persistence val {c:14s} MAE {a:6.2f}  MSE {m:7.2f}")
        lines.append(f"- {c}: MAE {a:.2f}, MSE {m:.2f}")
    append_metrics("\n".join(lines))

if __name__ == "__main__":
    main()
