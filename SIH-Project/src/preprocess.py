"""
Phase 1 — clean, gap-check, feature-engineer, chronological split, scale.
Adapted to daily data. Gaps stay NaN (real dropout defects); training skips
windows containing NaN. Scaler is fit on the TRAIN split only.
Done when: data/processed/processed.npz exists and you can state
"<N> rows, 3 sensors, 7 features".
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS, FIG = ROOT / "models", ROOT / "data" / "figures"
for d in (RAW, PROC, MODELS, FIG):
    d.mkdir(parents=True, exist_ok=True)

SENSORS = ["Temperature_C", "DewPoint_C", "Pressure_hPa"]
N_SENSORS = len(SENSORS)
FEATURE_COLS = SENSORS + ["sin1", "cos1", "sin2", "cos2"]   # 7 features
LOOKBACK = 14   # days (roadmap used 24h on hourly data)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["Timestamp"].dt.dayofyear.astype(float)
    out["sin1"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos1"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin2"] = np.sin(4 * np.pi * doy / 365.25)
    out["cos2"] = np.cos(4 * np.pi * doy / 365.25)
    return out


def append_metrics(text: str) -> None:
    with open(ROOT / "metrics.md", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def main() -> None:
    df = pd.read_csv(RAW / "srinagar_daily.csv", parse_dates=["Timestamp"])
    df = (df.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True))

    # gap check: reindex to a complete daily calendar
    s = df.set_index("Timestamp")[SENSORS]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D", name="Timestamp")
    full = s.reindex(full_idx).reset_index()

    absent = full[SENSORS].isna().all(axis=1)
    partial = full[SENSORS].isna().any(axis=1) & ~absent
    print(f"rows in file      : {len(df)}")
    print(f"calendar days     : {len(full)} ({full.Timestamp.min().date()} -> {full.Timestamp.max().date()})")
    print(f"absent days (gap) : {int(absent.sum())} {list(full.loc[absent, 'Timestamp'].dt.date)}")
    print(f"partial NaN rows  : {int(partial.sum())}")

    feats = add_calendar(full)
    arr = feats[FEATURE_COLS].to_numpy(float)
    n = len(arr)
    n_train, n_val = int(0.70 * n), int(0.15 * n)

    mu = np.nanmean(arr[:n_train], axis=0)          # train-only scaler (nan-aware)
    sigma = np.nanstd(arr[:n_train], axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xz = (arr - mu) / sigma

    np.savez(PROC / "processed.npz", X=Xz, raw=full[SENSORS].to_numpy(float),
             dates=full["Timestamp"].to_numpy(), mu=mu, sigma=sigma,
             n_train=n_train, n_val=n_val, lookback=LOOKBACK,
             sensors=np.array(SENSORS), feature_cols=np.array(FEATURE_COLS))
    full.to_csv(PROC / "processed.csv", index=False)

    print(f"\nsummary: {n} rows, {N_SENSORS} sensors, {len(FEATURE_COLS)} features "
          f"(splits {n_train}/{n_val}/{n - n_train - n_val})")
    append_metrics(f"\n## Phase 1 — data\n- {n} daily rows, {N_SENSORS} sensors, "
                   f"{len(FEATURE_COLS)} features, splits {n_train}/{n_val}/{n-n_train-n_val}\n"
                   f"- absent days (real dropouts): {int(absent.sum())}")


if __name__ == "__main__":
    main()
