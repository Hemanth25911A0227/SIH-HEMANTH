"""Phase 5b — Layer 1 physics (rules) + Layer 2 forecast residual (LSTM,
calibrated) + Layer 3 flatline. scan(df) -> df + flag/score/reasons columns.
score = max|z| (NaN when Layer 2 is unavailable, e.g. NaN inside the window)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from preprocess import (RAW, PROC, MODELS, SENSORS, N_SENSORS, FEATURE_COLS,
                        LOOKBACK, add_calendar, append_metrics)
from rules import build_bounds, check_row, check_flatline
from model import ForecastLSTM

class Detector:
    def __init__(self):
        d = np.load(PROC / "processed.npz", allow_pickle=True)
        self.mu, self.sigma = d["mu"].astype(np.float32), d["sigma"].astype(np.float32)
        self.n_train, self.n_val, self.lookback = int(d["n_train"]), int(d["n_val"]), int(d["lookback"])
        ck = torch.load(MODELS / "forecast_lstm.pt", map_location="cpu", weights_only=False)
        self.model = ForecastLSTM(ck["n_features"], ck["n_sensors"], ck["hidden"], ck["layers"])
        self.model.load_state_dict(ck["state"]); self.model.eval()
        c = np.load(MODELS / "calibration.npz")
        self.res_mu, self.res_sigma = c["res_mu"], c["res_sigma"]
        self.threshold = float(c["threshold"])

        proc = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"])
        clim = pd.read_csv(RAW / "climatological.csv") if (RAW / "climatological.csv").exists() else None
        try:
            station = pd.read_csv(RAW / "srinagar_daily.csv", nrows=1)["Station_Name"].iloc[0]
        except Exception:
            station = "SRINAGAR"
        self.bounds = build_bounds(proc.iloc[: self.n_train], clim, station)
        print(f"[detector] bounds: {self.bounds['source']} | threshold {self.threshold:.2f}σ")

    def scan(self, df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df.index, pd.RangeIndex)
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        feats = add_calendar(df)[FEATURE_COLS].to_numpy(np.float32)
        Xz = (feats - self.mu) / self.sigma
        n = len(df)
        vals = df[SENSORS].to_numpy(float)
        months = df["Timestamp"].dt.month.to_numpy()

        reasons_all = [check_row(vals[i], months[i], self.bounds) + check_flatline(vals, i)
                       for i in range(n)]
        scores = np.full(n, np.nan)

        # Layer 2 — batched forecast residuals (skip windows containing NaN)
        wins, widx = [], []
        for i in range(self.lookback, n):
            if np.isfinite(Xz[i - self.lookback: i + 1]).all():
                wins.append(Xz[i - self.lookback: i]); widx.append(i)
        if wins:
            with torch.no_grad():
                pred = self.model(torch.tensor(np.stack(wins))).numpy()
            for k, i in enumerate(widx):
                z = (Xz[i, :N_SENSORS] - pred[k] - self.res_mu) / self.res_sigma
                j = int(np.abs(z).argmax())
                scores[i] = float(np.abs(z).max())
                if scores[i] > self.threshold:
                    reasons_all[i].append(f"{SENSORS[j]} {z[j]:+.1f}σ vs forecast")

        out = df.copy()
        out["score"] = scores
        out["flag"] = [bool(r) for r in reasons_all]
        out["reasons"] = ["; ".join(r) for r in reasons_all]
        return out

def main() -> None:  # Phase 5 'done when': clean-val fire rate matches the quantile
    det = Detector()
    proc = pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"]).reset_index(drop=True)
    seg = proc.iloc[det.n_train - det.lookback: det.n_train + det.n_val].reset_index(drop=True)
    res = det.scan(seg).iloc[det.lookback:]
    fp = res["flag"].mean()
    print(f"clean-validation fire rate: {fp:.3%} (target ≈ 0.5%)")
    for _, r in res[res["flag"]].iterrows():
        print(f"  {r['Timestamp'].date()}  {r['reasons']}")
    append_metrics(f"\n## Phase 5 — detector on clean val\n- fire rate {fp:.3%} (target 0.5%)")

if __name__ == "__main__":
    main()
