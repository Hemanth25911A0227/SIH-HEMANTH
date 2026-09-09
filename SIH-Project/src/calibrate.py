"""Phase 5a — calibration on the CLEAN validation split: per-channel residual
mu/sigma, threshold = 99.5% quantile of max|z| (~1 false flag / 200 days)."""
from __future__ import annotations
import numpy as np
import torch
from preprocess import PROC, MODELS, N_SENSORS, LOOKBACK, SENSORS, append_metrics
from dataset import make_windows, split_by_target
from model import ForecastLSTM

def main() -> None:
    d = np.load(PROC / "processed.npz", allow_pickle=True)
    X, y, idx = make_windows(d["X"].astype(np.float32), N_SENSORS, LOOKBACK)
    _, va, _ = split_by_target(idx, int(d["n_train"]), int(d["n_val"]))
    ck = torch.load(MODELS / "forecast_lstm.pt", map_location="cpu", weights_only=False)
    model = ForecastLSTM(ck["n_features"], ck["n_sensors"], ck["hidden"], ck["layers"])
    model.load_state_dict(ck["state"]); model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X[va])).numpy()
    resid = y[va] - pred
    res_mu = resid.mean(0)
    res_sigma = resid.std(0); res_sigma[res_sigma < 1e-8] = 1.0
    z = (resid - res_mu) / res_sigma
    zmax = np.abs(z).max(1)
    thr = float(np.quantile(zmax, 0.995))
    fp = float((zmax > thr).mean())

    np.savez(MODELS / "calibration.npz", res_mu=res_mu, res_sigma=res_sigma,
             threshold=thr, quantile=0.995, observed_fp_rate=fp)

    errs = res_sigma * d["sigma"][:N_SENSORS]
    print("typical 1-day forecast error (1 sigma):")
    for c, e in zip(SENSORS, errs):
        print(f"  {c:14s} +/-{e:.2f}")
    print(f"\nthreshold = {thr:.2f} sigma | observed fire rate on clean val = {fp:.3%}")
    if fp == 0.0 or fp > 0.05:
        print("  ^ outside 0-5% -- residual stats or threshold are broken, investigate")
    append_metrics(f"\n## Phase 5 -- calibration\n- threshold {thr:.2f} sigma, clean-val fire rate "
                   f"{fp:.3%}\n- 1 sigma forecast error: " + ", ".join(f"{c} +/-{e:.2f}" for c, e in zip(SENSORS, errs)))

if __name__ == "__main__":
    main()
