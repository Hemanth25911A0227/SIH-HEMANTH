"""Phase 4c — Adam(1e-3), MSE, early stopping on val loss, best checkpoint.
Sanity bar: val MAE at or below the persistence baseline (tying is fine —
detection power comes from Phase 5 calibration)."""
from __future__ import annotations
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import PROC, MODELS, FIG, N_SENSORS, LOOKBACK, SENSORS, append_metrics
from dataset import make_windows, split_by_target
from model import ForecastLSTM

def mae_mse(pred, y, sigma):
    e = (pred - y) * sigma[:N_SENSORS]      # back to original units
    return np.abs(e).mean(0), (e ** 2).mean(0)

def main() -> None:
    d = np.load(PROC / "processed.npz", allow_pickle=True)
    X, y, idx = make_windows(d["X"].astype(np.float32), N_SENSORS, LOOKBACK)
    tr, va, te = split_by_target(idx, int(d["n_train"]), int(d["n_val"]))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"windows: train {len(Xtr)}  val {len(Xva)}  test {len(X[te])}")

    p_mae, p_mse = mae_mse(Xva[:, -1, :N_SENSORS], yva, d["sigma"])   # persistence

    torch.manual_seed(0)
    model = ForecastLSTM(int(X.shape[2]), N_SENSORS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    Xtr_t, ytr_t, Xva_t, yva_t = map(torch.tensor, (Xtr, ytr, Xva, yva))

    best, best_state, hist, wait = np.inf, None, [], 0
    for epoch in range(150):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(Xtr_t), 64):
            j = perm[b: b + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[j]), ytr_t[j])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva_t), yva_t).item()
        hist.append((loss.item(), vloss))
        if vloss < best - 1e-6:
            best, wait, best_state = vloss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        elif (wait := wait + 1) >= 15:
            print(f"early stop at epoch {epoch} (best val {best:.4f})")
            break
    model.load_state_dict(best_state)
    torch.save({"state": model.state_dict(), "n_features": int(X.shape[2]),
                "n_sensors": N_SENSORS, "hidden": 64, "layers": 2,
                "lookback": LOOKBACK}, MODELS / "forecast_lstm.pt")

    with torch.no_grad():
        pred = model(Xva_t).numpy()
    l_mae, l_mse = mae_mse(pred, yva, d["sigma"])

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h[:, 0], label="train"); ax.plot(h[:, 1], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "loss_curve.png", dpi=120)

    print(f"\n{'channel':16s}{'LSTM MAE':>10s}{'persistence':>13s}")
    lines = ["\n## Phase 4 — LSTM vs persistence (val, original units)"]
    for c, a, p in zip(SENSORS, l_mae, p_mae):
        print(f"{c:16s}{a:10.2f}{p:13.2f}")
        lines.append(f"- {c}: LSTM MAE {a:.2f} vs persistence {p:.2f}")
    print(f"{'overall MSE':16s}{l_mse.mean():10.2f}{p_mse.mean():13.2f}")
    lines.append(f"- overall MSE: LSTM {l_mse.mean():.2f} vs persistence {p_mse.mean():.2f}")
    verdict = "OK (<= persistence)" if l_mae.mean() <= p_mae.mean() else "WARN: above persistence"
    print(f"\nval MAE check: {verdict}")
    lines.append(f"- verdict: {verdict}")
    append_metrics("\n".join(lines))

if __name__ == "__main__":
    main()
