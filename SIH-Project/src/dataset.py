"""Phase 4a — 14-day lookback -> predict the CURRENT day's 3 sensors.
Windows containing any NaN (real gaps) are skipped: train on clean data only."""
import numpy as np

def make_windows(arr: np.ndarray, n_sensors: int, lookback: int = 14):
    """arr: (T, F) scaled features -> X (N, lookback, F), y (N, n_sensors), idx (N,)."""
    ok = np.isfinite(arr).all(axis=1)
    X, y, idx = [], [], []
    for i in range(lookback, len(arr)):
        if not ok[i - lookback: i + 1].all():
            continue
        X.append(arr[i - lookback: i])       # strictly the PAST 14 days
        y.append(arr[i, :n_sensors])         # target = current day
        idx.append(i)
    if not X:
        raise SystemExit("no clean windows — check preprocessing")
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(idx)

def split_by_target(idx, n_train, n_val):
    return (idx < n_train), (idx >= n_train) & (idx < n_train + n_val), (idx >= n_train + n_val)
