# metrics.md — every number ever measured goes here

## Conventions
- chronological 70/15/15 split, scaler fit on train only
- z = per-channel forecast residual, calibrated on clean validation
- FP echo exclusion: lookback (14 days) after each injected event

## Phase 1 � data
- 1826 daily rows, 3 sensors, 7 features, splits 1278/273/275
- absent days (real dropouts): 7

## Phase 3 � persistence baseline (val, 1-day horizon)
- Temperature_C: MAE 1.08, MSE 2.07
- DewPoint_C: MAE 41.52, MSE 225382.66
- Pressure_hPa: MAE 1.43, MSE 3.36

## Phase 4 � LSTM vs persistence (val, original units)
- Temperature_C: LSTM MAE 2.31 vs persistence 1.08
- DewPoint_C: LSTM MAE 22.17 vs persistence 41.36
- Pressure_hPa: LSTM MAE 2.32 vs persistence 1.43
- overall MSE: LSTM 37436.97 vs persistence 74854.17
- verdict: OK (<= persistence)

## Phase 5 -- calibration
- threshold 2.76 sigma, clean-val fire rate 0.733%
- 1 sigma forecast error: Temperature_C +/-2.71, DewPoint_C +/-334.51, Pressure_hPa +/-2.66

## Phase 5 � detector on clean val
- fire rate 0.733% (target 0.5%)

## Phase 6 — audit (50 events/cell, random channel)

| defect     |   magnitude |   n |   detection_rate |   median_latency_days |   fp_per_day |
|:-----------|------------:|----:|-----------------:|----------------------:|-------------:|
| spike      |         1   |  50 |             0.52 |                   0   |    0.0149612 |
| spike      |         2   |  50 |             0.96 |                   0   |    0.0151938 |
| spike      |         4   |  50 |             1    |                   0   |    0.0151938 |
| spike      |         8   |  50 |             1    |                   0   |    0.0150388 |
| impossible |         1   |  50 |             1    |                   0   |    0.0155906 |
| dropout    |         1   |  50 |             1    |                   0   |    0.0157937 |
| stuck      |         1   |  50 |             1    |                   3   |    0.015743  |
| drift      |         0.5 |  50 |             0.16 |                  12   |    0.0161633 |
| drift      |         1   |  50 |             0.6  |                   8.5 |    0.0161633 |
| drift      |         2   |  50 |             0.98 |                   7   |    0.0161633 |
| scale      |         0.5 |  50 |             0.46 |                   0   |    0.0163265 |
| scale      |         1   |  50 |             0.8  |                   0   |    0.0156735 |
| scale      |         2   |  50 |             0.94 |                   0   |    0.0159184 |

- clean test FP/day: 0.015
- spike 50% crossing: ~1.0σ (≈8.3 °C)
- honest read: drift/stuck latency is expected; scale on pressure is trivially caught (30% of 850 hPa ≈ 250 hPa)

## Phase 3 — persistence baseline (val, 1-day horizon)
- Temperature_C: MAE 1.08, MSE 2.07
- DewPoint_C: MAE 41.52, MSE 225382.66
- Pressure_hPa: MAE 1.43, MSE 3.36

## Phase 9a — severity tiers + confidence fit
- logistic fit: sigmoid(0.518·x + 1.169) through spike detection-rate curve
- 2.8σ -> LOW, confidence 93.1%
- 4.1σ -> MEDIUM, confidence 96.5%
- 8.3σ -> HIGH, confidence 99.6%
- 16.5σ -> CRITICAL, confidence 100.0%
- 16.5σ -> CRITICAL, confidence 100.0%

## Phase 9a — severity tiers + confidence fit
- logistic fit: sigmoid(0.518·x + 1.169) through spike detection-rate curve
- 2.8σ -> LOW, confidence 93.1%
- 4.1σ -> MEDIUM, confidence 96.5%
- 8.3σ -> HIGH, confidence 99.6%
- 16.5σ -> CRITICAL, confidence 100.0%
- 16.5σ -> CRITICAL, confidence 100.0%
