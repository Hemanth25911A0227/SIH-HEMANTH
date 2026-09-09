# Phase 8 — Command Center dashboard: status & handoff notes

Built and verified in the session of 2026-09-06/07. This file records what was done,
what was fixed, and the one item left to re-verify, so any future session can pick up
without redoing work.

## Files that make up this feature

| File | Purpose |
|---|---|
| `src/app.py` | The dashboard (KPIs, folium map, forecast chart, score timeline, ticker, time-travel playback, noise injector) |
| `.streamlit/config.toml` | Dark theme (`base=dark`, red `#e63946` primary) |
| `requirements.txt` | Added `streamlit`, `streamlit-folium`, `folium`, `plotly` |
| `README.md` | Phase 8 section with the run command |
| `docs/phase8/screenshots/` | Browser-verification screenshots (this folder) |

## How to run

```powershell
# from SIH-Project/
.venv\Scripts\python.exe -m streamlit run src/app.py
```

Installed versions at verification time: streamlit 1.63.0, streamlit-folium 0.27.4,
folium 0.20.0, plotly 5.x.

## What was verified (all passed)

- Headless regression: `streamlit.testing.v1.AppTest` runs the app with zero exceptions.
- Boot state opens on the highest-scoring real anomaly — 2023-01-12, DewPoint
  5537.72 °C → **16.5σ**, map marker pulsing red, 47 events pre-loaded in the ticker
  (screenshot `01_boot_state_2023-01-12_alert.png`).
- Charts render: actual vs LSTM forecast with ±2·res_σ band, red flagged markers,
  full-series score timeline with the dashed 2.76σ threshold line
  (`02_charts_forecast_vs_lstm_timeline.png`).
- Injector flow works live: ⚡ Inject (spike, Temperature_C, 6σ) → auto-rewind 21 days →
  report box: **detected: YES · latency 0 d · peak 16.9σ · caught by L2 · LSTM residual**
  (`04_injector_report_16.9sigma.png`).

## Fixes applied during verification (do not undo)

1. **`width="stretch"`** replaced the deprecated `use_container_width` on all buttons
   and plotly charts — EXCEPT inside `st_folium(...)`, which in 0.27.4 still requires
   `use_container_width=True` (its `width` param is int-only).
2. **Map basemap swapped**: CartoDB dark_matter now stamps tiles with diagonal
   "API KEY REQUIRED" watermarks. `make_map()` now uses Esri World_Dark_Gray_Base plus
   a World_Dark_Gray_Reference (labels) overlay — constants `ESRI_DARK` / `ESRI_ATTR`
   at the top of `app.py`. Verified clean in `03_map_esri_dark_clean.png`.
3. **Injector mode bug**: the radio option string is "🧪 Noise injector" (lowercase i),
   so `"Injector" in mode` never matched and the panel never opened. Fixed to
   `"injector" in mode.lower()`.
4. **Map click resolution**: streamlit-folium returns only `{lat, lng}` for clicks, not
   tooltip/popup names. `render_map()` now resolves the nearest station within 0.5°.

## Data-quality finding (settled — no action needed)

The raw CSV contains exactly ONE corrupted row: 2023-01-12, DewPoint 5537.72 °C.
It lies past the 70% train boundary, so the train split and scaler are clean —
**no pre-scrub or retrain is required**. The row staying in the test region is the
best demo moment (physics rule + LSTM both fire on it).

## ⚠ One item left unverified (do this first next session)

The last edit to `render_report()` in `src/app.py` made layer-crediting **non-exclusive**:
it splits the first alert reason on `";"` and credits every layer that fired, so a +6 °C
spike (which trips both the climatological-range rule and the forecast residual) should
now report `caught by: L1 · physics rules + L2 · LSTM residual`. That edit was made
AFTER the last browser test — re-verify:

1. Run the app (command above), sidebar → 🧪 Noise injector → ⚡ Inject.
2. Expect the INJECTION REPORT to list **L1 + L2 together**; if it still shows only L2,
   debug the part-parsing block in `render_report()`.
3. Re-run the AppTest one-liner:
   `python -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('src/app.py', default_timeout=600); at.run(); assert not at.exception, at.exception"`

## Known cosmetic quirks (from the original runbook, all benign)

- Playback re-renders the map only on status flips, which resets its zoom; Turbo covers
  the full series in ~20–40 s.
- Single live station (the BigQuery export is SRINAGAR-only); gray dots are the ~420
  climatology context stations. Genuine multi-station would need a multi-station export.
- Marker click → sidebar detail can lag one rerun; click again or touch any control.

## Sanity numbers for regression checks

1826 days · clean-scan alerts = 51 total (47 up to day 1473) · threshold 2.76σ ·
lookback 14 d · channel raw σ ≈ Temperature 8.3, DewPoint 6.7, Pressure 4.6 ·
+6σ injected spike peaks ≈ 16–19σ.

## Optional next steps (priority order)

1. Verify the layer-crediting fix above.
2. Preserve map zoom across status-flip re-renders (`st_folium` 0.27.4 accepts
   `zoom=` / `center=`).
3. CSV export button for alerts up to the current playback day.
4. Git commit the feature (files: `src/app.py`, `.streamlit/`, `requirements.txt`,
   `README.md`, `docs/phase8/`).
