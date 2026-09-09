# SIH-Project
weather station defects detection(not the exact problem statement) - we must train an AI with past 1-2 years of weather api data  - check if a given current data is possible( a sudden 20° temp change isn't possible, humidity over 100% isn't possible etc) - a noise injector in python to show working during demo

## Phase 8 — Command Center dashboard (`src/app.py`)

```powershell
.venv\Scripts\python.exe -m streamlit run src/app.py
```

- Opens on the highest-scoring real anomaly (the 5537.7 °C dewpoint day, 16.5σ).
- **⏭ next anomaly** walks real defects/gaps; sidebar **🧪 Noise injector** fires synthetic defects (spike/impossible/stuck/drift/scale/dropout) at any magnitude, then **▶** streams the timeline — pulsing red marker, threshold jump, ticker entry, and an injection report (latency / peak σ / which layer caught it).
- Dark map shows SRINAGAR live + ~420 context stations from `climatological.csv` (click one for climatology details in the sidebar).

## Phase 9a — severity, confidence & corrected-data estimation

Post-processing layer, no retraining (`src/severity.py`, selftest: `python src/severity.py`):

- **Severity** — score bands anchored to the calibrated threshold: `<1.5T LOW` ·
  `<3T MEDIUM` · `<5T HIGH` · else `CRITICAL`; rule-only catches (stuck / missing /
  impossible values, score NaN) show `RULE`. Chips appear in the alert ticker and KPIs.
- **Confidence** — logistic fit through the Phase-6 spike detection-rate curve
  (`sigmoid(a·σ+b)`, fit in `severity.py` from `audit_results.csv`); shown as a
  percentage per alert and in the injection report.
- **Corrected data estimation** — on flagged days the chart adds green diamond
  markers ("corrected (LSTM estimate)"); the injection report shows
  `corrected estimate: X °C (LSTM) vs Y °C reported · Δ`.
