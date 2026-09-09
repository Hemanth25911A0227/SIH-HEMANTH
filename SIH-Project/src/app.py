"""
src/app.py — Phase 8: Command Center dashboard.

Run from repo root:
    .venv\\Scripts\\python.exe -m streamlit run src/app.py

Layout
    left sidebar  : demo control (mode, injector), detector info, station details
    main (2/3)    : KPIs, Folium map, forecast-vs-actual chart, score timeline
    right (1/3)   : injection report + live alert ticker
    bottom        : time-travel slider, play, speed, next-anomaly, rewind

Playback is placeholder-based (no rerun per tick) and only re-renders the
map when station status flips, so it stays fast.
"""
from __future__ import annotations

import html as html_mod
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
import folium
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

try:
    from folium.features import DivIcon          # DivIcon moved across folium versions
except ImportError:
    from folium.map import DivIcon
try:
    _Element = folium.Element
except AttributeError:
    from branca.element import Element as _Element

from preprocess import (RAW, PROC, SENSORS, N_SENSORS, FEATURE_COLS, add_calendar)
from detect import Detector
from injector import inject, DEFECT_KINDS
from severity import severity_of, confidence_of

GRACE = 2
MAP_H, CHART_H, SCORE_H = 400, 300, 170
DEFAULT_WINDOW = 120

PULSE_CSS = """
@keyframes cc-pulse {0% {box-shadow:0 0 0 0 rgba(230,57,70,.8);}
 70% {box-shadow:0 0 0 24px rgba(230,57,70,0);} 100% {box-shadow:0 0 0 0 rgba(230,57,70,0);}}
.cc-pulse {animation:cc-pulse 1.2s infinite; border-radius:50%;}
"""

APP_CSS = """
.app-title {font-size:1.55rem; font-weight:800; letter-spacing:.08em; color:#e8edf2;}
.app-sub {color:#8b98a5; font-size:.85rem; margin-bottom:.8rem;}
.status-bar {display:flex; gap:14px; align-items:center; background:#161b22;
  border:1px solid #2d333b; border-radius:10px; padding:8px 14px; margin-bottom:8px;
  font-family:ui-monospace,Menlo,monospace;}
.sb-date {font-weight:700; color:#e8edf2; font-size:1.05rem;}
.sb-day {color:#8b98a5; font-size:.8rem;}
.sb-status {margin-left:auto; padding:3px 10px; border-radius:999px; font-size:.75rem;
  font-weight:700; color:#fff;}
.ticker-box {background:#0d1117; border:1px solid #2d333b; border-radius:10px;
  padding:8px; font-family:ui-monospace,Menlo,monospace; font-size:.72rem;}
.ticker-header {color:#e63946; font-weight:700; letter-spacing:.12em; padding:4px 6px 8px;
  border-bottom:1px solid #2d333b; margin-bottom:6px;}
.ticker-body {max-height:560px; overflow-y:auto;}
.alert-row {padding:6px; border-bottom:1px dotted #21262d; color:#c9d1d9; line-height:1.45;}
.a-date {color:#8b98a5;} .a-sig {color:#ffd166; font-weight:700;} .a-msg {color:#e63946;}
.a-tag {color:#fff; background:#e63946; border-radius:4px; padding:0 4px; font-size:.65rem; margin-left:4px;}
.sev-chip {border-radius:4px; padding:1px 6px; font-size:.62rem; font-weight:800;
  margin-left:5px; letter-spacing:.05em; white-space:nowrap;}
.a-conf {color:#8ecae6; font-weight:700; margin-left:5px; white-space:nowrap;}
.inj-box {background:#1c2333; border:1px solid #e63946; border-radius:10px;
  padding:10px 12px; font-size:.8rem; margin-bottom:10px; line-height:1.7;}
.inj-box h4 {margin:0 0 6px; color:#ffd166; font-size:.85rem; letter-spacing:.06em;}
.inj-box .big {color:#e63946; font-size:1.5rem; font-weight:800;}
div[data-testid="stMetric"] {background:#161b22; border:1px solid #2d333b;
  border-radius:10px; padding:10px 14px;}
"""


# ----------------------------------------------------------------------------- backend
def forecast_series(det: Detector, df: pd.DataFrame):
    """Model forecast for every day (raw units); NaN where the window is unclean."""
    feats = add_calendar(df)[FEATURE_COLS].to_numpy(np.float32)
    xz = (feats - det.mu) / det.sigma
    pred = np.full((len(df), N_SENSORS), np.nan)
    wins, widx = [], []
    for i in range(det.lookback, len(df)):
        if np.isfinite(xz[i - det.lookback: i + 1]).all():
            wins.append(xz[i - det.lookback: i])
            widx.append(i)
    if wins:
        with torch.no_grad():
            p = det.model(torch.tensor(np.stack(wins))).numpy()
        pred[np.asarray(widx)] = p * det.sigma[:N_SENSORS] + det.mu[:N_SENSORS]
    return pred, np.asarray(widx)


def station_meta() -> dict:
    p = RAW / "srinagar_daily.csv"
    if p.exists():
        r = pd.read_csv(p, nrows=1).iloc[0]
        return dict(name=str(r["Station_Name"]), sid=int(r["Station_ID"]),
                    lat=float(r["Latitude"]), lon=float(r["Longitude"]),
                    elev=float(r["Elevation_m"]))
    return dict(name="SRINAGAR", sid=420270, lat=34.083, lon=74.833, elev=1587.0)


@st.cache_resource
def load_clim() -> pd.DataFrame:
    p = RAW / "climatological.csv"
    if not p.exists():
        return pd.DataFrame()
    c = pd.read_csv(p).dropna(subset=["latitude", "longitude"])
    if not len(c):
        return pd.DataFrame()
    return (c.groupby("station_name", as_index=False)
             .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                  altitude=("altitude", "mean"), state=("state", "first"),
                  district=("district", "first"),
                  station_number=("station_number", "first"),
                  mean_wind_kmph=("mean_wind_speed_in_kmph", "mean")))


@st.cache_resource(show_spinner="Booting detector — loading LSTM, scanning the full series…")
def load_backend():
    det = Detector()
    proc = (pd.read_csv(PROC / "processed.csv", parse_dates=["Timestamp"])
            .reset_index(drop=True))
    scan = det.scan(proc)
    pred, _ = forecast_series(det, proc)
    raw = np.load(PROC / "processed.npz", allow_pickle=True)["raw"]
    sig = dict(zip(SENSORS, np.nanstd(raw[: det.n_train], axis=0)))
    return det, proc, scan, pred, station_meta(), load_clim(), sig


def build_alerts(scan: pd.DataFrame, dates: pd.Series, threshold: float) -> list[dict]:
    fl = scan["flag"].to_numpy()
    sc = scan["score"].to_numpy(float)
    rs = scan["reasons"].fillna("").astype(str).tolist()
    out = []
    for i in np.flatnonzero(fl):
        lab, css = severity_of(sc[i], threshold)
        out.append(dict(day=int(i), date=dates.iloc[i], score=float(sc[i]),
                        reasons=rs[i], tag="", sev=lab, sev_css=css,
                        conf=confidence_of(sc[i])))
    return out


def alert_row_html(a: dict) -> str:
    sig = f"{a['score']:.1f}σ" if np.isfinite(a["score"]) else "rule"
    d = pd.Timestamp(a["date"]).strftime("%Y-%m-%d")
    tag = " <span class='a-tag'>INJECTED</span>" if a.get("tag") == "inj" else ""
    msg = html_mod.escape(a["reasons"]) or "flagged"
    chip = (f"<span class='sev-chip' style='{a['sev_css']}'>{a['sev']}</span>"
            if a.get("sev") else "")
    conf = (f"<span class='a-conf'>{a['conf'] * 100:.0f}%</span>"
            if a.get("conf") is not None else "")
    return (f"<div class='alert-row'><span class='a-date'>[{d}]</span> <b>SRINAGAR</b> "
            f"<span class='a-sig'>{sig}</span>{chip}{conf} · "
            f"<span class='a-msg'>{msg}</span>{tag}</div>")


ESRI_DARK = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = "Esri, HERE, Garmin, © OpenStreetMap contributors"


def make_map(meta: dict, clim: pd.DataFrame, anomaly: bool):
    # Esri dark canvas: keyless (CartoDB dark_matter now watermarks tiles)
    m = folium.Map(location=[meta["lat"], meta["lon"]], zoom_start=5,
                   tiles=ESRI_DARK, attr=ESRI_ATTR, control_scale=True)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
               "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"),
        attr=ESRI_ATTR, name="labels").add_to(m)
    m.get_root().html.add_child(_Element(f"<style>{PULSE_CSS}</style>"))
    active = meta["name"].upper().strip()
    if len(clim):                                   # context stations from climatology
        for r in clim.itertuples():
            if str(r.station_name).upper().strip() == active:
                continue
            folium.CircleMarker([r.latitude, r.longitude], radius=3, color="#586274",
                                fill=True, fill_color="#586274", fill_opacity=0.55,
                                tooltip=str(r.station_name)).add_to(m)
    if anomaly:
        dot = ("<div class='cc-pulse' style='width:26px;height:26px;background:#e63946;"
               "border:3px solid #ffd166;'></div>")
    else:
        dot = ("<div style='width:16px;height:16px;background:#2a9d8f;border:2px solid "
               "#b7e4c7;border-radius:50%;box-shadow:0 0 8px rgba(42,157,143,.9);'></div>")
    folium.Marker([meta["lat"], meta["lon"]], tooltip=meta["name"], popup=meta["name"],
                  icon=DivIcon(html=dot, icon_size=(26, 26),
                               icon_anchor=(13, 13))).add_to(m)
    return m


# ----------------------------------------------------------------------------- app
def main() -> None:
    st.set_page_config(page_title="Weather Defect Command Center", page_icon="🛰",
                       layout="wide")
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

    det, proc, scan, pred, meta, clim, sig = load_backend()
    dates = proc["Timestamp"]
    n = len(proc)
    base_alerts = build_alerts(scan, dates, det.threshold)

    # open on the highest-scoring REAL anomaly (the 5537.7°C dewpoint day)
    if "day" not in st.session_state:
        cand = [a for a in base_alerts if np.isfinite(a["score"])]
        d0 = max(cand, key=lambda a: a["score"])["day"] if cand else det.lookback + 365
        st.session_state.day = int(np.clip(d0, det.lookback, n - 1))
    day = int(np.clip(st.session_state.day, det.lookback, n - 1))

    # ---------------- left sidebar: demo control panel
    with st.sidebar:
        st.markdown("## 🎛 Demo control")
        mode = st.radio("Mode", ["📡 Historical scan", "🧪 Noise injector"])
        injector_mode = "injector" in mode.lower()

        if injector_mode:
            st.markdown("#### 🧪 Noise injector")
            kind = st.selectbox("Defect type", DEFECT_KINDS, index=0)
            chan_inj = st.selectbox("Channel", SENSORS, index=0)
            mag = st.slider("Magnitude (σ of channel)", 0.5, 8.0, 6.0, 0.5)
            length = 1 if kind == "spike" else st.slider("Event length (days)", 2, 30, 10)
            st.caption(f"Event starts at the current playback date — "
                       f"**{dates.iloc[day].date()}** (auto-rewinds 21 d so you can "
                       f"watch it approach).")
            b1_, b2_ = st.columns(2)
            fire = b1_.button("⚡ Inject", type="primary", width="stretch")
            reset = b2_.button("♻ Reset", width="stretch",
                               disabled="injection" not in st.session_state)
            if reset:
                st.session_state.pop("injection", None)
                st.rerun()
            if fire:
                start = int(np.clip(day, det.lookback + 1,
                                    n - length - det.lookback - GRACE - 1))
                cdf, labels = inject(proc, kind, start, length, chan_inj,
                                     float(mag), sig)
                cscan = det.scan(cdf)
                cpred, _ = forecast_series(det, cdf)
                st.session_state.injection = dict(
                    df=cdf, scan=cscan, pred=cpred,
                    alerts=build_alerts(cscan, dates, det.threshold),
                    labels=labels, start=start, length=length, kind=kind,
                    col=chan_inj, mag=float(mag))
                st.session_state.day = max(det.lookback, start - 21)
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🩺 Detector")
        st.markdown(
            f"- station: **{meta['name']}** · id {meta['sid']} · {meta['elev']:.0f} m\n"
            f"- lookback **{det.lookback} d** → 1 d forecast\n"
            f"- threshold **{det.threshold:.2f}σ** (99.5% quantile, clean val)\n"
            f"- pressure bounds: {det.bounds['source']}\n"
            f"- channels: {', '.join(SENSORS)}")

    # ---------------- active view (clean or injected)
    inj = st.session_state.get("injection") if injector_mode else None
    if inj is not None:
        vdf, vscan, vpred, valerts = inj["df"], inj["scan"], inj["pred"], inj["alerts"]
        for a in valerts:      # tag injected-window alerts for the ticker
            if inj["start"] <= a["day"] < inj["start"] + inj["length"] + GRACE:
                a["tag"] = "inj"
    else:
        vdf, vscan, vpred, valerts = proc, scan, pred, base_alerts
    vflags = vscan["flag"].to_numpy()
    vscores = vscan["score"].to_numpy(float)

    # ---------------- header + KPI placeholders
    st.markdown("<div class='app-title'>🛰 WEATHER STATION DEFECT COMMAND CENTER</div>"
                "<div class='app-sub'>3-layer defect detector — LSTM forecast residual · "
                "physics rules · flatline — station SRINAGAR</div>",
                unsafe_allow_html=True)
    kph = [c.empty() for c in st.columns(4)]

    mcol, tcol = st.columns([0.66, 0.34], gap="medium")
    with mcol:
        badge_ph = st.empty()
        map_ph = st.empty()
        st.markdown("#### 📈 Sensor readings vs LSTM forecast")
        ctl = st.columns(2)
        chan = ctl[0].selectbox("channel", SENSORS, index=0,
                                label_visibility="collapsed")
        win_lbl = ctl[1].selectbox("window", ["60 days", "120 days", "1 year"], index=1,
                                   label_visibility="collapsed")
        WIN = {"60 days": 60, "120 days": 120, "1 year": 365}[win_lbl]
        chart_ph = st.empty()
        st.markdown("#### 📉 Anomaly score timeline (full series)")
        score_ph = st.empty()
    with tcol:
        report_ph = st.empty()
        ticker_ph = st.empty()

    # ---------------- bottom: time travel + playback
    st.markdown("---")
    cA, cB, cC, cD, cE = st.columns([0.50, 0.10, 0.17, 0.12, 0.11])
    with cA:
        cur = st.slider("⏱ Time travel", min_value=det.lookback,
                        max_value=n - 1, value=day)
        if cur != st.session_state.day:
            st.session_state.day = int(cur)
    play = cB.button("▶", width="stretch",
                     help="Stream the series from the current date")
    speed_lbl = cC.selectbox("speed", ["Slow · 1 d/tick", "Fast · 7 d/tick",
                                       "Turbo · 28 d/tick"], index=2,
                             label_visibility="collapsed")
    step = int(speed_lbl.split("·")[1].split()[0])
    nxt = cD.button("⏭ next anomaly", width="stretch")
    rwd = cE.button("⏮ rewind", width="stretch")
    if nxt:
        ups = [a["day"] for a in valerts if a["day"] > st.session_state.day]
        if ups:
            st.session_state.day = min(ups)
        st.rerun()
    if rwd:
        st.session_state.day = det.lookback
        st.rerun()

    # ---------------- render helpers
    def status_at(i: int, span: int = 1) -> str:
        return "anomaly" if vflags[i: min(i + span, n)].any() else "healthy"

    def render_kpis(i: int) -> None:
        detected = sum(a["day"] <= i for a in valerts)
        s = vscores[i]
        delta = None
        if vflags[i]:
            lab, _ = severity_of(s, det.threshold)
            delta = f"{lab} · {confidence_of(s):.0%} confidence"
        kph[0].metric("Days monitored", f"{i + 1:,}")
        kph[1].metric("Events logged", f"{detected:,}")
        kph[2].metric("Live score", f"{s:.1f} σ" if np.isfinite(s) else "—",
                      delta=delta, delta_color="off")
        kph[3].metric("Station", "🚨 ALERT" if status_at(i) == "anomaly" else "🟢 NOMINAL")

    def render_badge(i: int, span: int = 1) -> None:
        stt = status_at(i, span)
        color = "#e63946" if stt == "anomaly" else "#2a9d8f"
        icon = "🚨 SENSOR ALERT" if stt == "anomaly" else "🟢 NOMINAL"
        badge_ph.markdown(
            f"<div class='status-bar'><span class='sb-date'>📅 "
            f"{dates.iloc[i]:%Y-%m-%d}</span><span class='sb-day'>DAY {i + 1:,} / {n:,}"
            f"</span><span class='sb-status' style='background:{color}'>{icon}"
            f"</span></div>", unsafe_allow_html=True)

    def render_map(i: int, span: int = 1) -> None:
        with map_ph:
            ret = st_folium(make_map(meta, clim, status_at(i, span) == "anomaly"),
                            height=MAP_H, use_container_width=True,
                            returned_objects=["last_object_clicked"])
        # st_folium returns only {lat, lng} for clicks -> resolve nearest station
        loc = (ret or {}).get("last_object_clicked")
        if isinstance(loc, dict) and "lat" in loc:
            cands = [(meta["name"], meta["lat"], meta["lon"])]
            if len(clim):
                cands += [(str(r.station_name), float(r.latitude), float(r.longitude))
                          for r in clim.itertuples()]
            best, bd = None, 1e9
            for nm, la, lo in cands:
                d = abs(la - loc["lat"]) + abs(lo - loc.get("lng", 0.0))
                if d < bd:
                    bd, best = d, nm
            if best is not None and bd < 0.5:
                st.session_state.selected_station = best

    def main_fig(i: int) -> go.Figure:
        ci = SENSORS.index(chan)
        lo, hi = max(det.lookback, i - WIN), min(n - 1, i + max(10, WIN // 4))
        x = dates.iloc[lo: hi + 1]
        y = vdf[chan].iloc[lo: hi + 1].to_numpy(float)
        f = vpred[lo: hi + 1, ci]
        band = 2.0 * float(det.res_sigma[ci] * det.sigma[ci])
        fl = vflags[lo: hi + 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=f + band, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=f - band, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(255,183,3,0.10)", hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=f, name="LSTM forecast",
                                 line=dict(color="#ffb703", width=1.4, dash="dot")))
        fig.add_trace(go.Scatter(x=x, y=y, name="actual",
                                 line=dict(color="#8ecae6", width=1.8)))
        m = fl & np.isfinite(y)
        if m.any():
            fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", name="flagged",
                                     marker=dict(color="#e63946", size=9,
                                                 line=dict(color="white", width=1))))
        mc = m & np.isfinite(f)
        if mc.any():   # corrected data estimation: LSTM forecast as the true value
            fig.add_trace(go.Scatter(x=x[mc], y=f[mc], mode="markers",
                                     name="corrected (LSTM estimate)",
                                     marker=dict(color="#06d6a0", size=10,
                                                 symbol="diamond",
                                                 line=dict(color="#04352a", width=1))))
        if inj is not None:
            fig.add_vrect(x0=dates.iloc[inj["start"]],
                          x1=dates.iloc[min(inj["start"] + inj["length"] - 1, n - 1)],
                          fillcolor="rgba(230,57,70,0.15)", line_width=0)
        fig.add_vline(x=dates.iloc[i], line_color="#e9c46a", line_width=1)
        fig.update_layout(template="plotly_dark", height=CHART_H,
                          margin=dict(l=10, r=10, t=8, b=10), yaxis_title=chan,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        return fig

    def score_fig(i: int) -> go.Figure:
        lo = det.lookback
        x, s, fl = dates.iloc[lo:], vscores[lo:], vflags[lo:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=s, name="max |z|",
                                 line=dict(color="#c9d1d9", width=1.2)))
        fig.add_hline(y=det.threshold, line_color="#e63946", line_dash="dash",
                      annotation_text=f"threshold {det.threshold:.1f}σ",
                      annotation_position="top left")
        m = fl & np.isfinite(s)
        if m.any():
            fig.add_trace(go.Scatter(x=x[m], y=s[m], mode="markers", name="flag",
                                     marker=dict(color="#e63946", size=5)))
        fig.add_vline(x=dates.iloc[i], line_color="#e9c46a", line_width=1)
        fig.update_layout(template="plotly_dark", height=SCORE_H, yaxis_title="σ",
                          showlegend=False, margin=dict(l=10, r=10, t=8, b=10))
        return fig

    def render_ticker(i: int) -> None:
        shown = [a for a in valerts if a["day"] <= i]
        body = "".join(alert_row_html(a) for a in shown[-80:][::-1]) or \
               "<div class='alert-row'>no anomalies logged yet…</div>"
        ticker_ph.markdown(
            f"<div class='ticker-box'><div class='ticker-header'>⚠ LIVE ALERT FEED — "
            f"{len(shown)} event{'s' if len(shown) != 1 else ''} logged</div>"
            f"<div class='ticker-body'>{body}</div></div>", unsafe_allow_html=True)

    def render_report() -> None:
        if inj is None:
            report_ph.empty()
            return
        start, stop = inj["start"], inj["start"] + inj["length"]
        win = vflags[start: stop + GRACE]
        caught = bool(win.any())
        if caught:
            k = int(np.flatnonzero(win)[0])
            first = str(vscan["reasons"].iloc[start + k])
            swin = vscores[start: stop + GRACE]
            has_sc = bool(np.isfinite(swin).any())
            peak = float(np.nanmax(swin)) if has_sc else float("nan")
            peak_txt = f"{peak:.1f}σ" if has_sc else "rule"
            sev_lab, sev_css = severity_of(peak, det.threshold)
            parts = [p.strip().lower() for p in first.split(";")]
            layers = []
            if any("vs forecast" not in p and "stuck" not in p for p in parts):
                layers.append("L1 · physics rules")
            if any("vs forecast" in p for p in parts):
                layers.append("L2 · LSTM residual")
            if any("stuck" in p for p in parts):
                layers.append("L3 · flatline")
            # corrected data estimation at the worst day of the event
            ci = SENSORS.index(inj["col"])
            widx = np.arange(start, min(stop + GRACE, n))
            wday = int(widx[int(np.nanargmax(swin))]) if has_sc else start + k
            rep = float(vdf[inj["col"]].iloc[wday])
            fc = float(vpred[wday, ci])
            unit = " °C" if inj["col"].endswith("_C") else " hPa"
            corr = (f"corrected estimate: <b style='color:#06d6a0'>{fc:.1f}{unit}</b> "
                    f"(LSTM) vs <b>{rep:.1f}{unit}</b> reported · Δ {abs(rep - fc):.1f}"
                    if np.isfinite(fc) else
                    "corrected estimate: unavailable — forecast window spoiled "
                    "by the defect")
            body = (f"detected: <b style='color:#2a9d8f'>YES</b> · latency "
                    f"<b>{k} d</b> · peak <span class='big'>{peak_txt}</span> "
                    f"<span class='sev-chip' style='{sev_css}'>{sev_lab}</span> · "
                    f"confidence <b>{confidence_of(peak):.0%}</b><br>"
                    f"caught by: <b>{' + '.join(layers)}</b><br>"
                    f"reason: {html_mod.escape(first)}<br>{corr}")
        else:
            body = ("detected: <b style='color:#e63946'>NO</b> — the detector missed "
                    "this one. Raise magnitude/length and re-inject.")
        report_ph.markdown(
            f"<div class='inj-box'><h4>⚡ INJECTION REPORT</h4>"
            f"event: <b>{inj['kind']}</b> on {inj['col']} ({inj['mag']:g}σ · "
            f"{inj['length']} d) from <b>{dates.iloc[inj['start']].date()}</b><br>"
            f"{body}</div>", unsafe_allow_html=True)

    # ---------------- initial render
    i = int(np.clip(st.session_state.day, det.lookback, n - 1))
    st.session_state.day = i
    render_kpis(i); render_badge(i); render_map(i)
    render_report(); render_ticker(i)
    chart_ph.plotly_chart(main_fig(i), width="stretch",
                          config={"displayModeBar": False})
    score_ph.plotly_chart(score_fig(i), width="stretch",
                          config={"displayModeBar": False})

    # ---------------- playback (placeholder-based; map only re-renders on flips)
    if play:
        prog = st.empty()
        last_status = status_at(i, step)
        try:
            while i < n:
                render_badge(i, step); render_kpis(i); render_ticker(i)
                chart_ph.plotly_chart(main_fig(i), width="stretch",
                                      config={"displayModeBar": False})
                score_ph.plotly_chart(score_fig(i), width="stretch",
                                      config={"displayModeBar": False})
                cur_status = status_at(i, step)
                if cur_status != last_status:
                    render_map(i, step)
                    last_status = cur_status
                prog.progress(min(1.0, (i - det.lookback) / (n - 1 - det.lookback)),
                              text=f"▶ streaming · {dates.iloc[i].date()}")
                time.sleep(0.06)
                i += step
        finally:
            st.session_state.day = int(min(i, n - 1))
        st.rerun()

    # ---------------- left sidebar (block 2): selected station details
    sel = st.session_state.get("selected_station", meta["name"])
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"#### 📍 Station: {sel}")
        if sel.strip().upper() == meta["name"].upper():
            st.markdown(f"- id **{meta['sid']}** · {meta['lat']:.3f}°N, "
                        f"{meta['lon']:.3f}°E · {meta['elev']:.0f} m asl")
            row = (clim[clim["station_name"].astype(str).str.upper().str.strip()
                        == sel.upper()] if len(clim) else pd.DataFrame())
            if len(row):
                r = row.iloc[0]
                st.markdown(f"- {r.get('state', '—')} · {r.get('district', '—')}")
            j = int(np.clip(st.session_state.day, det.lookback, n - 1))
            lo = max(det.lookback, j - 90)
            fig = go.Figure(go.Scatter(x=dates.iloc[lo: j + 1],
                                       y=vdf["Temperature_C"].iloc[lo: j + 1],
                                       line=dict(color="#8ecae6", width=1.6)))
            fig.update_layout(template="plotly_dark", height=170, showlegend=False,
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
            st.caption(f"Temperature_C · last {j - lo + 1} days ending "
                       f"{dates.iloc[j].date()}")
        else:
            row = (clim[clim["station_name"].astype(str) == sel]
                   if len(clim) else pd.DataFrame())
            if len(row):
                r = row.iloc[0]
                alt = r.get("altitude")
                st.markdown(f"- state: **{r.get('state', '—')}**"
                            + (f" · altitude {alt:.0f} m" if pd.notna(alt) else ""))
                mw = r.get("mean_wind_kmph")
                if pd.notna(mw):
                    st.markdown(f"- mean wind speed: {mw:.1f} km/h")
            st.info("Context station from climatological.csv — no daily time-series "
                    "in this dataset.")


if __name__ == "__main__":
    main()
