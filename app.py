# =========================================================
# MERVAL DASHBOARD — FIXED & FUTURE-PROOF (2025-12)
# - Market-closed banner (data-driven, no calendars)
# - Fix: BigQuery DATE already returns datetime.date
# - Removes deprecated Streamlit params
# - Handles BigQuery Storage warning cleanly
# =========================================================

import os
import warnings
from datetime import date

import streamlit as st
import pandas as pd
from google.cloud import bigquery

# Silence BigQuery Storage warning explicitly
warnings.filterwarnings(
    "ignore",
    message="BigQuery Storage module not found"
)

# ======================
# CONFIG
# ======================
PROJECT_ID = "merval-482121"
DATASET = "merval_signals"
TABLE = "daily_signal"

PORTFOLIO_FILE = "portfolio_snapshots.csv"

st.set_page_config(page_title="MERVAL Dashboard", layout="centered")
st.title("📊 MERVAL – Dashboard Mensual")

# ======================
# LOAD MODEL SIGNAL
# ======================
@st.cache_data(ttl=3600)
def load_signal():
    client = bigquery.Client(project=PROJECT_ID)

    q = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        ORDER BY created_at DESC
        LIMIT 1
    """

    return client.query(q).to_dataframe(
        create_bqstorage_client=False
    )

sig_df = load_signal()
sig = sig_df.iloc[0]

# ======================
# MARKET CLOSED DETECTION
# ======================
signal_date = sig.date          # already datetime.date
today_date = date.today()
market_closed = signal_date < today_date

# ======================
# LOAD PORTFOLIO HISTORY
# ======================
today = today_date.isoformat()

if os.path.exists(PORTFOLIO_FILE):
    hist = pd.read_csv(PORTFOLIO_FILE)
else:
    hist = pd.DataFrame(
        columns=["date", "spy_pct", "merval_pct", "ars_total"]
    )

# Defaults = last saved snapshot if exists
spy_default, merval_default, ars_default = 50.0, 50.0, 0.0
if not hist.empty:
    last = hist.sort_values("date").iloc[-1]
    spy_default = last.spy_pct * 100
    merval_default = last.merval_pct * 100
    ars_default = last.ars_total

# ======================
# INPUTS
# ======================
st.subheader("🧾 Cartera actual (manual)")

c1, c2, c3 = st.columns(3)
with c1:
    spy_now = st.number_input(
        "SPY (%)", 0.0, 100.0, spy_default
    ) / 100
with c2:
    merval_now = st.number_input(
        "MERVAL (%)", 0.0, 100.0, merval_default
    ) / 100
with c3:
    ars_total = st.number_input(
        "ARS totales", min_value=0.0, value=float(ars_default)
    )

if spy_now + merval_now > 1.0:
    st.error("SPY % + MERVAL % no puede superar 100%")
    st.stop()

# ======================
# SAVE SNAPSHOT (1 PER DAY)
# ======================
if st.button("💾 Guardar cartera de hoy"):
    new_row = pd.DataFrame([{
        "date": today,
        "spy_pct": spy_now,
        "merval_pct": merval_now,
        "ars_total": ars_total
    }])

    hist = hist[hist["date"] != today]
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.sort_values("date")

    hist.to_csv(PORTFOLIO_FILE, index=False)
    st.success("Cartera guardada (una sola versión por día)")

# ======================
# MODEL TARGETS
# ======================
spy_tgt = float(sig.exposure_spy)
merval_tgt = float(sig.exposure_ggal)

delta_spy = spy_tgt - spy_now
delta_merval = merval_tgt - merval_now

# ======================
# SIGNAL DASHBOARD
# ======================
st.divider()

if market_closed:
    st.info(
        f"🛑 **Mercado cerrado** — mostrando última señal válida\n\n"
        f"Último día operado: **{signal_date.strftime('%Y-%m-%d')}**"
    )

st.subheader("📡 Señal del modelo")

a, b, c = st.columns(3)
a.metric("Fecha señal", signal_date.strftime("%Y-%m-%d"))
b.metric("Shock", "SI" if sig.shock_nextday else "NO")
c.metric(
    "Riesgo MERVAL objetivo",
    f"{merval_tgt * 100:.1f}%"
)

# ======================
# TARGET PORTFOLIO
# ======================
st.divider()
st.subheader("🎯 Cartera objetivo")

x, y = st.columns(2)
x.metric("SPY objetivo", f"{spy_tgt * 100:.1f}%")
y.metric("MERVAL objetivo", f"{merval_tgt * 100:.1f}%")

# ======================
# ACTION
# ======================
st.divider()
st.subheader("🧠 Acción sugerida")

def action(delta):
    return "COMPRAR" if delta > 0 else "VENDER"

st.write(
    f"**{action(delta_spy)} SPY {abs(delta_spy * 100):.1f}%**"
)
st.write(
    f"**{action(delta_merval)} MERVAL {abs(delta_merval * 100):.1f}%**"
)

if sig.shock_nextday:
    st.warning("Shock detectado → no aumentar riesgo este mes.")

# ======================
# PORTFOLIO HISTORY
# ======================
st.divider()
st.subheader("📈 Historial de tu cartera")

if hist.empty:
    st.info("Todavía no hay historial guardado.")
else:
    hist_view = hist.copy()
    hist_view["SPY %"] = (hist_view.spy_pct * 100).round(1)
    hist_view["MERVAL %"] = (hist_view.merval_pct * 100).round(1)
    hist_view["Cash %"] = (
        100 - hist_view["SPY %"] - hist_view["MERVAL %"]
    ).round(1)

    st.dataframe(
        hist_view[
            ["date", "SPY %", "MERVAL %", "Cash %", "ars_total"]
        ],
        width="stretch"
    )

    st.subheader("Distribución de cartera (%)")
    st.line_chart(
        hist_view.set_index("date")[
            ["SPY %", "MERVAL %", "Cash %"]
        ]
    )

    st.subheader("Capital total (ARS)")
    st.line_chart(
        hist_view.set_index("date")[["ars_total"]]
    )
