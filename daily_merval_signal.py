"""
DAILY MERVAL SIGNAL — PRODUCTION SCRIPT
Runs once per day after market close.
NO backtest. NO training loop.
Outputs: signal_strength, exposure_target, shock flag.
Saves to BigQuery.
"""
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from google.cloud import bigquery

# ===============================
# CONFIG
# ===============================
START_DATE = "2011-03-03"
END_DATE = datetime.now().strftime("%Y-%m-%d")
USD_CORE = 0.70
MAX_TACTICAL = 0.30
SHOCK_DAILY_RET = -0.055
VIX_JUMP_2D = 0.22
BASE_THR = 0.60
ALPHA = 0.15
THR_MIN = 0.20
THR_MAX = 0.80

PROJECT_ID = os.environ.get('PROJECT_ID', 'merval-482121')
DATASET = 'merval_signals'
TABLE = 'daily_signal'

# ===============================
# HELPERS
# ===============================
def yf_fetch(ticker):
    df = yf.download(ticker, START_DATE, END_DATE, progress=False)
    if df.empty:
        raise RuntimeError(f"Failed to load {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df

def save_to_bigquery(date, signal_strength, exposure_spy, exposure_ggal, shock_nextday):
    """Save daily signal to BigQuery"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"
        
        rows_to_insert = [{
            "date": date,
            "signal_strength": float(signal_strength),
            "exposure_spy": float(exposure_spy),
            "exposure_ggal": float(exposure_ggal),
            "shock_nextday": int(shock_nextday),
            "created_at": datetime.utcnow().isoformat()
        }]
        
        errors = client.insert_rows_json(table_id, rows_to_insert)
        
        if errors:
            print(f"BigQuery errors: {errors}")
        else:
            print(f"Saved to BigQuery: {date}")
    except Exception as e:
        print(f"BigQuery save failed: {e}")

# ===============================
# DATA
# ===============================
print("Downloading market data...")
merval = yf_fetch("^MERV")
vix = yf_fetch("^VIX")
argt = yf_fetch("ARGT")
ggal_us = yf_fetch("GGAL")
ggal_ba = yf_fetch("GGAL.BA")

idx = merval.index
df = pd.DataFrame(index=idx)
df["merval_ret"] = merval["close"].pct_change()
df["vix"] = vix["close"].reindex(idx).ffill()
df["argt"] = argt["close"].reindex(idx).ffill()
df["usd_ccl"] = (ggal_ba["close"] / ggal_us["close"] * 10).reindex(idx).ffill()
df["usd_ret"] = df["usd_ccl"].pct_change()

# ===============================
# SHOCK DETECTION
# ===============================
shock_ret = (df["merval_ret"] <= SHOCK_DAILY_RET)
vix_jump = (df["vix"] / df["vix"].shift(2) - 1 >= VIX_JUMP_2D)
df["shock_nextday"] = (shock_ret | vix_jump).shift(1).fillna(0).astype(int)

# ===============================
# FEATURES
# ===============================
df["vix_reg"] = pd.cut(df["vix"], [0,15,20,30,40,100], labels=[0,1,2,3,4]).astype(float)
df["argt_stress"] = -np.log(df["argt"]).diff()
df["ret_5d"] = df["merval_ret"].rolling(5).sum()
df["vol_10d"] = df["merval_ret"].rolling(10).std()
df = df.dropna()

FEATURES = df[["vix_reg", "argt_stress", "ret_5d", "vol_10d"]]

# ===============================
# MODEL
# ===============================
print("Training model...")
LABEL = (df["merval_ret"] < -0.02).astype(int)
sc = StandardScaler()
X = sc.fit_transform(FEATURES)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)
clf.fit(X, LABEL)

# ===============================
# TODAY SIGNAL
# ===============================
print("Computing signal...")
latest = FEATURES.iloc[-1:]
latest_scaled = sc.transform(latest)
prob = clf.predict_proba(latest_scaled)[0,1]
vix_norm = np.clip(latest["vix_reg"].iloc[0] / 4, 0, 1)
thr = np.clip(BASE_THR - ALPHA * vix_norm, THR_MIN, THR_MAX)
signal_strength = float(np.clip(1 - prob / max(thr, 1e-6), 0, 1))

tactical = signal_strength * MAX_TACTICAL
total_exposure = USD_CORE + tactical
if df.iloc[-1]["shock_nextday"] == 1:
    total_exposure = USD_CORE

# ===============================
# OUTPUT
# ===============================
output = {
    "date": df.index[-1].strftime("%Y-%m-%d"),
    "signal_strength": round(signal_strength, 3),
    "exposure_spy": round(total_exposure, 2),
    "exposure_ggal": round(1 - total_exposure, 2),
    "shock_nextday": int(df.iloc[-1]["shock_nextday"]),
}

print("DAILY SIGNAL")
print(output)

# ===============================
# SAVE
# ===============================
out_df = pd.DataFrame([output])
out_df.to_csv("daily_signal.csv", mode="a", header=not os.path.exists("daily_signal.csv"), index=False)
print("Saved to CSV")

# SAVE TO BIGQUERY
save_to_bigquery(
    date=output["date"],
    signal_strength=output["signal_strength"],
    exposure_spy=output["exposure_spy"],
    exposure_ggal=output["exposure_ggal"],
    shock_nextday=output["shock_nextday"]
)

print("Script completed!")