"""
DAILY MERVAL SIGNAL — PRODUCTION SCRIPT (FINAL v2.1)
Runs once per day after market close (~18:00 ART).

Fixes:
- Correct BigQuery table_id formatting + error logs
- Robust yfinance parsing (MultiIndex, adj close fallback)
- VIX jump division protection
- Enforce BASE_SAFE_CORE floor in allocation
"""

import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from google.cloud import bigquery

# ===============================
# CONFIG
# ===============================
START_DATE = os.environ.get("START_DATE", "2015-01-01")

# Portfolio: SPY = safe core, GGAL = tactical risk
BASE_SAFE_CORE = float(os.environ.get("BASE_SAFE_CORE", "0.50"))         # Min % in SPY/USD
MAX_RISK_ALLOCATION = float(os.environ.get("MAX_RISK_ALLOCATION", "0.50"))  # Max % in GGAL

# Shock rules
SHOCK_DAILY_RET = float(os.environ.get("SHOCK_DAILY_RET", "-0.055"))
VIX_JUMP_2D = float(os.environ.get("VIX_JUMP_2D", "0.22"))

# Threshold shaping
BASE_THR = float(os.environ.get("BASE_THR", "0.60"))
ALPHA = float(os.environ.get("ALPHA", "0.15"))
THR_MIN = float(os.environ.get("THR_MIN", "0.20"))
THR_MAX = float(os.environ.get("THR_MAX", "0.80"))

# Model params
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "300"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "5"))
MIN_SAMPLES_LEAF = int(os.environ.get("MIN_SAMPLES_LEAF", "30"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

# BigQuery
PROJECT_ID = os.environ.get("PROJECT_ID", "merval-482121")
DATASET = os.environ.get("BQ_DATASET", "merval_signals")
TABLE = os.environ.get("BQ_TABLE", "daily_signal")

# Output file
CSV_PATH = os.environ.get("CSV_PATH", "daily_signal.csv")


# ===============================
# HELPERS
# ===============================
def yf_fetch(ticker: str) -> pd.DataFrame:
    """
    Fetches data from Yahoo Finance with robust MultiIndex handling.
    """
    end_dt = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=START_DATE, end=end_dt, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"Failed to load data for {ticker}")

    # Robust flattening of MultiIndex columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            # fall back to stringifying
            df.columns = [str(c) for c in df.columns]

    # Normalize column names
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Handle 'adj close' vs 'close'
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    if "close" not in df.columns:
        raise RuntimeError(f"{ticker} missing 'close' column. Cols found: {list(df.columns)}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill()
    df = df.dropna(subset=["close"])
    return df


def save_to_bigquery(date: str,
                     signal_strength: float,
                     exp_spy: float,
                     exp_ggal: float,
                     shock_bool: int) -> None:
    """
    Appends one row to BigQuery table.
    """
    client = None
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    try:
        client = bigquery.Client(project=PROJECT_ID)

        rows_to_insert = [{
            "date": date,
            "signal_strength": float(signal_strength),
            "exposure_spy": float(exp_spy),
            "exposure_ggal": float(exp_ggal),
            "shock_nextday": int(shock_bool),
            "created_at": datetime.now(timezone.utc).isoformat()
        }]

        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            print(f"BigQuery insert errors: {errors}")
        else:
            print(f"Saved to BigQuery: {table_id}")

    except Exception:
        print(f"BigQuery save failed for {table_id}")
        traceback.print_exc()


# ===============================
# MAIN
# ===============================
def main() -> None:
    print(f"Starting job at {datetime.now(timezone.utc).isoformat()}")

    # Basic sanity: ensure portfolio constraints are consistent
    if BASE_SAFE_CORE < 0.0 or BASE_SAFE_CORE > 1.0:
        raise ValueError(f"BASE_SAFE_CORE must be in [0,1], got {BASE_SAFE_CORE}")
    if MAX_RISK_ALLOCATION < 0.0 or MAX_RISK_ALLOCATION > 1.0:
        raise ValueError(f"MAX_RISK_ALLOCATION must be in [0,1], got {MAX_RISK_ALLOCATION}")
    if BASE_SAFE_CORE + MAX_RISK_ALLOCATION > 1.0000001:
        raise ValueError(
            f"Invalid constraints: BASE_SAFE_CORE({BASE_SAFE_CORE}) + MAX_RISK_ALLOCATION({MAX_RISK_ALLOCATION}) > 1.0"
        )

    # 1. FETCH DATA
    merval = yf_fetch("^MERV")
    vix = yf_fetch("^VIX")
    argt = yf_fetch("ARGT")

    # 2. ALIGN TO MERVAL CALENDAR
    idx = merval.index
    df = pd.DataFrame(index=idx)
    df["merval_close"] = merval["close"]
    df["merval_ret"] = df["merval_close"].pct_change()

    df["vix"] = vix["close"].reindex(idx).ffill()
    df["argt"] = argt["close"].reindex(idx).ffill()

    # 3. FEATURE ENGINEERING
    # Upper bound 999 to handle extreme VIX spikes without NaNs
    df["vix_reg"] = pd.cut(df["vix"], [0, 15, 20, 30, 40, 999],
                           labels=[0, 1, 2, 3, 4]).astype(float)

    # ARGT stress proxy
    df["argt_stress"] = -np.log(df["argt"]).diff()

    # Momentum + volatility
    df["ret_5d"] = df["merval_ret"].rolling(5).sum()
    df["vol_10d"] = df["merval_ret"].rolling(10).std()

    df = df.dropna()

    if len(df) < 300:
        raise RuntimeError(f"Not enough data: {len(df)} rows")

    feature_cols = ["vix_reg", "argt_stress", "ret_5d", "vol_10d"]

    # 4. SHOCK DETECTION (TODAY)
    latest_ret = float(df["merval_ret"].iloc[-1])
    latest_vix = float(df["vix"].iloc[-1])

    if len(df) >= 3:
        prev2_vix = float(df["vix"].iloc[-3])
    else:
        prev2_vix = float(df["vix"].iloc[0])

    vix_jump = 0.0
    if prev2_vix > 0:
        vix_jump = latest_vix / prev2_vix - 1.0

    is_shock = (latest_ret <= SHOCK_DAILY_RET) or (vix_jump >= VIX_JUMP_2D)

    # 5. TRAIN MODEL (Daily Walk-Forward)
    # Label = tomorrow is a "bad day"
    label = (df["merval_ret"].shift(-1) < -0.02).astype(int)

    train_df = df.iloc[:-1]          # All data except today
    y_train = label.iloc[:-1]
    X_train_raw = train_df[feature_cols]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_raw)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    # 6. PREDICT TOMORROW
    latest_row = df.iloc[-1:]
    latest_X = sc.transform(latest_row[feature_cols])
    prob_crash = float(clf.predict_proba(latest_X)[0, 1])

    # 7. THRESHOLD + SIGNAL
    vix_norm = float(np.clip(float(latest_row["vix_reg"].iloc[0]) / 4.0, 0.0, 1.0))
    thr = float(np.clip(BASE_THR - ALPHA * vix_norm, THR_MIN, THR_MAX))

    # signal in [0,1] where 1 = fully risk-on
    signal_strength = float(np.clip(1.0 - prob_crash / max(thr, 1e-6), 0.0, 1.0))

    # 8. PORTFOLIO ALLOCATION (ENFORCE BASE_SAFE_CORE)
    risk_alloc = signal_strength * MAX_RISK_ALLOCATION

    if is_shock:
        print(f"!!! SHOCK TRIGGERED (Ret: {latest_ret:.2%}, VixJump2D: {vix_jump:.2%}) !!!")
        risk_alloc = 0.0

    exposure_ggal = float(np.clip(risk_alloc, 0.0, MAX_RISK_ALLOCATION))

    # Enforce safe core floor and keep weights summing to 1
    exposure_spy = max(BASE_SAFE_CORE, 1.0 - exposure_ggal)
    exposure_spy = float(np.clip(exposure_spy, 0.0, 1.0))
    exposure_ggal = float(np.clip(1.0 - exposure_spy, 0.0, MAX_RISK_ALLOCATION))

    # 9. OUTPUT & SAVE
    date_str = df.index[-1].strftime("%Y-%m-%d")
    output = {
        "date": date_str,
        "signal_strength": round(signal_strength, 3),
        "exposure_spy": round(exposure_spy, 2),
        "exposure_ggal": round(exposure_ggal, 2),
        "shock_nextday": int(is_shock),
    }

    print("=" * 30)
    print("DAILY SIGNAL")
    print("=" * 30)
    print(output)

    # Save CSV (append)
    out_df = pd.DataFrame([output])
    out_df.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)

    # Save BigQuery
    save_to_bigquery(
        date=output["date"],
        signal_strength=output["signal_strength"],
        exp_spy=output["exposure_spy"],
        exp_ggal=output["exposure_ggal"],
        shock_bool=output["shock_nextday"],
    )

    print("Script completed successfully.")


if __name__ == "__main__":
    main()
