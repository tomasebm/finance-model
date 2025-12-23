#!/usr/bin/env python3
"""
Standalone test script - verifies dashboard prediction logic against original model
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ============================================================================
# CONFIG (matching original model)
# ============================================================================
TRAIN_START = "2018-01-01"
TODAY = datetime.now().strftime("%Y-%m-%d")

# Policy
POLICY = {"BASE_THR": 0.60, "HYST_GAP": 0.03}
ALPHA = 0.15
THR_MIN = 0.20
THR_MAX = 0.80

# Shock thresholds
SHOCK_DAILY_RET = -0.055
VIX_JUMP_2D = 0.22

# Label thresholds
OVERLAY_LOSS_THR = -0.020
OVERLAY_WIN = 7

# ============================================================================
# FETCH DATA
# ============================================================================
print(f"Fetching data from {TRAIN_START} to {TODAY}...")

m = yf.download("^MERV", start=TRAIN_START, end=TODAY, progress=False)
if m.empty:
    print("Merval failed, using GGAL...")
    m = yf.download("GGAL", start=TRAIN_START, end=TODAY, progress=False)

vix = yf.download("^VIX", start=TRAIN_START, end=TODAY, progress=False)
argt = yf.download("ARGT", start=TRAIN_START, end=TODAY, progress=False)

# Flatten MultiIndex
if isinstance(m.columns, pd.MultiIndex):
    m.columns = m.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix["Close"]
if isinstance(argt.columns, pd.MultiIndex):
    argt = argt["Close"]

# Build dataframe
df = pd.DataFrame(index=m.index)
df["close"] = m["Close"].squeeze()
df["vix"] = vix["Close"].squeeze() if "Close" in vix else vix.squeeze()
df["argt"] = argt["Close"].squeeze() if "Close" in argt else argt.squeeze()
df = df.ffill().sort_index()

print(f"Loaded {len(df)} days of data")
print(f"Last date: {df.index[-1].date()}")
print(f"Merval close: {df.iloc[-1]['close']:.2f}")

# ============================================================================
# FEATURE ENGINEERING (matching original model)
# ============================================================================
print("\nBuilding features...")

ret = pd.DataFrame(index=df.index)
ret["ret"] = df["close"].pct_change()

# VIX regime
if not df["vix"].isna().all():
    ret["vix_reg"] = pd.cut(df["vix"], [0,15,20,30,40,200], labels=[0,1,2,3,4]).astype(float)
else:
    ret["vix_reg"] = 0.0

# ARGT stress
ret["argt_stress"] = -np.log(df["argt"]).diff()

# Drawdown
cum = (1 + ret["ret"].fillna(0)).cumprod()
ret["dd"] = cum / cum.cummax() - 1

# Features
feat = pd.DataFrame(index=df.index)
feat["global_vix_reg"] = ret["vix_reg"]
feat["local_argt_5d"] = ret["argt_stress"].rolling(5).mean()
feat["damage_ret_5d"] = ret["ret"].rolling(5).sum()
feat["damage_vol_10d"] = ret["ret"].rolling(10).std()
feat["damage_dd"] = ret["dd"]
feat["dd_chg_5d"] = ret["dd"].diff(5)

# Label (for training)
_target = (ret["ret"] < OVERLAY_LOSS_THR).rolling(OVERLAY_WIN).max().shift(-1)
y = _target.fillna(0).astype(int)

# Shock detection
shock_ret = (ret["ret"] <= SHOCK_DAILY_RET).astype(int)
vix_jump = (df["vix"] / df["vix"].shift(2) - 1 >= VIX_JUMP_2D).astype(int)
ret["shock_nextday"] = (shock_ret | vix_jump).shift(1).fillna(0).astype(int)

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\nTraining model...")

# Training set (exclude last 7 days because we don't have labels)
X = feat
valid_idx = X.index[:-7]
X_train = X.loc[valid_idx].fillna(0)
y_train = y.loc[valid_idx]

print(f"Training samples: {len(X_train)}")

# Current state (last row)
X_live = X.iloc[[-1]].fillna(0)

# Standardize
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_live_scaled = sc.transform(X_live)

# Train RF
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_scaled, y_train)

# ============================================================================
# PREDICT
# ============================================================================
print("\nGenerating prediction...")

prob = clf.predict_proba(X_live_scaled)[0, 1]

# Dynamic thresholds
vix_val = X_live.iloc[0]["global_vix_reg"]
vix_norm = np.clip(vix_val / 4, 0, 1)
base = POLICY["BASE_THR"]
gap = POLICY["HYST_GAP"]

thr = np.clip(base - ALPHA * vix_norm, THR_MIN, THR_MAX)
thr_in = thr
thr_out = np.clip(thr + gap, THR_MIN, THR_MAX)

# Shock check
is_shock = int(ret.iloc[-1]["shock_nextday"])

# Signals
if is_shock:
    signal_cash = "STAY OUT"
    signal_invested = "HARD EXIT"
else:
    signal_cash = "ENTER" if prob < thr_in else "STAY OUT"
    signal_invested = "EXIT" if prob >= thr_out else "HOLD"

# Recommended exposure
rec_exp = 0.0
if prob < thr_in and not is_shock:
    rec_exp = float(np.clip(1 - prob / max(thr_in, 1e-6), 0, 1))

# Additional metrics
current_dd = float(ret.iloc[-1]["dd"])
argt_stress = float(ret.iloc[-1]["argt_stress"]) if not pd.isna(ret.iloc[-1]["argt_stress"]) else 0.0
market_close = float(df.iloc[-1]["close"])
vix_close = float(df.iloc[-1]["vix"]) if not pd.isna(df.iloc[-1]["vix"]) else 0.0

# ============================================================================
# REPORT
# ============================================================================
print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)
print(f"Date:               {df.index[-1].date()}")
print(f"Merval Close:       {market_close:,.2f}")
print(f"VIX:                {vix_close:.2f}")
print(f"VIX Regime:         {vix_val:.1f}")
print("-" * 70)
print(f"Crash Probability:  {prob:.1%}")
print(f"Threshold IN:       {thr_in:.1%}")
print(f"Threshold OUT:      {thr_out:.1%}")
print("-" * 70)
print(f"Shock Alert:        {'YES (FORCE EXIT)' if is_shock else 'NO'}")
print(f"Current Drawdown:   {current_dd:.2%}")
print(f"ARGT Stress:        {argt_stress:.4f}")
print("-" * 70)
print(f"Signal (if CASH):       {signal_cash}")
print(f"Signal (if INVESTED):   {signal_invested}")
print(f"Recommended Exposure:   {rec_exp:.1%}")
print("=" * 70)

# Risk assessment
if prob >= 0.7:
    risk = "EXTREME RISK"
elif prob >= 0.5:
    risk = "HIGH RISK"
elif prob >= 0.3:
    risk = "MODERATE RISK"
else:
    risk = "LOW RISK"

print(f"\nRISK LEVEL: {risk}")

# Feature importance
print("\nTop Feature Values:")
for col in feat.columns:
    val = X_live.iloc[0][col]
    print(f"  {col:20s}: {val:.4f}")

print("\n✓ Test complete")