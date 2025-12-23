# ============================================================================
# MERVAL TAIL-RISK SYSTEM — EXECUTION REALITY FIX
# FIXES:
# 1) Open-to-Open Execution (Reduces lag from 24h to overnight)
# 2) USD Risk-Off Leg (Cash != 0, Cash = USD)
# 3) Robust Open Data Handling
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIG
# ============================================================================
START_DATE = "2015-01-01"
END_DATE   = "2025-12-23"

FFILL_LIMIT      = 5
MIN_TRAIN_DAYS   = 252 * 2
RETRAIN_FREQ     = "QE"
TRANSACTION_COST = 0.0015  # Slightly higher to account for bid/ask spread at Open

# Shock override
SHOCK_DAILY_RET = -0.055
VIX_JUMP_2D     = 0.22

# Labels
OVERLAY_LOSS_THR = -0.020
OVERLAY_WIN      = 7

CRISIS_DD_THR    = -0.12
CRISIS_MIN_DAYS  = 10
CRISIS_LOOKAHEAD = 15

# Crisis exposure cap
CRISIS_REGIME_X     = 0.60
CRISIS_MAX_EXPOSURE = 0.40

# Policy params
POLICY = {
    "overlay": {"BASE_THR": 0.60, "HYST_GAP": 0.03, "MIN_HOLD_DAYS": 2},
    "crisis":  {"BASE_THR": 0.45, "HYST_GAP": 0.08, "MIN_HOLD_DAYS": 8},
}

ALPHA   = 0.15
THR_MIN = 0.20
THR_MAX = 0.80

# ============================================================================
# HELPERS
# ============================================================================
def yf_fetch(ticker):
    """Fetches Open and Close to handle execution logic."""
    df = yf.download(ticker, START_DATE, END_DATE, progress=False)
    if df is None or df.empty:
        return None
    
    # Flatten MultiIndex if exists
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    # Clean column names
    df.columns = [c.lower() for c in df.columns]
    
    # Data cleaning for "Open" (sometimes Yahoo has 0 or NaN for EM opens)
    if 'open' in df.columns and 'close' in df.columns:
        df['open'] = df['open'].replace(0, np.nan).fillna(df['close'])
        
    return df

def align(s, idx):
    return s.reindex(idx).ffill(limit=FFILL_LIMIT)

def max_dd(r):
    eq = (1 + r.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()

def perf_metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return 0,0,0,0
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sh = ann_ret / ann_vol if ann_vol > 0 else 0
    return ann_ret, ann_vol, sh, max_dd(r)

# ============================================================================
# DATA LOADING
# ============================================================================
# 1. Main Asset (Merval)
m_df = yf_fetch("^MERV")
if m_df is None or m_df.empty:
    m_df = yf_fetch("GGAL") # Fallback

# Ensure we have a master index
m_df = m_df.sort_index()
MASTER = m_df.index
levels = pd.DataFrame(index=MASTER)
levels["merval_close"] = m_df["close"]
levels["merval_open"]  = m_df["open"]

# 2. Auxiliaries
for t, c in {"^VIX": "vix", "ARGT": "argt", "ARS=X": "usd_ars"}.items():
    aux = yf_fetch(t)
    if aux is not None:
        levels[c] = align(aux["close"], MASTER)
        if c == "usd_ars":
            # We also need Open for USD if we want to be precise, 
            # but Close-to-Close is standard for FX hedging. 
            # We will use close-to-close for the risk-off leg to keep it simple.
            pass

# ============================================================================
# RETURNS COMPUTATION
# ============================================================================
ret = pd.DataFrame(index=MASTER)

# A) Feature Generation Returns (Close-to-Close) -> Used for SIGNALS
ret["merval_ret_cc"] = levels["merval_close"].pct_change()

# B) Execution Returns (Open-to-Open) -> Used for STRATEGY P&L
# If we decide at Close(t), we enter at Open(t+1).
# We hold until Open(t+2). The return experienced is Open(t+2)/Open(t+1) - 1.
# This return belongs to decision time 't'.
ret["merval_ret_oo_next"] = (levels["merval_open"].shift(-2) / levels["merval_open"].shift(-1) - 1)

# C) Risk Off Return (USD)
# Assuming we sit in USD/ARS when not in Merval.
# Standard execution for FX is usually T+1 or T+0 spot, but let's assume
# we get the Close-to-Close return of the USD.
# We shift it -1 to match the implementation lag (Decision T applies to T+1 returns).
ret["usd_ret_next"] = levels["usd_ars"].pct_change().shift(-1)

# Feature Construction continues on Close-to-Close basis
if "vix" in levels:
    ret["vix_reg"] = pd.cut(levels["vix"], [0,15,20,30,40,100], labels=[0,1,2,3,4]).astype(float)
else:
    ret["vix_reg"] = 0.0

ret["argt_stress"] = -np.log(levels["argt"]).diff() if "argt" in levels else 0.0
cum = (1 + ret["merval_ret_cc"].fillna(0)).cumprod()
ret["dd"] = cum / cum.cummax() - 1

ret["regime_score"] = (
    0.5 * (ret["vix_reg"] / 4).fillna(0) +
    0.5 * ret["argt_stress"].rolling(5).mean().clip(0,0.05).fillna(0) / 0.05
)

# ============================================================================
# LABELS & FEATURES (UNCHANGED LOGIC)
# ============================================================================
_overlay_raw = (ret["merval_ret_cc"] < OVERLAY_LOSS_THR).rolling(OVERLAY_WIN).max().shift(-1)
ret["tail_overlay"] = _overlay_raw.fillna(0).astype(int)

deep_dd = (ret["dd"] <= CRISIS_DD_THR).astype(int)
deep_dd_persistent = (deep_dd.rolling(CRISIS_MIN_DAYS).sum() >= CRISIS_MIN_DAYS).astype(int)
_crisis_raw = deep_dd_persistent.rolling(CRISIS_LOOKAHEAD).max().shift(-1)
ret["tail_crisis"] = _crisis_raw.fillna(0).astype(int)

# Shock override
shock_ret = (ret["merval_ret_cc"] <= SHOCK_DAILY_RET).astype(int)
if "vix" in levels:
    vix_jump = (levels["vix"] / levels["vix"].shift(2) - 1 >= VIX_JUMP_2D).astype(int)
else:
    vix_jump = 0
ret["shock_nextday"] = (shock_ret | vix_jump).shift(1).fillna(0).astype(int)

# Feature Set
feat = pd.DataFrame(index=MASTER)
feat["global_vix_reg"] = ret["vix_reg"]
feat["local_argt_5d"]  = ret["argt_stress"].rolling(5).mean()
feat["damage_ret_5d"]  = ret["merval_ret_cc"].rolling(5).sum()
feat["damage_vol_10d"] = ret["merval_ret_cc"].rolling(10).std()
feat["damage_dd"]      = ret["dd"]
feat["dd_chg_5d"]      = ret["dd"].diff(5)

# Join and dropna. Note: this drops rows where we don't have future open returns too.
data = pd.concat([ret, feat], axis=1).dropna()

FEATURES = data[[
    "global_vix_reg","local_argt_5d","damage_ret_5d",
    "damage_vol_10d","damage_dd","dd_chg_5d"
]]

# ============================================================================
# WALK-FORWARD (OPEN-TO-OPEN EXECUTION)
# ============================================================================
def walk_forward(label, policy_name):
    pset = POLICY[policy_name]
    BASE_THR = pset["BASE_THR"]
    HYST_GAP = pset["HYST_GAP"]
    MIN_HOLD_DAYS = pset["MIN_HOLD_DAYS"]

    y = data[label].astype(int)
    X = FEATURES
    dates = X.index

    retrain_dates = pd.date_range(dates[MIN_TRAIN_DAYS], dates[-1], freq=RETRAIN_FREQ)

    rows = []
    cur_pos = 0 # 0 = Risk Off (USD), 1 = Risk On (Merval)
    hold = 0

    for i, rd in enumerate(retrain_dates):
        tr_end = dates.get_indexer([rd], method="ffill")[0]
        te_end = (dates.get_indexer([retrain_dates[i+1]], method="ffill")[0] 
                  if i < len(retrain_dates)-1 else len(dates))

        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_te = X.iloc[tr_end:te_end]
        
        if len(X_te) == 0: continue

        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr)
        X_tes = sc.transform(X_te)

        clf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30, 
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_trs, y_tr)
        probs = clf.predict_proba(X_tes)[:,1]

        for d, prob in zip(X_te.index, probs):
            vix_norm = np.clip(data.loc[d,"global_vix_reg"]/4, 0, 1)
            thr = np.clip(BASE_THR - ALPHA*vix_norm, THR_MIN, THR_MAX)
            thr_in, thr_out = thr, np.clip(thr + HYST_GAP, THR_MIN, THR_MAX)

            # Signal Logic (Decision at Close t)
            if data.loc[d,"shock_nextday"] == 1:
                new_pos = 0
            elif hold > 0:
                new_pos = cur_pos
            else:
                if cur_pos == 1:
                    new_pos = 0 if prob >= thr_out else 1
                else:
                    new_pos = 1 if prob < thr_in else 0

            if new_pos != cur_pos:
                cur_pos = new_pos
                hold = MIN_HOLD_DAYS
            else:
                hold = max(0, hold-1)

            exp = 0.0 if cur_pos == 0 else float(np.clip(1 - prob/max(thr_in,1e-6), 0, 1))
            if policy_name == "crisis" and data.loc[d,"regime_score"] > CRISIS_REGIME_X:
                exp = min(exp, CRISIS_MAX_EXPOSURE)

            # RECORDING RESULTS
            # Crucial: The decision made at 'd' (Close) determines return for d+1 -> d+2
            # 'merval_ret_oo_next' is already aligned such that value at 'd' is the return 
            # from Open(d+1) to Open(d+2).
            
            strat_ret = 0.0
            
            # Gross Return
            r_merv = data.loc[d, "merval_ret_oo_next"]
            r_usd  = data.loc[d, "usd_ret_next"]
            
            # Simple Mixture: Exposure% in Merval, Rest in USD
            # Note: If exposure is 0, we are 100% USD.
            strat_ret = (exp * r_merv) + ((1 - exp) * r_usd)
            
            rows.append({
                "date": d,
                "position": int(cur_pos),
                "exposure": float(exp),
                "gross_ret": strat_ret
            })

    out = pd.DataFrame(rows).set_index("date").sort_index()

    # TRANSACTION COSTS
    # Costs apply when 'exposure' changes. 
    # Since we trade at Open, we pay costs on the delta.
    out["turnover"] = out["exposure"].diff().abs().fillna(0)
    out["net_ret"]  = out["gross_ret"] - (out["turnover"] * TRANSACTION_COST)
    
    return out

# ============================================================================
# RUN & REPORT
# ============================================================================
print("Running Overlay Strategy (Open-to-Open execution)...")
overlay = walk_forward("tail_overlay", "overlay")

print("Running Crisis Strategy (Open-to-Open execution)...")
crisis  = walk_forward("tail_crisis", "crisis")

# Benchmark: Buy & Hold (Open-to-Open to be fair comparison)
bnh_ret = data.loc[overlay.index, "merval_ret_oo_next"]

print("\n" + "="*60)
print("FINAL RESULTS: OPEN-TO-OPEN EXECUTION + USD RISK-OFF")
print("="*60)

metrics = {}
for n, r in {
  "OVERLAY": overlay["net_ret"],
  "CRISIS":  crisis["net_ret"],
  "BUY&HOLD": bnh_ret
}.items():
    ar, vol, sh, dd = perf_metrics(r)
    metrics[n] = {"AnnRet": ar, "Vol": vol, "Sharpe": sh, "MaxDD": dd}
    print(f"{n:10} | AnnRet {ar:6.2%} | Vol {vol:6.2%} | Sharpe {sh:5.2f} | MaxDD {dd:6.2%}")

print("-" * 60)
print("TRADING STATISTICS")
print(f"Overlay Switches: {int(overlay['turnover'].sum())} (approx)")
print(f"Overlay Avg Exp : {overlay['exposure'].mean():.2%}")

overlay.to_csv("overlay_oo_signals.csv")