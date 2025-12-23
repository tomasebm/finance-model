# ============================================================================
# MERVAL TAIL-RISK SYSTEM — FINAL (CCL + OPEN-TO-OPEN + CRISIS CAP)
# FIXES:
# 1) Open-to-Open execution (decision at Close t -> P&L Open t+1 to Open t+2)
# 2) Risk-off leg uses implied CCL (GGAL.BA / GGAL * 10)
# 3) Robust Yahoo open/close cleaning + holiday mismatch handling
# 4) FINAL Crisis rule: if regime_score > 0.30 => cap exposure at 10%
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
TRANSACTION_COST = 0.0015

# Shock override
SHOCK_DAILY_RET = -0.055
VIX_JUMP_2D     = 0.22

# Labels
OVERLAY_LOSS_THR = -0.020
OVERLAY_WIN      = 7

CRISIS_DD_THR    = -0.12
CRISIS_MIN_DAYS  = 10
CRISIS_LOOKAHEAD = 15

# ============================================================================
# FINAL PRODUCTION CONFIG (CRISIS CAP)
# ============================================================================
CRISIS_REGIME_X     = 0.30
CRISIS_MAX_EXPOSURE = 0.10

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
    df = yf.download(ticker, START_DATE, END_DATE, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    if "close" in df.columns:
        df["close"] = df["close"].replace(0, np.nan).ffill()

    if "open" in df.columns:
        df["open"] = df["open"].replace(0, np.nan).fillna(df["close"])

    return df

def align(s, idx):
    return s.reindex(idx).ffill(limit=FFILL_LIMIT)

def max_dd(r):
    r = r.fillna(0.0)
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())

def perf_metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0, 0.0
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sh = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sh), max_dd(r)

# ============================================================================
# DATA LOADING
# ============================================================================
m_df = yf_fetch("^MERV")
if m_df is None or m_df.empty:
    m_df = yf_fetch("GGAL")

m_df = m_df.sort_index()
MASTER = m_df.index

levels = pd.DataFrame(index=MASTER)
levels["merval_close"] = m_df["close"]
levels["merval_open"]  = m_df["open"]

vix_df  = yf_fetch("^VIX")
argt_df = yf_fetch("ARGT")
ggal_us_df = yf_fetch("GGAL")
ggal_ba_df = yf_fetch("GGAL.BA")

if vix_df is None or argt_df is None or ggal_us_df is None or ggal_ba_df is None:
    raise RuntimeError("Missing critical data (VIX/ARGT/GGAL/GGAL.BA).")

levels["vix"]  = align(vix_df["close"], MASTER)
levels["argt"] = align(argt_df["close"], MASTER)

raw_ccl = (ggal_ba_df["close"] / ggal_us_df["close"]) * 10.0
levels["usd_ccl"] = raw_ccl.reindex(MASTER).ffill()

# ============================================================================
# RETURNS COMPUTATION
# ============================================================================
ret = pd.DataFrame(index=MASTER)

# Signals/features: Close-to-Close
ret["merval_ret_cc"] = levels["merval_close"].pct_change()

# Execution P&L: Open(t+1)->Open(t+2) aligned to t
ret["merval_ret_oo_next"] = (levels["merval_open"].shift(-2) / levels["merval_open"].shift(-1) - 1)

# Risk-off (CCL): (t+2)/(t+1) aligned to t
ret["usd_ret_next"] = levels["usd_ccl"].pct_change().shift(-2).fillna(0.0)

ret["vix_reg"] = pd.cut(levels["vix"], [0,15,20,30,40,100], labels=[0,1,2,3,4]).astype(float)
ret["argt_stress"] = -np.log(levels["argt"]).diff()

cum = (1 + ret["merval_ret_cc"].fillna(0.0)).cumprod()
ret["dd"] = cum / cum.cummax() - 1

ret["regime_score"] = (
    0.5 * (ret["vix_reg"] / 4).fillna(0.0) +
    0.5 * ret["argt_stress"].rolling(5).mean().clip(0, 0.05).fillna(0.0) / 0.05
)

# ============================================================================
# LABELS
# ============================================================================
_overlay_raw = (ret["merval_ret_cc"] < OVERLAY_LOSS_THR).rolling(OVERLAY_WIN).max().shift(-1)
ret["tail_overlay"] = _overlay_raw.fillna(0).astype(int)

deep_dd = (ret["dd"] <= CRISIS_DD_THR).astype(int)
deep_dd_persistent = (deep_dd.rolling(CRISIS_MIN_DAYS).sum() >= CRISIS_MIN_DAYS).astype(int)
_crisis_raw = deep_dd_persistent.rolling(CRISIS_LOOKAHEAD).max().shift(-1)
ret["tail_crisis"] = _crisis_raw.fillna(0).astype(int)

shock_ret = (ret["merval_ret_cc"] <= SHOCK_DAILY_RET).astype(int)
vix_jump  = (levels["vix"] / levels["vix"].shift(2) - 1 >= VIX_JUMP_2D).astype(int)
ret["shock_nextday"] = (shock_ret | vix_jump).shift(1).fillna(0).astype(int)

# ============================================================================
# FEATURES
# ============================================================================
feat = pd.DataFrame(index=MASTER)
feat["global_vix_reg"] = ret["vix_reg"]
feat["local_argt_5d"]  = ret["argt_stress"].rolling(5).mean()
feat["damage_ret_5d"]  = ret["merval_ret_cc"].rolling(5).sum()
feat["damage_vol_10d"] = ret["merval_ret_cc"].rolling(10).std()
feat["damage_dd"]      = ret["dd"]
feat["dd_chg_5d"]      = ret["dd"].diff(5)

data = pd.concat([ret, feat], axis=1).dropna()

FEATURES = data[[
    "global_vix_reg","local_argt_5d","damage_ret_5d",
    "damage_vol_10d","damage_dd","dd_chg_5d"
]]

# ============================================================================
# WALK-FORWARD (OPEN-TO-OPEN EXECUTION)
# ============================================================================
def walk_forward(label, policy_name):
    p = POLICY[policy_name]
    BASE_THR = p["BASE_THR"]
    HYST_GAP = p["HYST_GAP"]
    MIN_HOLD = p["MIN_HOLD_DAYS"]

    y = data[label].astype(int)
    X = FEATURES
    dates = X.index

    retrain_dates = pd.date_range(dates[MIN_TRAIN_DAYS], dates[-1], freq=RETRAIN_FREQ)

    rows = []
    cur_pos = 0  # 0=USD(CCL), 1=Equity
    hold = 0

    for i, rd in enumerate(retrain_dates):
        tr_end = dates.get_indexer([rd], method="ffill")[0]
        te_end = dates.get_indexer([retrain_dates[i+1]], method="ffill")[0] if i < len(retrain_dates)-1 else len(dates)

        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_te = X.iloc[tr_end:te_end]
        if len(X_te) == 0:
            continue

        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr)
        X_tes = sc.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_trs, y_tr)
        probs = clf.predict_proba(X_tes)[:, 1]

        for d, prob in zip(X_te.index, probs):
            vix_norm = np.clip(data.loc[d, "global_vix_reg"] / 4, 0, 1)
            thr = np.clip(BASE_THR - ALPHA * vix_norm, THR_MIN, THR_MAX)
            thr_in  = thr
            thr_out = np.clip(thr + HYST_GAP, THR_MIN, THR_MAX)

            if data.loc[d, "shock_nextday"] == 1:
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
                hold = MIN_HOLD
            else:
                hold = max(0, hold - 1)

            exp = 0.0 if cur_pos == 0 else float(np.clip(1 - prob / max(thr_in, 1e-6), 0, 1))

            # FINAL CRISIS RULE (cap exposure in high-stress regimes)
            if policy_name == "crisis" and data.loc[d, "regime_score"] > CRISIS_REGIME_X:
                exp = min(exp, CRISIS_MAX_EXPOSURE)

            r_m = data.loc[d, "merval_ret_oo_next"]
            r_u = data.loc[d, "usd_ret_next"]
            if pd.isna(r_m): r_m = 0.0
            if pd.isna(r_u): r_u = 0.0

            gross = exp * r_m + (1 - exp) * r_u

            rows.append({
                "date": d,
                "position": int(cur_pos),
                "exposure": float(exp),
                "gross_ret": float(gross),
            })

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["turnover"] = out["exposure"].diff().abs().fillna(0.0)
    out["net_ret"]  = out["gross_ret"] - out["turnover"] * TRANSACTION_COST
    return out

# ============================================================================
# RUN + REPORT
# ============================================================================
overlay = walk_forward("tail_overlay", "overlay")
crisis  = walk_forward("tail_crisis",  "crisis")

bnh_ret = data.loc[overlay.index, "merval_ret_oo_next"]

for name, series in {
    "OVERLAY": overlay["net_ret"],
    "CRISIS":  crisis["net_ret"],
    "BUYHOLD": bnh_ret
}.items():
    ar, vol, sh, dd = perf_metrics(series)
    print(f"{name} | AnnRet {ar:.2%} | Vol {vol:.2%} | Sharpe {sh:.2f} | MaxDD {dd:.2%}")

overlay.to_csv("overlay_oo_signals_ccl_final.csv")
crisis.to_csv("crisis_oo_signals_ccl_final.csv")
