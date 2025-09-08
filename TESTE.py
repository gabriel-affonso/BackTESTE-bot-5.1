#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, logging, warnings
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import DMatrix, Booster
import skops.io as sio

# -------------------------- USER CONFIG ---------------------------------

MODEL_DIR       = r"C:\Users\CES\Dropbox\Coisas\Coisas do PC\4\4.18_gpu_v5"
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.joblib")
CLF_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_classifier.joblib")  # if missing, we'll fallback to SCALER_PATH
REG_MODEL_PATH  = os.path.join(MODEL_DIR, "xgb_reg.json")
CLF_MODEL_PATH  = os.path.join(MODEL_DIR, "stacked_clf_v5.pkl")
FEATURE_TXT     = os.path.join(MODEL_DIR, "feature_columns.txt")       # for regressor scaler
FEATURE_CLF_TXT = os.path.join(MODEL_DIR, "feature_classifier.txt")    # for classifier scaler

# Debugging: optionally dump feature names/shapes seen at inference to a local log
DEBUG_FEATURE_IO = True
FEATURE_DEBUG_PATH = os.path.join(os.path.dirname(__file__), "feature_debug.log")

# Random Forest gate (optional)
RF_DIR          = r"C:\Users\CES\Dropbox\Coisas\Coisas do PC\4\6.04"
RF_SKOPS_PATH   = os.path.join(RF_DIR, "rf_model.joblib")
RF_FEATURES_TXT = os.path.join(RF_DIR, "rf_feature_columns.txt")

# Strategy thresholds (keep same as live)
CLF_THRESHOLD     = 0.60
IA_REG_THRESHOLD  = 0.008

# Money management & fills
INITIAL_CAPITAL   = 10_000.0
BASE_RISK         = 0.03
SL_PCT_LONG       = 0.005     # 0.5%
TP_PCT_LONG       = 0.02      # 2.0%
BE_TRIGGER        = 0.005     # 0.5%
PARTIAL_PCT       = 0.008     # +0.8%
PARTIAL_SIZE      = 0.50
TAKER_FEE         = 0.0010    # 0.10% per side (adjust to your venue)
SLIPPAGE_PCT      = 0.0005    # 0.05% slippage (optional)
DD_STOP_PCT       = 0.10      # reduce risk if >10% DD
ENTER_NEXT_BAR    = True      # to avoid look-ahead

EMA_SHORT = 9
EMA_MED   = 50
EMA_LONG  = 200

WARMUP_BARS_15M = max(EMA_LONG, 200) + 30   # large enough to build features safely
WARMUP_BARS_1H  = 60

# Intrabar resolution rule when both SL and TP occur in the same bar
INTRABAR_PRIORITY = "stop_first"   # or "target_first"

# -------------------------- UTILITIES -----------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def pct(a, b): 
    return (a/b - 1.0) if b else 0.0

def apply_fee(price, side):
    # Apply taker fee to execution price as effective “worse” price
    # For buys, we pay slightly more; for sells, we receive slightly less.
    if side == "buy":
        return price * (1 + TAKER_FEE)
    else:
        return price * (1 - TAKER_FEE)

def apply_slippage(price, side):
    return price * (1 + SLIPPAGE_PCT) if side == "buy" else price * (1 - SLIPPAGE_PCT)

def worst_case_fill(price, side):
    return apply_fee(apply_slippage(price, side), side)

def ensure_all_features(df, features):
    missing = [c for c in features if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df.reindex(columns=features)

# ---------------------- FEATURE ENGINEERING (OFFLINE) -------------------

def resample_1m_to_15m(df1m):
    # expects index = UTC Timestamp at 1m
    out = df1m.resample('15min', label='left', closed='left').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    })
    return out.dropna()

def resample_1m_to_1h(df1m):
    out = df1m.resample('1h', label='left', closed='left').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    })
    return out.dropna()

def compute_15m_indicators(df15):
    df15['ema9']   = df15['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df15['ema50']  = df15['close'].ewm(span=EMA_MED,   adjust=False).mean()
    df15['ema200'] = df15['close'].ewm(span=EMA_LONG,  adjust=False).mean()
    df15['vol_sma']= df15['volume'].rolling(20).mean()
    return df15

def compute_1h_macd(df1h):
    ema12 = df1h['close'].ewm(span=12, adjust=False).mean()
    ema26 = df1h['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig  = macd.ewm(span=9, adjust=False).mean()
    return macd, sig

def full_feature_block_15m(df15):
    from ta.trend      import ADXIndicator
    from ta.momentum   import RSIIndicator, StochasticOscillator, TSIIndicator
    from ta.volume     import MFIIndicator
    from ta.volatility import BollingerBands

    out = df15.copy()
    out['return']       = out['close'].pct_change()
    out['price_change'] = out['close'] - out['open']
    out['volatility']   = out['return'].rolling(14).std()
    out['direction']    = np.sign(out['return']).fillna(0)

    out['sma_20']  = out['close'].rolling(20).mean()
    out['ema_50']  = out['close'].ewm(span=50,  adjust=False).mean()
    out['ema_9']   = out['close'].ewm(span=9,   adjust=False).mean()
    out['ema_200'] = out['close'].ewm(span=200, adjust=False).mean()

    out['adx_14']  = ADXIndicator(out['high'], out['low'], out['close'], window=14).adx()
    out['rsi_14']  = RSIIndicator(out['close'], window=14).rsi()
    sto            = StochasticOscillator(out['high'], out['low'], out['close'], window=14, smooth_window=3)
    out['sto_k']   = sto.stoch()
    out['sto_d']   = sto.stoch_signal()
    out['macd']    = out['close'].ewm(span=12, adjust=False).mean() - out['close'].ewm(span=26, adjust=False).mean()
    out['macd_sig']= out['macd'].ewm(span=9, adjust=False).mean()
    out['roc_10']  = out['close'].pct_change(10)
    out['tsi_25']  = TSIIndicator(out['close'], window_slow=25, window_fast=13).tsi()

    hl = out['high'] - out['low']
    hc = (out['high'] - out['close'].shift()).abs()
    lc = (out['low']  - out['close'].shift()).abs()
    out['atr_14']   = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    bb             = BollingerBands(out['close'], window=20, window_dev=2)
    out['bb_width']     = bb.bollinger_hband() - bb.bollinger_lband()
    out['bb_percent_b'] = bb.bollinger_pband()

    out['dc_width']     = out['high'].rolling(20).max() - out['low'].rolling(20).min()
    out['vol_ma_20']    = out['volume'].rolling(20).mean()
    out['vol_ratio_20'] = out['volume'] / out['vol_ma_20']
    out['obv']          = (np.sign(out['close'].diff()) * out['volume']).cumsum()
    out['vpt']          = (out['close'].pct_change() * out['volume']).cumsum()

    for p in [5,10,20,60]:
        out[f'ret_{p}']       = out['close'].pct_change(p)
        out[f'logret_{p}']    = np.log(out['close']/out['close'].shift(p))
        out[f'roll_std_{p}']  = out['return'].rolling(p).std()
        out[f'roll_skew_{p}'] = out['return'].rolling(p).skew()

    # compatibility with old scaler
    out['roll_std_6'] = out['return'].rolling(6).std()

    # candle anatomy
    out['upper_shadow'] = out['high'] - out[['close','open']].max(axis=1)
    out['lower_shadow'] = out[['close','open']].min(axis=1) - out['low']
    out['body_size']    = (out['close'] - out['open']).abs()

    # MFI
    out['mfi_14'] = MFIIndicator(out['high'], out['low'], out['close'], out['volume'], window=14).money_flow_index()

    # “return_5min” alias used in your pipeline
    out['return_5min'] = out['close'].pct_change(5)

    # calendar/time features
    out['minute']             = out.index.minute
    out['hour']               = out.index.hour
    out['dayofweek']          = out.index.dayofweek
    out['dayofmonth']         = out.index.day
    out['month']              = out.index.month
    out['year']               = out.index.year
    out['weekday']            = out.index.weekday
    out['is_month_end']       = out.index.is_month_end.astype(int)
    out['is_month_start']     = out.index.is_month_start.astype(int)
    out['is_quarter_end']     = out.index.is_quarter_end.astype(int)
    out['mins_since_daystart']= out.index.hour*60 + out.index.minute
    out['hour_sin']           = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos']           = np.cos(2*np.pi*out['hour']/24)
    out['dow_sin']            = np.sin(2*np.pi*out['dayofweek']/7)
    out['dow_cos']            = np.cos(2*np.pi*out['dayofweek']/7)
    return out

# ---------------------- MODELS (LOAD) -----------------------------------

def load_models():
    scaler = joblib.load(SCALER_PATH)
    try:
        scaler_clf = joblib.load(CLF_SCALER_PATH)
    except Exception:
        scaler_clf = scaler

    reg = Booster()
    reg.load_model(REG_MODEL_PATH)
    clf: CalibratedClassifierCV = joblib.load(CLF_MODEL_PATH)

    # feature lists
    if os.path.exists(FEATURE_TXT):
        with open(FEATURE_TXT) as f: feat_reg = [l.strip() for l in f if l.strip()]
    else:
        feat_reg = list(getattr(scaler, "feature_names_in_", []))

    if os.path.exists(FEATURE_CLF_TXT):
        with open(FEATURE_CLF_TXT) as f: feat_clf = [l.strip() for l in f if l.strip()]
    else:
        feat_clf = list(getattr(scaler_clf, "feature_names_in_", []))

    # RF (optional)
    rf_model = None
    rf_feats = []
    try:
        if os.path.exists(RF_SKOPS_PATH):
            rf_model = sio.load(RF_SKOPS_PATH)
        if os.path.exists(RF_FEATURES_TXT):
            with open(RF_FEATURES_TXT) as f:
                rf_feats = [l.strip() for l in f if l.strip()]
    except Exception:
        rf_model = None
        rf_feats = []

    return scaler, scaler_clf, reg, clf, feat_reg, feat_clf, rf_model, rf_feats

# ---------------------- SIGNALS (OFFLINE) --------------------------------

def math_gate(df15, df1h, t_idx):
    """Implements your 5 conditions, using bars <= t_idx only."""
    if t_idx < 4: 
        return False
    bar15 = df15.iloc[t_idx]
    price = bar15['close']

    macd, macd_sig = compute_1h_macd(df1h.iloc[:])  # precomputed ok
    # align to nearest 1h bar <= df15.index[t_idx]
    last1h = df1h.index[df1h.index <= df15.index[t_idx]]
    if len(last1h) == 0:
        return False
    macd_last = macd.loc[last1h[-1]]
    macd_sig_last = macd_sig.loc[last1h[-1]]

    cond1 = (bar15['ema50'] > bar15['ema200']) and (macd_last > macd_sig_last)
    # recent 1h bar bullish (the last fully closed 1h bar)
    bar1h = df1h.loc[last1h[-1]]
    cond2 = bar1h['close'] > bar1h['open']

    ema50_recent = df15['ema50'].iloc[t_idx-3:t_idx]  # last 3 fully closed before t
    cond3 = ema50_recent.is_monotonic_increasing if len(ema50_recent)==3 else False
    cond4 = bar15['volume'] >= bar15['vol_sma']
    cond5 = (price > bar15['ema9']) and (bar15['ema9'] > bar15['ema50'])

    score = sum([cond1, cond2, cond3, cond4, cond5])
    return score >= 1   # your code currently uses >=1; set to >=3 if you want stricter

def ia_reg_gate(feat_row, scaler, feat_reg, reg):
    # Use the feature list provided (from FEATURE_TXT) as the canonical input for the regressor.
    # This ensures we only pass the exact features the regressor expects according to the saved list.
    cols = list(feat_reg)

    X = ensure_all_features(feat_row.to_frame().T, cols).astype('float32')
    # scaler.transform may accept a DataFrame; try passing the DataFrame first for name-aware scalers
    try:
        Xs = scaler.transform(X)
    except Exception:
        Xs = scaler.transform(X.values)

    # debug dump
    if DEBUG_FEATURE_IO:
        msg1 = f"IA_REG | cols_expected={cols}"
        msg2 = f"IA_REG | X columns={list(X.columns)} shape={X.shape}"
        try:
            msg3 = f"IA_REG | Xs type={type(Xs)}"
        except Exception:
            msg3 = "IA_REG | Xs type=unknown"
        try:
            with open(FEATURE_DEBUG_PATH, 'a', encoding='utf8') as f:
                f.write(msg1 + "\n")
                f.write(msg2 + "\n")
                f.write(msg3 + "\n---\n")
        except Exception:
            pass
        try:
            logging.info(msg1)
            logging.info(msg2)
            logging.info(msg3)
        except Exception:
            pass

    pred = float(reg.predict(DMatrix(Xs))[0])
    return pred >= IA_REG_THRESHOLD

def ia_clf_gate(feat_row, scaler_clf, feat_clf, clf):
    # Use the feature list provided for the classifier (from FEATURE_CLF_TXT) as canonical input
    # to the scaler. After scaling, subset the transformed DataFrame to the features the
    # stacking base learners actually expect.

    scaler_cols = list(feat_clf)

    # Build full feature DataFrame according to feat_clf
    X_full = ensure_all_features(feat_row.to_frame().T, scaler_cols).astype('float32')
    try:
        Xs_full = scaler_clf.transform(X_full)
    except Exception:
        Xs_full = scaler_clf.transform(X_full.values)

    # Coerce transformed data back to DataFrame with feat_clf columns
    try:
        Xs_full_df = pd.DataFrame(Xs_full, columns=scaler_cols)
    except Exception:
        Xs_full_df = pd.DataFrame(np.asarray(Xs_full), columns=scaler_cols)

    # --- detect base learner expected columns (try classifier, pipelines, or base estimators) ---
    def _extract_cols(est):
        """Return feature column names used by ``est`` if available."""
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
        if hasattr(est, "get_booster"):
            try:
                bn = est.get_booster().feature_names
                if bn:
                    return list(bn)
            except Exception:
                pass
        if hasattr(est, "n_features_in_"):
            # no explicit names, fallback to first n columns from scaler_cols
            return list(scaler_cols[: est.n_features_in_])
        return None

    base_cols = _extract_cols(clf)

    if base_cols is None:
        for attr in (
            "calibrated_classifiers_",
            "estimators_",
            "estimators",
            "base_estimator_",
            "final_estimator_",
        ):
            if not hasattr(clf, attr):
                continue
            cand = getattr(clf, attr)
            # ``cand`` may be a list/tuple of estimators or a single estimator
            cand_list = cand if isinstance(cand, (list, tuple)) else [cand]
            for est in cand_list:
                # if estimator is a named tuple (name, estimator) take the estimator
                if isinstance(est, tuple) and len(est) > 1:
                    est = est[1]
                # dive into pipelines to reach the final estimator
                while hasattr(est, "steps") and len(est.steps) > 0:
                    est = est.steps[-1][1]
                cols = _extract_cols(est)
                if cols is not None:
                    base_cols = cols
                    break
            if base_cols is not None:
                break

    if base_cols is None:
        # if we still couldn't detect base_cols, fall back to the scaler column list
        base_cols = list(scaler_cols)

    # final columns to pass to base learners: intersection of base_cols and feat_clf (preserves order in base_cols)
    final_cols = [c for c in base_cols if c in scaler_cols]
    Xs = Xs_full_df.reindex(columns=final_cols)

    # debug dump (report scaler/full vs base subset)
    if DEBUG_FEATURE_IO:
        try:
            msg1 = f"IA_CLF | scaler_cols={scaler_cols}"
            msg2 = f"IA_CLF | base_cols={base_cols}"
            msg3 = f"IA_CLF | X_full_df columns={list(X_full.columns)} shape={X_full.shape}"
            msg4 = f"IA_CLF | Xs_full_df columns={list(Xs_full_df.columns)} shape={Xs_full_df.shape}"
            msg5 = f"IA_CLF | Xs (final) columns={list(Xs.columns)} shape={Xs.shape}"
        except Exception:
            msg1 = msg2 = msg3 = msg4 = msg5 = "IA_CLF | debug info unavailable"
        try:
            with open(FEATURE_DEBUG_PATH, 'a', encoding='utf8') as f:
                f.write(msg1 + "\n")
                f.write(msg2 + "\n")
                f.write(msg3 + "\n")
                f.write(msg4 + "\n")
                f.write(msg5 + "\n---\n")
        except Exception:
            pass
        try:
            logging.info(msg1)
            logging.info(msg2)
            logging.info(msg3)
            logging.info(msg4)
            logging.info(msg5)
        except Exception:
            pass

    # final predict: pass DataFrame (with proper column names) when possible to keep order
    try:
        proba = float(clf.predict_proba(Xs)[0,1])
    except Exception:
        proba = float(clf.predict_proba(Xs.values)[0,1])
    return proba > CLF_THRESHOLD

def rf_gate(feat_row_for_rf, rf_model, rf_feats):
    if rf_model is None or len(rf_feats)==0:
        return True  # no RF gate if model not present
    X = ensure_all_features(feat_row_for_rf.to_frame().T, rf_feats).astype('float32').values
    label = int(rf_model.predict(X)[0])
    return label == 2

# ---------------------- TRADING ENGINE ----------------------------------

@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    sl: float
    tp: float
    be_trigger: float
    partial_taken: bool = False
    be_moved: bool = False

def simulate_bar_outcome(long_pos: Position, bar_open, bar_high, bar_low, bar_close):
    """Check partial/BE/TP/SL within a single 15m bar (intrabar ordering rule)."""
    # Partial at +0.8%, BE trigger at +0.5%, TP +2%, SL -0.5%
    p = long_pos

    # compute trigger prices
    partial_trg = p.entry_price * (1 + PARTIAL_PCT)
    be_trg      = p.entry_price * (1 + BE_TRIGGER)
    tp_trg      = p.entry_price * (1 + TP_PCT_LONG)
    sl_trg      = p.sl

    # We’ll decide if the bar touches these levels
    def hit(level, direction="up"):
        return (bar_high >= level) if direction=="up" else (bar_low <= level)

    # Order of events inside a bar
    events = []
    if not p.partial_taken and hit(partial_trg, "up"): events.append("partial")
    if not p.be_moved and hit(be_trg, "up"):          events.append("be")
    # Depending on policy:
    if INTRABAR_PRIORITY == "stop_first":
        if hit(sl_trg, "down"):  events.insert(0, "sl")
        if hit(tp_trg, "up"):    events.append("tp")
    else:
        if hit(tp_trg, "up"):    events.insert(0, "tp")
        if hit(sl_trg, "down"):  events.append("sl")

    return events

def backtest_symbol(df1m_symbol: pd.DataFrame,
                    scaler, scaler_clf, reg, clf, feat_reg, feat_clf,
                    rf_model, rf_feats):
    # 1m must be UTC indexed
    df1m_symbol = df1m_symbol.sort_index()
    df15 = resample_1m_to_15m(df1m_symbol)
    df1h = resample_1m_to_1h(df1m_symbol)

    # Indicators for MATH gate
    df15 = compute_15m_indicators(df15)

    # Full features for IA gates
    feat15 = full_feature_block_15m(df15)

    # If you want an RF feature set that is different, reuse feat15 last-row slice.
    rf_feat15 = feat15  # adjust here if your RF used a different recipe

    # Walk forward
    equity = INITIAL_CAPITAL
    peak   = INITIAL_CAPITAL
    trades = []
    pos: Position | None = None

    # warmup windows
    start_idx = max(WARMUP_BARS_15M, 1)

    for t in range(start_idx, len(df15)-1):  # ensure we have a "next bar" for entry
        bar_time = df15.index[t]
        prev_1h = df1h.index[df1h.index <= bar_time]
        if len(prev_1h) < WARMUP_BARS_1H:
            continue

        # manage open position first on this fully closed bar
        if pos is not None:
            # evaluate outcomes on *this* already closed bar
            bar = df15.iloc[t]
            events = simulate_bar_outcome(pos, bar['open'], bar['high'], bar['low'], bar['close'])

            # apply events in order
            for ev in events:
                if ev == "partial" and not pos.partial_taken:
                    sell_qty = pos.qty * PARTIAL_SIZE
                    fill_px  = worst_case_fill(bar['high'], "sell")  # optimistic partial at high touch
                    proceeds = sell_qty * fill_px
                    equity  += proceeds - (sell_qty * pos.entry_price)  # realized pnl
                    pos.qty *= (1 - PARTIAL_SIZE)
                    pos.sl   = pos.entry_price  # move to BE
                    pos.partial_taken = True
                    trades.append({"time": bar_time, "type":"PARTIAL", "price": float(fill_px), "qty": float(sell_qty)})
                elif ev == "be" and not pos.be_moved:
                    pos.sl = pos.entry_price
                    pos.be_moved = True
                elif ev == "tp":
                    sell_qty = pos.qty
                    fill_px  = worst_case_fill(bar['high'], "sell")
                    proceeds = sell_qty * fill_px
                    equity  += proceeds - (sell_qty * pos.entry_price)
                    trades.append({"time": bar_time, "type":"TP", "price": float(fill_px), "qty": float(sell_qty)})
                    pos = None
                    break
                elif ev == "sl":
                    sell_qty = pos.qty
                    fill_px  = worst_case_fill(bar['low'], "sell")
                    proceeds = sell_qty * fill_px
                    equity  += proceeds - (sell_qty * pos.entry_price)
                    trades.append({"time": bar_time, "type":"SL", "price": float(fill_px), "qty": float(sell_qty)})
                    pos = None
                    break

        # if position closed above, skip entry this bar (enter next bar if signal)
        if pos is None:
            # gates computed on *closed* bar t using features up to t
            math_ok = math_gate(df15, df1h, t)
            if not math_ok: 
                continue

            feat_row = feat15.iloc[t]  # features “as of” bar t
            ia_ok = ia_reg_gate(feat_row, scaler, feat_reg, reg)
            if not ia_ok:
                continue

            clf_ok = ia_clf_gate(feat_row, scaler_clf, feat_clf, clf)
            if not clf_ok:
                continue

            rf_ok = rf_gate(rf_feat15.iloc[t], rf_model, rf_feats)
            if not rf_ok:
                continue

            # Entry at NEXT bar open (no look-ahead)
            nxt = t+1
            open_px = df15.iloc[nxt]['open']
            fill_px = worst_case_fill(open_px, "buy")

            # dynamic risk like your loop (simple version)
            # compute equity peak and adjust risk
            peak = max(peak, equity)
            drawdown = (equity/peak) - 1.0
            risk = 0.005 if drawdown < -DD_STOP_PCT else BASE_RISK

            sl_price = fill_px * (1 - SL_PCT_LONG)
            tp_price = fill_px * (1 + TP_PCT_LONG)
            be_trg   = fill_px * (1 + BE_TRIGGER)

            risk_amt = equity * risk
            risk_per_unit = max(fill_px - sl_price, 1e-8)
            qty = (risk_amt / risk_per_unit)

            pos = Position(
                entry_time = df15.index[nxt],
                entry_price= float(fill_px),
                qty        = float(qty),
                sl         = float(sl_price),
                tp         = float(tp_price),
                be_trigger = float(be_trg)
            )
            cost = qty * fill_px
            equity -= cost  # convert to “invested” notion (cash + position value)
            trades.append({"time": df15.index[nxt], "type":"BUY", "price": float(fill_px), "qty": float(qty)})

    # If a position remains open, close at last close for reporting
    if pos is not None:
        last = df15.iloc[-1]
        fill_px = worst_case_fill(last['close'], "sell")
        proceeds = pos.qty * fill_px
        equity += proceeds - (pos.qty * pos.entry_price)
        trades.append({"time": df15.index[-1], "type":"FORCED_EXIT", "price": float(fill_px), "qty": float(pos.qty)})
        pos = None

    # Build equity curve (simple: start + realized cashflow deltas from trades)
    equity_curve = pd.DataFrame(trades).copy()
    if not equity_curve.empty:
        equity_curve['cashflow'] = 0.0
        # BUY reduces equity (cash), SELL increases equity
        for i, row in equity_curve.iterrows():
            if row['type'] == "BUY":
                equity_curve.at[i, 'cashflow'] = -row['price'] * row['qty']
            else:
                equity_curve.at[i, 'cashflow'] = row['price'] * row['qty']
        equity_curve = equity_curve.set_index('time').sort_index()
        equity_curve['equity'] = INITIAL_CAPITAL + equity_curve['cashflow'].cumsum()
    else:
        equity_curve = pd.DataFrame(columns=['equity'])

    return pd.DataFrame(trades), equity_curve

# ---------------------- METRICS -----------------------------------------

def summarize(trades_df, equity_curve):
    if trades_df.empty:
        return {"trades":0, "net":0.0, "win_rate":0.0, "pf":0.0, "max_dd":0.0, "sharpe":0.0}

    # wins/losses
    realized = []
    pos_cost = 0.0
    qty_running = 0.0
    entry_px = None
    for _, r in trades_df.iterrows():
        if r['type'] == 'BUY':
            pos_cost += r['price'] * r['qty']
            qty_running += r['qty']
            entry_px = r['price']
        else:
            pnl = (r['price'] - entry_px) * r['qty'] if entry_px is not None else 0.0
            realized.append(pnl)

    wins = [x for x in realized if x>0]
    losses = [-x for x in realized if x<0]
    win_rate = (len(wins) / len(realized)) if realized else 0.0
    pf = (sum(wins)/sum(losses)) if losses else np.inf
    net = sum(realized)

    # max drawdown & Sharpe from equity curve (15m → daily approx)
    if not equity_curve.empty and 'equity' in equity_curve:
        e = equity_curve['equity'].dropna()
        roll_max = e.cummax()
        dd = (e/roll_max - 1.0).min()
        ret = e.pct_change().dropna()
        sharpe = (ret.mean() / (ret.std()+1e-12)) * np.sqrt(96)  # ~96 15m bars per day
    else:
        dd, sharpe = 0.0, 0.0

    return {
        "trades": int((trades_df['type']=='BUY').sum()),
        "net": float(net),
        "win_rate": float(win_rate),
        "pf": float(pf if np.isfinite(pf) else 0.0),
        "max_dd": float(dd),
        "sharpe": float(sharpe),
    }

# ---------------------- DRIVER -----------------------------------------

def run_backtest(csv_path, symbol_col=None, symbol=None, tz='UTC'):
    setup_logging()
    logging.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Expect columns: timestamp, open, high, low, close, volume, [symbol?]
    # Normalize timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    elif 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
        df = df.set_index('open_time')
    else:
        raise ValueError("CSV must have 'timestamp' or 'open_time' column.")

    # If multi-symbol CSV, select one symbol at a time
    if symbol_col and symbol_col in df.columns:
        symbols = [symbol] if symbol else sorted(df[symbol_col].unique().tolist())
    else:
        symbols = [symbol] if symbol else [None]

    scaler, scaler_clf, reg, clf, feat_reg, feat_clf, rf_model, rf_feats = load_models()
    all_trades = []
    curve_parts = []

    for sym in symbols:
        if sym:
            logging.info(f"Backtesting symbol: {sym}")
            df_sym = df[df[symbol_col]==sym][['open','high','low','close','volume']].copy()
        else:
            logging.info(f"Backtesting single series (no symbol column)")
            cols_needed = ['open','high','low','close','volume']
            missing = [c for c in cols_needed if c not in df.columns]
            if missing:
                raise ValueError(f"CSV missing columns: {missing}")
            df_sym = df[cols_needed].copy()

        df_sym = df_sym.dropna().sort_index()

        trades_df, equity_curve = backtest_symbol(
            df_sym, scaler, scaler_clf, reg, clf, feat_reg, feat_clf, rf_model, rf_feats
        )
        trades_df['symbol'] = sym if sym else "SERIES"
        all_trades.append(trades_df)
        if not equity_curve.empty:
            equity_curve['symbol'] = sym if sym else "SERIES"
            curve_parts.append(equity_curve)

    all_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_curve = pd.concat(curve_parts) if curve_parts else pd.DataFrame()

    stats = summarize(all_trades, equity_curve)
    logging.info(f"RESULTS: {json.dumps(stats, indent=2)}")

    # Save optional outputs
    base = os.path.splitext(os.path.basename(csv_path))[0]
    trades_path = f"{base}_bt_trades.csv"
    curve_path  = f"{base}_bt_equity.csv"
    all_trades.to_csv(trades_path, index=False)
    equity_curve.to_csv(curve_path)
    logging.info(f"Saved trades → {trades_path}")
    logging.info(f"Saved equity → {curve_path}")

    return stats

if __name__ == "__main__":
    # Example:
    run_backtest(r"C:\Users\CES\Dropbox\Coisas\Coisas do PC\4\all_data_enriched2.csv", symbol_col="symbol")
    pass
