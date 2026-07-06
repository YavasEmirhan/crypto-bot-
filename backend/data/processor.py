"""
Advanced Technical Analysis Feature Engineering — v2
================================================
Full integration based on research findings:
  ✓ Normalized EMA/SMA ratios (not raw price)
  ✓ Ichimoku Cloud (NO look-ahead bias, ATR normalized)
  ✓ Candlestick patterns (15+ with trend context)
  ✓ Multi-timeframe (4H + 1D resample)
  ✓ Market microstructure (buy/sell vol, Amihud, efficiency)
  ✓ Volatility regime (ADX, squeeze, HV ratio)
  ✓ Support/Resistance (dynamic)
  ✓ Time features (hour/day cyclical encoding)
  ✓ Dynamic labeling (ATR-based threshold)
"""

import pandas as pd
import numpy as np
import ta


# ─────────────────────────────────────────────────────────────────────────────
# 1. TREND (normalized ratios)
# ─────────────────────────────────────────────────────────────────────────────

def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    ema = {}
    sma = {}
    for p in [9, 21, 50, 200]:
        ema[p] = ta.trend.ema_indicator(c, window=p)
        sma[p] = ta.trend.sma_indicator(c, window=p)
        # Normalize: price's position relative to EMA
        df[f"ema{p}_ratio"]  = c / (ema[p] + 1e-9) - 1
        df[f"sma{p}_ratio"]  = c / (sma[p] + 1e-9) - 1

    # EMA cross ratios (independent of candle price)
    df["ema_cross_9_21"]   = ema[9]  / (ema[21]  + 1e-9) - 1
    df["ema_cross_21_50"]  = ema[21] / (ema[50]  + 1e-9) - 1
    df["ema_cross_50_200"] = ema[50] / (ema[200] + 1e-9) - 1

    # EMA cross events (-1, 0, +1)
    def cross_event(a, b):
        above = (a > b).astype(int)
        return above.diff().clip(-1, 1)

    df["ema_9_21_cross"]   = cross_event(ema[9],  ema[21])
    df["ema_21_50_cross"]  = cross_event(ema[21], ema[50])

    # MACD (normalized with ATR)
    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["macd_norm"]   = df["macd_diff"] / (df.get("atr", c.rolling(14).std()) + 1e-9)
    df["macd_cross"]  = cross_event(df["macd"], df["macd_signal"])

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(df["high"], df["low"], c)
    df["psar"]        = psar.psar()
    df["psar_signal"] = (c > df["psar"]).astype(int)
    # psar_dist will be calculated after ATR

    # TRIX
    df["trix"] = ta.trend.trix(c, window=15)

    # Aroon
    aroon = ta.trend.AroonIndicator(df["high"], df["low"], window=25)
    df["aroon_up"]   = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    df["aroon_osc"]  = aroon.aroon_indicator()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    for p in [2, 7, 14, 21]:
        df[f"rsi_{p}"] = ta.momentum.rsi(c, window=p)

    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
    # RSI momentum (change over 5 bars)
    df["rsi_slope_5"]    = df["rsi_14"] - df["rsi_14"].shift(5)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"]    = stoch.stoch()
    df["stoch_d"]    = stoch.stoch_signal()
    df["stoch_kd"]   = df["stoch_k"] - df["stoch_d"]

    df["cci"]        = ta.trend.cci(h, l, c, window=20)
    df["williams_r"] = ta.momentum.williams_r(h, l, c, lbp=14)
    df["mfi"]        = ta.volume.money_flow_index(h, l, c, v, window=14)

    for p in [5, 10, 20]:
        df[f"roc_{p}"] = ta.momentum.roc(c, window=p)

    # Chande Momentum Oscillator
    delta  = c.diff()
    up14   = delta.clip(lower=0).rolling(14).sum()
    dn14   = (-delta.clip(upper=0)).rolling(14).sum()
    df["cmo"] = 100 * (up14 - dn14) / (up14 + dn14 + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOLATILITY + REGIME
# ─────────────────────────────────────────────────────────────────────────────

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    # ATR
    df["atr"]     = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_pct"] = df["atr"] / (c + 1e-9)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]      = bb.bollinger_hband()
    df["bb_lower"]      = bb.bollinger_lband()
    df["bb_mid"]        = bb.bollinger_mavg()
    df["bb_width"]      = bb.bollinger_wband()
    df["bb_pct"]        = bb.bollinger_pband()
    df["bb_width_norm"] = df["bb_width"] / (c + 1e-9)

    # Keltner Channel
    keltner = ta.volatility.KeltnerChannel(h, l, c, window=20)
    df["keltner_upper"] = keltner.keltner_channel_hband()
    df["keltner_lower"] = keltner.keltner_channel_lband()
    keltner_width       = df["keltner_upper"] - df["keltner_lower"]

    # Bollinger-Keltner Squeeze (volatility breakout predictor)
    df["bb_keltner_squeeze"]  = (df["bb_width_norm"] < keltner_width / (c + 1e-9)).astype(int)
    df["squeeze_intensity"]   = 1 - (df["bb_width"] / (keltner_width + 1e-9))

    # Donchian
    df["donchian_high"] = h.rolling(20).max()
    df["donchian_low"]  = l.rolling(20).min()
    df["donchian_pct"]  = (c - df["donchian_low"]) / (df["donchian_high"] - df["donchian_low"] + 1e-9)

    # ATR ratios (short/long volatility)
    atr50 = ta.volatility.average_true_range(h, l, c, window=50)
    df["atr_ratio_10"] = df["atr"] / (ta.volatility.average_true_range(h, l, c, window=10) + 1e-9)
    df["atr_ratio_30"] = df["atr"] / (atr50 + 1e-9)
    df["high_vol_regime"] = (df["atr_ratio_30"] > 1.5).astype(int)
    df["low_vol_regime"]  = (df["atr_ratio_30"] < 0.7).astype(int)

    # Historical Volatility (annualized)
    log_ret = np.log(c / c.shift(1))
    df["hv_10"] = log_ret.rolling(10).std() * np.sqrt(365 * 24)
    df["hv_30"] = log_ret.rolling(30).std() * np.sqrt(365 * 24)
    df["hv_ratio"] = df["hv_10"] / (df["hv_30"] + 1e-9)
    df["vol_accel"] = df["hv_10"].diff(5)

    # ADX + DI (trend strength)
    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]      = adx_ind.adx()
    df["adx_pos"]  = adx_ind.adx_pos()
    df["adx_neg"]  = adx_ind.adx_neg()
    df["di_cross"] = df["adx_pos"] - df["adx_neg"]
    df["adx_slope"]= df["adx"].diff(5)
    df["regime_trending"]      = (df["adx"] > 25).astype(int)
    df["regime_strong_trend"]  = (df["adx"] > 40).astype(int)
    df["regime_ranging"]       = (df["adx"] < 20).astype(int)

    # Choppiness Index
    h14 = h.rolling(14).max()
    l14 = l.rolling(14).min()
    df["chop"] = 100 * np.log10(df["atr"].rolling(14).sum() / (h14 - l14 + 1e-9)) / np.log10(14)

    # Z-score
    df["return_zscore"] = (
        c.pct_change(fill_method=None) - c.pct_change(fill_method=None).rolling(20).mean()
    ) / (c.pct_change(fill_method=None).rolling(20).std() + 1e-9)

    df["zscore_50"]  = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-9)
    df["zscore_200"] = (c - c.rolling(200).mean()) / (c.rolling(200).std() + 1e-9)

    df["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-9)
    df["range_pos_50"] = (c - l.rolling(50).min()) / (h.rolling(50).max() - l.rolling(50).min() + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. VOLUME & MICROSTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Relative volume
    for p in [5, 10, 20, 50]:
        df[f"rel_vol_{p}"] = v / (v.rolling(p).mean() + 1e-9)
    df["vol_trend"] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)

    # OBV
    df["obv"]       = ta.volume.on_balance_volume(c, v)
    df["obv_slope"] = df["obv"].diff(5) / (df["atr"] * v.mean() + 1e-9)

    # VWAP
    df["vwap"]          = ta.volume.volume_weighted_average_price(h, l, c, v, window=14)
    df["vwap_deviation"] = (c - df["vwap"]) / (df["atr"] + 1e-9)
    df["above_vwap"]    = (c > df["vwap"]).astype(int)
    df["vwap_trend"]    = df["vwap"].pct_change(5, fill_method=None)

    # CMF
    df["cmf"]     = ta.volume.chaikin_money_flow(h, l, c, v, window=20)
    df["cmf_sma"] = df["cmf"].rolling(3).mean()

    # Force Index
    df["force_index"] = ta.volume.force_index(c, v, window=13)

    # Buy/Sell volume (Lee-Ready approach)
    df["buy_vol"]       = v * (c - l) / (h - l + 1e-9)
    df["sell_vol"]      = v * (h - c) / (h - l + 1e-9)
    df["buy_sell_ratio"]= df["buy_vol"] / (df["sell_vol"] + 1e-9)
    df["buy_pressure"]  = df["buy_vol"].rolling(14).mean() / (v.rolling(14).mean() + 1e-9)

    # Amihud illiquidity ratio
    df["amihud"] = c.pct_change(fill_method=None).abs() / (v * c + 1e-9)
    df["amihud_norm"] = df["amihud"] / (df["amihud"].rolling(20).mean() + 1e-9)

    # Price efficiency ratio (how directional?)
    for p in [5, 10, 20]:
        directional = c.diff(p).abs()
        path        = c.diff().abs().rolling(p).sum()
        df[f"efficiency_{p}"] = directional / (path + 1e-9)

    # Volume Z-score
    df["vol_zscore"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRICE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    rng = h - l + 1e-9

    df["high_low_pct"]   = (h - l) / (l + 1e-9)
    df["close_open_pct"] = (c - o) / (o + 1e-9)
    df["upper_shadow"]   = (h - c.clip(lower=o)) / rng
    df["lower_shadow"]   = (c.clip(upper=o) - l) / rng

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"return_{lag}"] = c.pct_change(lag, fill_method=None)

    for p in [5, 10, 20, 50]:
        df[f"vol_{p}"]      = df["return_1"].rolling(p).std()
        df[f"vol_skew_{p}"] = df["return_1"].rolling(p).skew()

    # Psar normalize (ATR is now available)
    df["psar_dist"] = (c - df["psar"]) / (df["atr"] + 1e-9)

    # Support/Resistance distances (ATR normalized)
    df["resistance_20"] = h.rolling(20).max()
    df["support_20"]    = l.rolling(20).min()
    df["dist_to_res"]   = (df["resistance_20"] - c) / (df["atr"] + 1e-9)
    df["dist_to_sup"]   = (c - df["support_20"]) / (df["atr"] + 1e-9)
    df["sr_position"]   = (c - df["support_20"]) / (df["resistance_20"] - df["support_20"] + 1e-9)

    # EMA distances (ATR normalized)
    for p in [9, 21, 50, 200]:
        ema_col = f"ema{p}_ratio"
        if ema_col in df.columns:
            df[f"ema{p}_atr_dist"] = df[ema_col] / (df["atr_pct"] + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ICHIMOKU (no look-ahead, ATR normalized)
# ─────────────────────────────────────────────────────────────────────────────

def _ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    atr = df["atr"]

    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2               # no shift for ML
    span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2

    df["ich_tenkan"] = tenkan
    df["ich_kijun"]  = kijun
    df["ich_span_a"] = span_a
    df["ich_span_b"] = span_b

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

    df["ich_above_cloud"] = (c > cloud_top).astype(int)
    df["ich_below_cloud"] = (c < cloud_bot).astype(int)
    df["ich_in_cloud"]    = ((c >= cloud_bot) & (c <= cloud_top)).astype(int)
    df["ich_bull_cloud"]  = (span_a > span_b).astype(int)

    df["ich_dist_cloud_top"] = (c - cloud_top) / (atr + 1e-9)
    df["ich_dist_cloud_bot"] = (c - cloud_bot) / (atr + 1e-9)
    df["ich_cloud_thick"]    = (cloud_top - cloud_bot) / (atr + 1e-9)

    df["ich_kijun_dist"]  = (c - kijun) / (atr + 1e-9)
    df["ich_tenkan_dist"] = (c - tenkan) / (atr + 1e-9)

    df["ich_tk_cross"]       = np.sign(tenkan - kijun)
    df["ich_tk_cross_event"] = df["ich_tk_cross"].diff().clip(-1, 1)

    # Triple confirmation signal
    df["ich_strong_bull"] = (
        (df["ich_above_cloud"] == 1) &
        (df["ich_tk_cross"] == 1) &
        (df["ich_bull_cloud"] == 1)
    ).astype(int)
    df["ich_strong_bear"] = (
        (df["ich_below_cloud"] == 1) &
        (df["ich_tk_cross"] == -1) &
        (df["ich_bull_cloud"] == 0)
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. CANDLESTICK PATTERNS (with trend context)
# ─────────────────────────────────────────────────────────────────────────────

def _candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body     = c - o
    body_abs = body.abs()
    rng      = h - l + 1e-9
    up_wick  = h - c.clip(lower=o)
    dn_wick  = c.clip(upper=o) - l
    avg_body = body_abs.rolling(14).mean()

    # Single candle
    df["pat_doji"]            = (body_abs < 0.05 * rng).astype(int)
    df["pat_gravestone_doji"] = (
        (body_abs < 0.1 * rng) & (up_wick > 0.6 * rng) & (dn_wick < 0.1 * rng)
    ).astype(int)
    df["pat_dragonfly_doji"]  = (
        (body_abs < 0.1 * rng) & (dn_wick > 0.6 * rng) & (up_wick < 0.1 * rng)
    ).astype(int)

    # Hammer / Shooting Star (trend context)
    sma10 = c.rolling(10).mean()
    df["pat_hammer"] = (
        (dn_wick >= 2 * body_abs) & (up_wick <= 0.1 * rng) & (body_abs > 0) & (c < sma10)
    ).astype(int)
    df["pat_shooting_star"] = (
        (up_wick >= 2 * body_abs) & (dn_wick <= 0.1 * rng) & (body_abs > 0) & (c > sma10)
    ).astype(int)
    df["pat_spinning_top"]  = (
        (body_abs < 0.3 * rng) & (up_wick > 0.2 * rng) & (dn_wick > 0.2 * rng)
    ).astype(int)

    df["pat_bull_marubozu"] = (
        (body > 0) & (up_wick < 0.05 * rng) & (dn_wick < 0.05 * rng) & (body_abs > avg_body)
    ).astype(int)
    df["pat_bear_marubozu"] = (
        (body < 0) & (up_wick < 0.05 * rng) & (dn_wick < 0.05 * rng) & (body_abs > avg_body)
    ).astype(int)

    # Two candles
    pb, pba = body.shift(1), body_abs.shift(1)
    po, pc  = o.shift(1), c.shift(1)

    df["pat_bull_engulf"] = (
        (pb < 0) & (body > 0) & (o < pc) & (c > po) & (body_abs > pba)
    ).astype(int)
    df["pat_bear_engulf"] = (
        (pb > 0) & (body < 0) & (o > pc) & (c < po) & (body_abs > pba)
    ).astype(int)
    df["pat_bull_harami"] = (
        (pb < 0) & (body > 0) & (o > pc) & (c < po)
    ).astype(int)
    df["pat_bear_harami"] = (
        (pb > 0) & (body < 0) & (o < pc) & (c > po)
    ).astype(int)

    # Three candles
    pb2, pba2 = body.shift(2), body_abs.shift(2)
    po2, pc2  = o.shift(2), c.shift(2)

    df["pat_morning_star"] = (
        (pb2 < -avg_body.shift(2)) &
        (body_abs.shift(1) < 0.3 * avg_body.shift(1)) &
        (body > avg_body) &
        (c > (po2 + pc2) / 2)
    ).astype(int)
    df["pat_evening_star"] = (
        (pb2 > avg_body.shift(2)) &
        (body_abs.shift(1) < 0.3 * avg_body.shift(1)) &
        (body < -avg_body) &
        (c < (po2 + pc2) / 2)
    ).astype(int)
    df["pat_three_white_soldiers"] = (
        (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
        (o > o.shift(1)) & (c > c.shift(1)) &
        (body_abs > 0.5 * avg_body)
    ).astype(int)
    df["pat_three_black_crows"] = (
        (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
        (o < o.shift(1)) & (c < c.shift(1)) &
        (body_abs > 0.5 * avg_body)
    ).astype(int)
    df["pat_dark_cloud"] = (
        (pb > avg_body.shift(1)) & (body < 0) &
        (o > h.shift(1)) & (c < (po + pc) / 2)
    ).astype(int)
    df["pat_piercing_line"] = (
        (pb < -avg_body.shift(1)) & (body > 0) &
        (o < l.shift(1)) & (c > (po + pc) / 2)
    ).astype(int)

    # Composite scores
    bull_pats = ["pat_hammer", "pat_dragonfly_doji", "pat_bull_engulf",
                 "pat_bull_harami", "pat_morning_star", "pat_three_white_soldiers",
                 "pat_piercing_line", "pat_bull_marubozu"]
    bear_pats = ["pat_shooting_star", "pat_gravestone_doji", "pat_bear_engulf",
                 "pat_bear_harami", "pat_evening_star", "pat_three_black_crows",
                 "pat_dark_cloud", "pat_bear_marubozu"]

    df["pat_bull_score"] = df[bull_pats].sum(axis=1)
    df["pat_bear_score"] = df[bear_pats].sum(axis=1)
    df["pat_net_score"]  = df["pat_bull_score"] - df["pat_bear_score"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. TIME FEATURES (cyclical encoding)
# ─────────────────────────────────────────────────────────────────────────────

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        dow  = df.index.dayofweek
        df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"]= (dow >= 5).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. MULTI-TIMEFRAME (1H → 4H + 1D resample)
# ─────────────────────────────────────────────────────────────────────────────

def _multi_timeframe(df_1h: pd.DataFrame) -> pd.DataFrame:
    ohlcv_agg = {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}

    df = df_1h.copy()

    for tf_label, tf_rule in [("4h", "4h"), ("1d", "1D")]:
        try:
            # Only resample the OHLCV columns (do not include TA columns)
            tf = df_1h[["open","high","low","close","volume"]].resample(tf_rule).agg(ohlcv_agg).dropna()
            c_tf, h_tf, l_tf = tf["close"], tf["high"], tf["low"]
            atr_tf = ta.volatility.average_true_range(h_tf, l_tf, c_tf, 14)

            ema9  = ta.trend.ema_indicator(c_tf, 9)
            ema21 = ta.trend.ema_indicator(c_tf, 21)
            ema50 = ta.trend.ema_indicator(c_tf, 50)
            tf[f"ema9r"]         = c_tf / (ema9  + 1e-9) - 1
            tf[f"ema_cross_9_21"]= ema9 / (ema21 + 1e-9) - 1
            tf[f"ema_cross_21_50"]= ema21 / (ema50 + 1e-9) - 1

            tf[f"rsi_14"]  = ta.momentum.rsi(c_tf, 14)
            tf[f"rsi_norm"]= tf[f"rsi_14"] / 50 - 1

            macd_tf = ta.trend.MACD(c_tf)
            tf[f"macd_norm"] = macd_tf.macd_diff() / (atr_tf + 1e-9)

            adx_tf = ta.trend.ADXIndicator(h_tf, l_tf, c_tf, 14)
            tf[f"adx"]      = adx_tf.adx()
            tf[f"di_cross"] = adx_tf.adx_pos() - adx_tf.adx_neg()

            bb_tf = ta.volatility.BollingerBands(c_tf, 20, 2)
            tf[f"bb_pct"]  = bb_tf.bollinger_pband()
            tf[f"atr_pct"] = atr_tf / (c_tf + 1e-9)

            tf[f"range_pos_20"] = (
                (c_tf - l_tf.rolling(20).min()) /
                (h_tf.rolling(20).max() - l_tf.rolling(20).min() + 1e-9)
            )

            feat_cols = [c for c in tf.columns if c not in ["open","high","low","close","volume"]]
            rename = {c: f"{tf_label}_{c}" for c in feat_cols}
            tf = tf.rename(columns=rename)
            tf_feats = tf[[f"{tf_label}_{c}" for c in feat_cols]]

            # forward-fill: apply the higher TF bar to all 1h bars beneath it
            tf_aligned = tf_feats.reindex(df.index, method="ffill")
            df = pd.concat([df, tf_aligned], axis=1)
        except Exception as e:
            print(f"MTF {tf_label} hatası: {e}")

    # Multi-TF confluence
    try:
        bull_align = (
            (df.get("ema_cross_9_21", 0) > 0) &
            (df.get("4h_ema_cross_9_21", 0) > 0) &
            (df.get("1d_ema_cross_9_21", 0) > 0)
        )
        bear_align = (
            (df.get("ema_cross_9_21", 0) < 0) &
            (df.get("4h_ema_cross_9_21", 0) < 0) &
            (df.get("1d_ema_cross_9_21", 0) < 0)
        )
        df["mtf_bull_align"]   = bull_align.astype(int)
        df["mtf_bear_align"]   = bear_align.astype(int)
        df["mtf_trend_score"]  = (
            np.sign(df.get("ema_cross_9_21", pd.Series(0, index=df.index))) +
            np.sign(df.get("4h_ema_cross_9_21", pd.Series(0, index=df.index))) +
            np.sign(df.get("1d_ema_cross_9_21", pd.Series(0, index=df.index)))
        )
    except Exception:
        pass

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Order matters: ATR must be computed first (others depend on it)
    df = _trend_features(df)         # EMA/SMA ratios, MACD, PSAR, Aroon
    df = _volatility_features(df)   # ATR, BB, ADX, HV, regime
    df = _momentum_features(df)     # RSI, Stochastic, CCI, CMO…
    df = _volume_features(df)       # OBV, VWAP, CMF, buy/sell vol…
    df = _price_features(df)        # returns, shadows, S/R distances
    df = _ichimoku_features(df)     # Ichimoku (ATR normalized)
    df = _candlestick_patterns(df)  # 15+ patterns
    df = _time_features(df)         # hour/day cyclical
    df = _divergence_features(df)   # RSI/MACD-price divergences
    df = _regime_features(df)       # Volatility regime, GARCH, tail risk
    df = _order_flow_features(df)   # Order Flow Imbalance
    df = _lag_features(df)          # Lagged returns and RSI lags
    df = _signal_quality_features(df)  # Confluent signal quality
    df = _futures_features(df)      # Funding Rate / OI / Fear&Greed (if present)

    # Multi-timeframe only works with a timezone-aware DatetimeIndex
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = _multi_timeframe(df)

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop NaN rows in core columns (exclude MTF + large-window NaNs)
    optional_prefixes = ("4h_", "1d_", "mtf_", "fr_", "oi_", "fng_",
                         "cum_ret_", "return_lag_", "rsi_lag_",
                         "regime_", "garch_", "ret_skew", "ret_kurt",
                         "crash_", "dist_from_", "hurst", "autocorr",
                         "regime_trend", "regime_range",
                         "confluence_", "high_confluence",
                         "tb_tp", "tb_sl")
    # Large-window (200-bar) columns are also optional
    optional_contains = ("200", "zscore_200")
    optional_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in optional_prefixes)
        or any(s in c for s in optional_contains)
    ]
    core_cols = [c for c in df.columns if c not in optional_cols]
    df = df.dropna(subset=core_cols)
    # Forward-fill optional columns → remaining NaNs become 0
    if optional_cols:
        df[optional_cols] = df[optional_cols].ffill().fillna(0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW FEATURE BLOCKS (based on research findings)
# ─────────────────────────────────────────────────────────────────────────────

def _divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/MACD-price divergences — far more predictive than the raw indicator."""
    c = df["close"]
    window = 14

    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        # Bullish divergence: lower price low, higher RSI low
        df["bull_divergence"] = (
            (c < c.shift(window)) & (rsi > rsi.shift(window))
        ).astype(int)
        # Bearish divergence: higher price high, lower RSI high
        df["bear_divergence"] = (
            (c > c.shift(window)) & (rsi < rsi.shift(window))
        ).astype(int)
        # Hidden bullish: trend continuation signal (higher price low, lower RSI low)
        df["hidden_bull_div"] = (
            (c > c.shift(window)) & (rsi < rsi.shift(window))
        ).astype(int)
        # RSI momentum
        df["rsi_slope_5"] = rsi.diff(5) / 5
        df["rsi_slope_10"] = rsi.diff(10) / 10

    if "macd_diff" in df.columns:
        macd = df["macd_diff"]
        df["macd_bull_div"] = (
            (c < c.shift(window)) & (macd > macd.shift(window))
        ).astype(int)
        df["macd_bear_div"] = (
            (c > c.shift(window)) & (macd < macd.shift(window))
        ).astype(int)
        df["macd_diff_slope"] = macd.diff(3)

    # Volume-price divergence: price rising, volume falling = weak move
    if "volume" in df.columns:
        v = df["volume"]
        df["vol_price_div_bull"] = (
            (c.pct_change(5, fill_method=None) > 0) &
            (v.rolling(5).mean() < v.rolling(20).mean())
        ).astype(int)
        df["vol_price_div_bear"] = (
            (c.pct_change(5, fill_method=None) < 0) &
            (v.rolling(5).mean() < v.rolling(20).mean())
        ).astype(int)

    return df


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility regime detection — GARCH-like + Hurst exponent."""
    c = df["close"]
    log_ret = np.log(c / c.shift(1))

    # Realized volatility
    rv5  = log_ret.rolling(5).std() * np.sqrt(24 * 365)
    rv20 = log_ret.rolling(20).std() * np.sqrt(24 * 365)
    rv60 = log_ret.rolling(60).std() * np.sqrt(24 * 365)

    rv_pct = rv20.rolling(100, min_periods=20).apply(
        lambda x: (x < x[-1]).mean(), raw=True
    )
    df["regime_low_vol"]    = (rv_pct < 0.33).astype(int)
    df["regime_medium_vol"] = ((rv_pct >= 0.33) & (rv_pct < 0.67)).astype(int)
    df["regime_high_vol"]   = (rv_pct >= 0.67).astype(int)

    df["vol_expansion"]   = (rv5 > rv20).astype(int)
    df["vol_contraction"] = (rv5 < rv20 * 0.7).astype(int)
    df["vol_ratio_5_20"]  = rv5 / (rv20 + 1e-9)
    df["vol_ratio_5_60"]  = rv5 / (rv60 + 1e-9)

    # GARCH(1,1)-like EWMA volatility
    squared_ret = log_ret ** 2
    garch_var = squared_ret.ewm(alpha=0.1).mean()
    df["garch_vol"] = np.sqrt(garch_var * 24 * 365)
    df["garch_vol_change"] = df["garch_vol"].pct_change(5, fill_method=None)

    # Skewness and kurtosis (tail risk indicator)
    df["ret_skew_20"] = log_ret.rolling(20).skew()
    df["ret_kurt_20"] = log_ret.rolling(20).kurt()
    df["crash_risk"]  = (
        (df["ret_skew_20"] < -0.5) & (df["ret_kurt_20"] > 3)
    ).astype(int)

    # Trend regime combination
    if "adx" in df.columns:
        adx = df["adx"]
        df["regime_trend_low_vol"]  = ((adx > 25) & (rv_pct < 0.4)).astype(int)
        df["regime_trend_high_vol"] = ((adx > 25) & (rv_pct > 0.6)).astype(int)
        df["regime_range_low_vol"]  = ((adx < 20) & (rv_pct < 0.4)).astype(int)

    # Distance to ATH (120 bars = 5 days, compatible with live prediction)
    roll_max = c.rolling(120, min_periods=10).max()
    roll_min = c.rolling(120, min_periods=10).min()
    df["dist_from_high_5d"] = (roll_max - c) / (roll_max + 1e-9)
    df["dist_from_low_5d"]  = (c - roll_min) / (c - roll_min + 1e-9)

    return df


def _order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Order Flow Imbalance — volume-based proxy without tick data."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Classify volume by bar direction
    bar_return = (c - o) / (h - l + 1e-9)
    buy_frac = (bar_return.clip(-1, 1) + 1) / 2
    buy_vol  = v * buy_frac
    sell_vol = v * (1 - buy_frac)

    df["ofi_buy_vol"]  = buy_vol
    df["ofi_sell_vol"] = sell_vol

    for win in [5, 10, 20]:
        buy_sum  = buy_vol.rolling(win).sum()
        sell_sum = sell_vol.rolling(win).sum()
        df[f"ofi_{win}"] = (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-9)

    df["ofi_trend"]        = df["ofi_10"].diff(5)
    df["ofi_acceleration"] = df["ofi_10"].diff(1).diff(1)

    # Aggressive candles above average volume
    avg_vol = v.rolling(20).mean()
    df["aggressive_buy"] = (
        (c > o) & (v > 2 * avg_vol) &
        ((c - o) / (h - l + 1e-9) > 0.6)
    ).astype(int)
    df["aggressive_sell"] = (
        (c < o) & (v > 2 * avg_vol) &
        ((o - c) / (h - l + 1e-9) > 0.6)
    ).astype(int)

    # Ratio of volume occurring above the midpoint
    above_mid = c > (h + l) / 2
    df["vol_above_mid_10"] = (
        (v * above_mid.astype(float)).rolling(10).sum() /
        (v.rolling(10).sum() + 1e-9)
    )

    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged return and RSI features — for momentum and mean-reversion."""
    c = df["close"]
    log_ret = np.log(c / c.shift(1))

    for lag in [1, 2, 3, 4, 6, 12, 24]:
        df[f"return_lag_{lag}"] = log_ret.shift(lag)

    if "rsi_14" in df.columns:
        for lag in [1, 3, 6, 12]:
            df[f"rsi_lag_{lag}"] = df["rsi_14"].shift(lag)

    # Cumulative returns (momentum) — up to 48h for live prediction
    for period in [6, 12, 24, 48]:
        df[f"cum_ret_{period}h"] = c.pct_change(period, fill_method=None)

    return df


def _signal_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Confluent signal quality — how many indicators point the same direction?"""
    bull_signals, bear_signals = [], []

    if "rsi_14" in df.columns:
        bull_signals.append((df["rsi_14"] < 40).astype(int))
        bear_signals.append((df["rsi_14"] > 60).astype(int))

    if "macd_diff" in df.columns:
        bull_signals.append((df["macd_diff"] > 0).astype(int))
        bear_signals.append((df["macd_diff"] < 0).astype(int))

    if "bb_pct" in df.columns:
        bull_signals.append((df["bb_pct"] < 0.2).astype(int))
        bear_signals.append((df["bb_pct"] > 0.8).astype(int))

    if "adx" in df.columns and "macd_diff" in df.columns:
        adx = df["adx"]
        bull_signals.append(
            ((adx > 20) & (df["macd_diff"] > 0)).astype(int)
        )
        bear_signals.append(
            ((adx > 20) & (df["macd_diff"] < 0)).astype(int)
        )

    if "ich_strong_bull" in df.columns:
        bull_signals.append(df["ich_strong_bull"].astype(int))
    if "ich_strong_bear" in df.columns:
        bear_signals.append(df["ich_strong_bear"].astype(int))

    if bull_signals:
        n = len(bull_signals)
        df["confluence_bull"] = sum(bull_signals) / n
        df["confluence_bear"] = sum(bear_signals) / n
        df["confluence_net"]  = df["confluence_bull"] - df["confluence_bear"]
        df["high_confluence"] = (df["confluence_net"].abs() > 0.5).astype(int)

    return df


def _futures_features(df: pd.DataFrame) -> pd.DataFrame:
    """Funding Rate + Open Interest features (using data from the collector)."""
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].ffill().fillna(0)
        df["fr_8h"]           = fr
        df["fr_ma_3d"]        = fr.rolling(9, min_periods=1).mean()
        df["fr_ma_7d"]        = fr.rolling(21, min_periods=1).mean()
        fr_mean = fr.rolling(30, min_periods=5).mean()
        fr_std  = fr.rolling(30, min_periods=5).std() + 1e-9
        df["fr_zscore"]       = (fr - fr_mean) / fr_std
        df["fr_extreme_bull"] = (fr > fr.rolling(30, min_periods=5).quantile(0.9)).astype(int)
        df["fr_extreme_bear"] = (fr < fr.rolling(30, min_periods=5).quantile(0.1)).astype(int)
        df["fr_cumsum_3d"]    = fr.rolling(9, min_periods=1).sum()
        df["fr_direction_change"] = np.sign(fr).diff().abs()

    if "open_interest" in df.columns:
        oi = df["open_interest"].ffill().fillna(method=None)
        if oi.notna().sum() > 10:
            df["oi_change_1h"]  = oi.pct_change(1, fill_method=None)
            df["oi_change_4h"]  = oi.pct_change(4, fill_method=None)
            df["oi_change_24h"] = oi.pct_change(24, fill_method=None)
            oi_mean = oi.rolling(24, min_periods=5).mean()
            df["oi_ma_ratio"]   = oi / (oi_mean + 1e-9) - 1
            oi_std = oi.rolling(48, min_periods=10).std() + 1e-9
            df["oi_zscore"]     = (oi - oi.rolling(48, min_periods=10).mean()) / oi_std

            price_dir = np.sign(df["close"].pct_change(4, fill_method=None))
            oi_dir    = np.sign(df["oi_change_4h"])
            df["oi_price_agree"]        = (price_dir == oi_dir).astype(int)
            df["oi_bearish_div"]        = ((price_dir > 0) & (oi_dir < 0)).astype(int)
            df["oi_bull_liq_risk"]      = ((price_dir < 0) & (oi_dir > 0)).astype(int)

    if "fng_value" in df.columns:
        fng = df["fng_value"].ffill().fillna(0.5)
        df["fng_extreme_fear"]  = (fng < 0.25).astype(int)
        df["fng_extreme_greed"] = (fng > 0.75).astype(int)
        df["fng_change_7d"]     = fng.diff(7 * 24)
        df["fng_ma_14d"]        = fng.rolling(14 * 24, min_periods=24).mean()
        df["fng_above_ma"]      = (fng > df["fng_ma_14d"]).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABELING (dynamic ATR-based threshold)
# ─────────────────────────────────────────────────────────────────────────────

def create_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    use_dynamic_threshold: bool = True,
    fixed_threshold: float = 0.015,
    threshold_mult: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1

    if use_dynamic_threshold and "atr_pct" in df.columns:
        threshold = df["atr_pct"] * threshold_mult   # threshold_mult × ATR
    else:
        threshold = fixed_threshold * threshold_mult

    df["label"] = 1
    df.loc[future_return > threshold,  "label"] = 2   # BUY
    df.loc[future_return < -threshold, "label"] = 0   # SELL

    df["future_return"] = future_return
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE BARRIER LABELING (Lopez de Prado — AFML 2018)
# ─────────────────────────────────────────────────────────────────────────────

def create_labels_triple_barrier(
    df: pd.DataFrame,
    horizon: int = 24,
    tp_multiplier: float = 1.5,
    sl_multiplier: float = 1.0,
    min_atr_floor: float = 0.003,
) -> pd.DataFrame:
    """
    Triple Barrier Labeling:
    - Upper barrier (TP): close + ATR × tp_multiplier
    - Lower barrier (SL): close - ATR × sl_multiplier
    - Vertical barrier: after horizon bars (time limit)

    Whichever is hit first determines the label:
      TP first → BUY (2)
      SL first → SELL (0)
      Time's up → HOLD (1)

    Advantage: path-dependent, risk-aware, filters out noise.
    """
    df = df.copy()
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(df)

    atr_col = "atr" if "atr" in df.columns else None
    if atr_col:
        atr_vals = df["atr"].values
    else:
        log_ret = np.log(close[1:] / np.where(close[:-1] != 0, close[:-1], 1e-9))
        rolling_std = pd.Series(log_ret).rolling(14).std().fillna(0.01).values
        atr_vals = np.concatenate([[rolling_std[0]], rolling_std]) * close

    labels = np.ones(n, dtype=int)  # default: HOLD

    for i in range(n - horizon - 1):
        price_i = close[i]
        atr_i   = max(atr_vals[i], price_i * min_atr_floor)

        tp_price = price_i + tp_multiplier * atr_i
        sl_price = price_i - sl_multiplier * atr_i

        first_tp, first_sl = None, None

        for j in range(i + 1, min(i + horizon + 1, n)):
            if first_tp is None and high[j] >= tp_price:
                first_tp = j
            if first_sl is None and low[j] <= sl_price:
                first_sl = j
            if first_tp is not None and first_sl is not None:
                break

        if first_tp is None and first_sl is None:
            labels[i] = 1  # HOLD
        elif first_tp is None:
            labels[i] = 0  # SELL (SL hit first)
        elif first_sl is None:
            labels[i] = 2  # BUY (TP hit first)
        else:
            labels[i] = 2 if first_tp <= first_sl else 0

    labels[-horizon:] = 1  # Last horizon bars → HOLD

    df["label"] = labels

    dist = pd.Series(labels).value_counts().sort_index()
    print(f"  Triple Barrier | SELL:{dist.get(0,0)} "
          f"HOLD:{dist.get(1,0)} BUY:{dist.get(2,0)}")

    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_COLS = {
    "open", "high", "low", "close", "volume",
    "label", "future_return",
    # raw price levels (normalized versions exist)
    "ema9", "ema21", "ema50", "ema200",
    "sma9", "sma21", "sma50", "sma200",
    "bb_upper", "bb_lower", "bb_mid",
    "keltner_upper", "keltner_lower",
    "donchian_high", "donchian_low",
    "resistance_20", "support_20",
    "ich_tenkan", "ich_kijun", "ich_span_a", "ich_span_b",
    "psar", "vwap", "obv",
    "buy_vol", "sell_vol", "amihud",
    # OFI raw volumes (ratio versions exist)
    "ofi_buy_vol", "ofi_sell_vol",
    # raw funding/OI (derived versions exist)
    "funding_rate", "open_interest", "fng_value",
    # triple barrier meta
    "tb_tp_price", "tb_sl_price",
}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and not c.startswith("_")
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"]
    print(f"Feature sayısı: {X.shape[1]}")
    return X, y
