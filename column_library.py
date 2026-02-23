"""
===========================================================================
# ENTRY-ONLY COLUMN LIBRARY — 67 core + 103 indicators + 25 cruncher (optional)
===========================================================================
Portable Column Library (single-file, reusable across backtesting projects)

Dependencies:
    pip install pandas numpy pandas_ta

Usage example:
    import pandas as pd
    from column_library import (
        add_all_columns,
        add_all_missing_indicators,
        add_final_22_missing_columns,
        add_cruncher_context_columns,
        add_continuous_tracking,
        get_minute_by_minute_tracking,
        add_volatility_columns,
        add_volume_vwap_columns,
        get_all_column_names,
        get_column_groups,
        CONTINUOUS_TRACKING_COLUMNS,
    )

    # 1) Add everything (all categories)
    df = pd.read_csv("ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    df = add_all_columns(df)

    # 2) Add only specific categories (e.g., Volatility + VWAP/Volume)
    df2 = pd.read_csv("ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    df2 = add_volatility_columns(df2)
    df2 = add_volume_vwap_columns(df2)

    # 3) Pull values at an entry bar for filter/cruncher phase
    entry_ts = pd.Timestamp("2026-01-15 10:05:00")
    entry_row = df.loc[entry_ts]
    feature_cols = get_all_column_names()
    entry_features = entry_row[feature_cols]
    print(entry_features.head(10))

Data assumptions:
    - Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
    - Index: sorted pandas.DatetimeIndex (daily or intraday)
    - All calculations are vectorized and avoid lookahead.
"""

# pip install pandas numpy pandas_ta
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta


# ====================== COLUMN REGISTRY ======================
_COLUMN_GROUPS: Dict[str, List[str]] = {
    "volatility": [
        "Col_ATR14",
        "Col_NormalizedATR_Pct",
        "Col_ATR_vs_20dayAvg_Pct",
        "Col_ATR14_vs_5dayAvg_Pct",
        "Col_HistoricalVol_20day",
        "Col_TrueRange_vs_ATR",
        "Col_DailyRange_vs_ATR_Pct",
        "Col_RangeExpansionToday_Pct",
        "Col_VolatilityRatio_20_5",
    ],
    "trend_momentum": [
        "Col_DistTo50MA_ATR",
        "Col_DistTo200MA_ATR",
        "Col_PriceAbove200MA_ATR",
        "Col_ADX14",
        "Col_DI_Diff",
        "Col_MACD_Hist",
        "Col_MACDV_Normalized",
        "Col_ROC10",
        "Col_ROC20",
        "Col_20dayLinReg_Slope_ATR",
        "Col_ExtensionFromOpen_ATR",
    ],
    "oscillators": [
        "Col_RSI14",
        "Col_StochK_14_3",
        "Col_StochD",
        "Col_CCI20",
        "Col_BollingerPctB",
        "Col_DistUpperBB_ATR",
        "Col_DistLowerBB_ATR",
    ],
    "volume_vwap": [
        "Col_RelativeVolume",
        "Col_RelativeVolume_5min",
        "Col_TodayVol_vs_YestVol",
        "Col_VolumeSurge_15min_Pct",
        "Col_CumulativeVol_vs_Avg_Pct",
        "Col_OBV_Slope5",
        "Col_AccumDist",
        "Col_VWAP_Deviation_ATR",
        "Col_VWAP_Deviation_Pct",
        "Col_VWAP_Slope10_ATR",
        "Col_VWAP_ROC5",
        "Col_BarsSinceVWAP_Cross",
        "Col_VWAP_vs_Open_ATR",
        "Col_VWAP_PosIn2SD_Bands_Pct",
        "Col_PreMarketVolume_Ratio",
        "Col_AvgTradeSize_Ratio",
        "Col_TradeCount_5min",
    ],
    "price_action": [
        "Col_PctInYesterdayRange",
        "Col_PctIn5DayRange",
        "Col_DistYesterdayHigh_ATR",
        "Col_DistYesterdayLow_ATR",
        "Col_Dist52wHigh_ATR",
        "Col_Dist52wLow_ATR",
        "Col_OpenToClose_Pct_Sofar",
        "Col_CandleBody_vs_ATR",
        "Col_MomentumScore_5min",
        "Col_ConsecutiveUpBars",
        "Col_BodyToRangeRatio",
    ],
    "gaps": [
        "Col_Gap_Pct",
        "Col_Gap_ATR",
        "Col_PreMarketGap_ATR",
        "Col_GapFillProxy_ATR",
        "Col_GapFillProbability_Proxy",
    ],
    "key_levels": [
        "Col_DistNearestRound_ATR",
    ],
    "market_context": [
        "Col_StockVsSPX_TodayPct",
        "Col_RelStrengthVsSector_20d",
        "Col_Beta60d",
        "Col_CorrToSPY_10d",
    ],
    "time": [
        "Col_EntryTime_HourNumeric",
        "Col_DayOfWeek",
        "Col_SessionFlag",
        "Col_MinutesSinceOpen",
    ],
    "final_22": [
        "Col_ALMA_10_6_0.85",
        "Col_FWMA_10",
        "Col_HWMA_10",
        "Col_PWMA_10",
        "Col_ZLMA_10",
        "Col_SuperTrend",
        "Col_SuperTrend_Direction",
        "Col_Keltner_Upper",
        "Col_Keltner_Lower",
        "Col_Keltner_Width_ATR",
        "Col_Donchian_Upper",
        "Col_Donchian_Lower",
        "Col_MassIndex_25",
        "Col_UlcerIndex_14",
        "Col_ChoppinessIndex_14",
        "Col_ChaikinMoneyFlow_20",
        "Col_ElderForceIndex_13",
        "Col_EaseOfMovement_14",
        "Col_NVI",
        "Col_PVI",
        "Col_PSAR",
        "Col_QStick_10",
        "Col_RelativeVigorIndex",
        "Col_RVISignal",
        "Col_SchaffTrendCycle",
    ],
    "cruncher_context": [
        "Col_GapFill_15min_Pct",
        "Col_GapFill_15min_Zscore",
        "Col_ORB_5min_BreakHigh",
        "Col_ORB_5min_BreakLow",
        "Col_ORB_5min_DistHigh_ATR",
        "Col_ORB_15min_BreakHigh",
        "Col_ORB_15min_BreakLow",
        "Col_ORB_15min_DistHigh_ATR",
        "Col_ORB_30min_BreakHigh",
        "Col_ORB_30min_BreakLow",
        "Col_ORB_30min_DistHigh_ATR",
        "Col_ORB_60min_BreakHigh",
        "Col_ORB_60min_BreakLow",
        "Col_ORB_60min_DistHigh_ATR",
        "Col_ExtensionFromDaily9EMA_ATR",
        "Col_ExtensionFromDaily9EMA_Rank",
        "Col_MultiDaySlope_5d",
        "Col_InsideDay",
        "Col_DollarVolume_20dAvg",
        "Col_RelativeVolume_First30min",
        "Col_RelativeVolume_First30min_Rank",
        "Col_VolumeSurge_1min_Ratio",
        "Col_IntradayATR_Ratio",
        "Col_StdDev_Last10Bars_ATR",
        "Col_DistToVWAP_Slope10_ATR",
        "Col_MomentumDivergence_RSI",
        "Col_DistFromSessionVWAP_ATR",
        "Col_GapPct_x_RelVol30",
        "Col_GapPct_x_ExtensionATR",
    ],
}


# ====================== CORE HELPERS ======================
def _validate_ohlcv_df(df: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in ascending order")


def _safe_div(numer: pd.Series, denom: pd.Series | float | int) -> pd.Series:
    denom_series = denom if isinstance(denom, pd.Series) else pd.Series(denom, index=numer.index)
    out = numer / denom_series.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _atr_normalize(df: pd.DataFrame, values: pd.Series, atr_col: str = "Col_ATR14") -> pd.Series:
    return _safe_div(values, df[atr_col])


def _copy_if_needed(df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
    return df if inplace else df.copy()


def _session_labels(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.normalize(), index=idx)


def _is_intraday(idx: pd.DatetimeIndex) -> bool:
    if len(idx) < 2:
        return False
    return idx.normalize().duplicated().any()


def _session_prev_map(series: pd.Series, agg: str) -> pd.Series:
    session = _session_labels(series.index)
    grouped = series.groupby(session)
    if agg == "first":
        daily = grouped.first()
    elif agg == "last":
        daily = grouped.last()
    elif agg == "max":
        daily = grouped.max()
    elif agg == "min":
        daily = grouped.min()
    elif agg == "sum":
        daily = grouped.sum()
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    prev_daily = daily.shift(1)
    return session.map(prev_daily)


def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "Col_ATR14" not in df.columns:
        return add_volatility_columns(df, inplace=True)
    return df


def _pick_external_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)
    return None


def _indicator_col(
    indicator_df: Optional[pd.DataFrame],
    index: pd.Index,
    exact_names: Optional[List[str]] = None,
    starts_with: Optional[List[str]] = None,
) -> pd.Series:
    """
    Pull a series from pandas_ta output while tolerating version-specific column names.
    """
    if indicator_df is None or indicator_df.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    exact_names = exact_names or []
    starts_with = starts_with or []

    for name in exact_names:
        if name in indicator_df.columns:
            return indicator_df[name].astype(float)

    for col in indicator_df.columns:
        if any(str(col).startswith(prefix) for prefix in starts_with):
            return indicator_df[col].astype(float)

    return pd.Series(np.nan, index=index, dtype=float)


def _session_weighted_vwap_std(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    session = _session_labels(df.index)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"]
    pv2 = (typical**2) * df["Volume"]

    cum_v = df["Volume"].groupby(session).cumsum()
    cum_pv = pv.groupby(session).cumsum()
    cum_pv2 = pv2.groupby(session).cumsum()

    vwap = _safe_div(cum_pv, cum_v)
    variance = _safe_div(cum_pv2, cum_v) - (vwap**2)
    variance = variance.clip(lower=0)
    std = np.sqrt(variance)
    return vwap, std


def _bars_since_event(event: pd.Series) -> pd.Series:
    event = event.fillna(False).astype(bool)
    group_id = event.cumsum()
    bars = event.groupby(group_id).cumcount()
    bars[group_id == 0] = np.nan
    return bars


# ====================== VOLATILITY & NORMALIZATION ======================
def add_volatility_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)

    out["Col_ATR14"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)
    out["Col_NormalizedATR_Pct"] = _safe_div(out["Col_ATR14"], out["Close"]) * 100.0
    out["Col_ATR_vs_20dayAvg_Pct"] = _safe_div(out["Col_ATR14"], out["Col_ATR14"].rolling(20).mean()) * 100.0
    out["Col_ATR14_vs_5dayAvg_Pct"] = _safe_div(out["Col_ATR14"], out["Col_ATR14"].rolling(5).mean()) * 100.0

    log_ret = np.log(_safe_div(out["Close"], out["Close"].shift(1)))
    out["Col_HistoricalVol_20day"] = log_ret.rolling(20).std() * np.sqrt(252.0) * 100.0

    true_range = ta.true_range(out["High"], out["Low"], out["Close"])
    out["Col_TrueRange_vs_ATR"] = _safe_div(true_range, out["Col_ATR14"])
    out["Col_DailyRange_vs_ATR_Pct"] = _safe_div(out["High"] - out["Low"], out["Col_ATR14"]) * 100.0
    daily_range = out["High"] - out["Low"]
    roll5_range = out["High"].rolling(5).max() - out["Low"].rolling(5).min()
    out["Col_RangeExpansionToday_Pct"] = _safe_div(daily_range, roll5_range) * 100.0
    out["Col_VolatilityRatio_20_5"] = _safe_div(out["Col_ATR14"], out["Col_ATR14"].rolling(5).mean())
    return out


# ====================== TREND & MOMENTUM ======================
def add_trend_momentum_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    sma50 = ta.sma(out["Close"], length=50)
    sma200 = ta.sma(out["Close"], length=200)
    out["Col_DistTo50MA_ATR"] = _atr_normalize(out, out["Close"] - sma50)
    out["Col_DistTo200MA_ATR"] = _atr_normalize(out, out["Close"] - sma200)
    out["Col_PriceAbove200MA_ATR"] = _atr_normalize(out, out["Close"] - sma200)

    adx_df = ta.adx(out["High"], out["Low"], out["Close"], length=14)
    dmp = _indicator_col(adx_df, out.index, exact_names=["DMP_14"], starts_with=["DMP_"])
    dmn = _indicator_col(adx_df, out.index, exact_names=["DMN_14"], starts_with=["DMN_"])
    out["Col_ADX14"] = _indicator_col(adx_df, out.index, exact_names=["ADX_14"], starts_with=["ADX_"])
    out["Col_DI_Diff"] = dmp - dmn

    macd_df = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    macd_hist = _indicator_col(macd_df, out.index, exact_names=["MACDh_12_26_9"], starts_with=["MACDh_"])
    macd_val = _indicator_col(macd_df, out.index, exact_names=["MACD_12_26_9"], starts_with=["MACD_"])
    out["Col_MACD_Hist"] = macd_hist
    out["Col_MACDV_Normalized"] = _safe_div(macd_val, out["Col_ATR14"])

    out["Col_ROC10"] = ta.roc(out["Close"], length=10)
    out["Col_ROC20"] = ta.roc(out["Close"], length=20)

    linreg_20 = ta.linreg(out["Close"], length=20)
    out["Col_20dayLinReg_Slope_ATR"] = _atr_normalize(out, linreg_20 - linreg_20.shift(1))
    return out


# ====================== OSCILLATORS ======================
def add_oscillator_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    out["Col_RSI14"] = ta.rsi(out["Close"], length=14)

    stoch_df = ta.stoch(out["High"], out["Low"], out["Close"], k=14, d=3, smooth_k=3)
    out["Col_StochK_14_3"] = _indicator_col(
        stoch_df,
        out.index,
        exact_names=["STOCHk_14_3_3"],
        starts_with=["STOCHk_"],
    )
    out["Col_StochD"] = _indicator_col(
        stoch_df,
        out.index,
        exact_names=["STOCHd_14_3_3"],
        starts_with=["STOCHd_"],
    )

    out["Col_CCI20"] = ta.cci(out["High"], out["Low"], out["Close"], length=20)

    bb = ta.bbands(out["Close"], length=20, std=2.0)
    bb_low = _indicator_col(bb, out.index, exact_names=["BBL_20_2.0"], starts_with=["BBL_"])
    bb_mid = _indicator_col(bb, out.index, exact_names=["BBM_20_2.0"], starts_with=["BBM_"])
    bb_up = _indicator_col(bb, out.index, exact_names=["BBU_20_2.0"], starts_with=["BBU_"])

    _ = bb_mid  # explicit for readability in case strategy code later references middle band
    out["Col_BollingerPctB"] = _safe_div(out["Close"] - bb_low, bb_up - bb_low)
    out["Col_DistUpperBB_ATR"] = _atr_normalize(out, out["Close"] - bb_up)
    out["Col_DistLowerBB_ATR"] = _atr_normalize(out, out["Close"] - bb_low)
    return out


# ====================== VOLUME & LIQUIDITY + VWAP ======================
def add_volume_vwap_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    session = _session_labels(out.index)
    intraday = _is_intraday(out.index)
    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0

    out["Col_RelativeVolume"] = _safe_div(out["Volume"], out["Volume"].rolling(20).mean())
    out["Col_RelativeVolume_5min"] = _safe_div(out["Volume"], out["Volume"].rolling(5).mean())

    if intraday:
        today_cum_vol = out["Volume"].groupby(session).cumsum()
        yest_total_vol = _session_prev_map(out["Volume"], agg="sum")
        out["Col_TodayVol_vs_YestVol"] = _safe_div(today_cum_vol, yest_total_vol)
    else:
        out["Col_TodayVol_vs_YestVol"] = _safe_div(out["Volume"], out["Volume"].shift(1))

    out["Col_VolumeSurge_15min_Pct"] = _safe_div(out["Volume"], out["Volume"].rolling(15).mean()) * 100.0
    minutes_idx = out.index.hour * 60 + out.index.minute + 1
    out["Col_CumulativeVol_vs_Avg_Pct"] = _safe_div(out["Volume"].cumsum(), out["Volume"].rolling(390).mean() * minutes_idx) * 100.0
    out["Col_PreMarketVolume_Ratio"] = _safe_div(out["Volume"], out["Volume"].shift(390))
    out["Col_AvgTradeSize_Ratio"] = np.nan  # placeholder - can be filled later if you have trade count data
    out["Col_TradeCount_5min"] = np.nan  # placeholder

    obv = ta.obv(out["Close"], out["Volume"])
    out["Col_OBV_Slope5"] = _safe_div(obv - obv.shift(5), out["Col_ATR14"])
    out["Col_AccumDist"] = ta.ad(out["High"], out["Low"], out["Close"], out["Volume"])

    # Use pandas_ta VWAP anchors as requested.
    daily_vwap = ta.vwap(out["High"], out["Low"], out["Close"], out["Volume"], anchor="D")
    weekly_vwap = ta.vwap(out["High"], out["Low"], out["Close"], out["Volume"], anchor="W")
    if daily_vwap is None:
        daily_vwap = _safe_div((typical * out["Volume"]).cumsum(), out["Volume"].cumsum())
    if weekly_vwap is None:
        weekly_vwap = daily_vwap

    out["Col_VWAP_Deviation_ATR"] = _atr_normalize(out, out["Close"] - daily_vwap)
    out["Col_VWAP_Deviation_Pct"] = _safe_div(out["Close"] - daily_vwap, daily_vwap) * 100.0

    _, daily_vwap_std = _session_weighted_vwap_std(out)
    vwap_p2 = daily_vwap + 2.0 * daily_vwap_std
    vwap_m2 = daily_vwap - 2.0 * daily_vwap_std

    out["Col_VWAP_Slope10_ATR"] = _atr_normalize(out, daily_vwap - daily_vwap.shift(10))
    out["Col_VWAP_ROC5"] = daily_vwap.pct_change(5) * 100.0

    vwap_side = out["Close"] >= daily_vwap
    cross_event = vwap_side.ne(vwap_side.shift(1)).fillna(False)
    out["Col_BarsSinceVWAP_Cross"] = _bars_since_event(cross_event)

    out["Col_VWAP_vs_Open_ATR"] = _atr_normalize(out, daily_vwap - out["Open"])
    out["Col_VWAP_PosIn2SD_Bands_Pct"] = _safe_div(out["Close"] - vwap_m2, vwap_p2 - vwap_m2) * 100.0
    return out


# ====================== PRICE ACTION & RANGE POSITION ======================
def add_price_action_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    session = _session_labels(out.index)
    intraday = _is_intraday(out.index)

    if intraday:
        y_high = _session_prev_map(out["High"], agg="max")
        y_low = _session_prev_map(out["Low"], agg="min")
    else:
        y_high = out["High"].shift(1)
        y_low = out["Low"].shift(1)

    out["Col_PctInYesterdayRange"] = _safe_div(out["Close"] - y_low, y_high - y_low) * 100.0

    roll5_high = out["High"].rolling(5).max()
    roll5_low = out["Low"].rolling(5).min()
    out["Col_PctIn5DayRange"] = _safe_div(out["Close"] - roll5_low, roll5_high - roll5_low) * 100.0

    out["Col_DistYesterdayHigh_ATR"] = _atr_normalize(out, out["Close"] - y_high)
    out["Col_DistYesterdayLow_ATR"] = _atr_normalize(out, out["Close"] - y_low)

    daily_close = out["Close"].groupby(session).last()
    hi_52w = daily_close.rolling(252).max()
    lo_52w = daily_close.rolling(252).min()
    mapped_hi_52w = session.map(hi_52w)
    mapped_lo_52w = session.map(lo_52w)
    out["Col_Dist52wHigh_ATR"] = _atr_normalize(out, out["Close"] - mapped_hi_52w)
    out["Col_Dist52wLow_ATR"] = _atr_normalize(out, out["Close"] - mapped_lo_52w)

    out["Col_OpenToClose_Pct_Sofar"] = _safe_div(out["Close"] - out["Open"], out["Open"]) * 100.0
    out["Col_CandleBody_vs_ATR"] = _safe_div((out["Close"] - out["Open"]).abs(), out["Col_ATR14"])
    out["Col_ExtensionFromOpen_ATR"] = _atr_normalize(out, out["Close"] - out["Open"])
    out["Col_BodyToRangeRatio"] = (out["Close"] - out["Open"]).abs() / (out["High"] - out["Low"] + 1e-8)
    up_bars = (out["Close"] > out["Open"]).astype(int)
    out["Col_ConsecutiveUpBars"] = up_bars.groupby((up_bars.diff().ne(0).cumsum())).cumcount()
    out["Col_MomentumScore_5min"] = _atr_normalize(out, out["Close"] - out["Open"].shift(5))
    return out


# ====================== GAPS & OVERNIGHT MOVES ======================
def add_gaps_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    intraday = _is_intraday(out.index)
    prev_close = _session_prev_map(out["Close"], agg="last") if intraday else out["Close"].shift(1)
    day_open = out["Open"].groupby(_session_labels(out.index)).transform("first") if intraday else out["Open"]

    out["Col_Gap_Pct"] = _safe_div(day_open - prev_close, prev_close) * 100.0
    out["Col_Gap_ATR"] = _atr_normalize(out, day_open - prev_close)
    out["Col_PreMarketGap_ATR"] = _atr_normalize(out, day_open - prev_close)
    out["Col_GapFillProxy_ATR"] = _atr_normalize(out, out["Close"] - prev_close)
    out["Col_GapFillProbability_Proxy"] = 1.0 - _safe_div((day_open - prev_close).abs(), out["Col_ATR14"])
    return out


# ====================== DISTANCE FROM KEY LEVELS ======================
def add_key_levels_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)
    out = _ensure_atr(out)

    close = out["Close"]
    increments = np.select(
        [close < 20.0, close < 100.0, close < 500.0],
        [0.5, 1.0, 5.0],
        default=10.0,
    )
    inc_series = pd.Series(increments, index=out.index, dtype=float)
    nearest_round = (close / inc_series).round() * inc_series
    out["Col_DistNearestRound_ATR"] = _atr_normalize(out, close - nearest_round)

    return out


# ====================== MARKET CONTEXT & RELATIVE ======================
def add_market_context_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Market-relative columns.

    Extend with external benchmark data by pre-merging one or more of these columns
    into the same DataFrame index before calling this function:
        - SPX_Close / SPY_Close (broad market proxy)
        - Sector_Close / XLK_Close / XLF_Close / etc. (sector proxy)

    If external series are missing, output columns are created as NaN.
    """
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)

    stock_ret_1d = out["Close"].pct_change()
    stock_ret_20d = out["Close"].pct_change(20)

    spx = _pick_external_series(out, ["SPX_Close", "SPY_Close", "Benchmark_Close"])
    sector = _pick_external_series(out, ["Sector_Close", "XLK_Close", "XLF_Close", "XLY_Close", "XLI_Close"])
    spy_for_beta = _pick_external_series(out, ["SPY_Close", "SPX_Close", "Benchmark_Close"])

    if spx is not None:
        out["Col_StockVsSPX_TodayPct"] = (stock_ret_1d - spx.pct_change()) * 100.0
    else:
        out["Col_StockVsSPX_TodayPct"] = np.nan

    if sector is not None:
        out["Col_RelStrengthVsSector_20d"] = (stock_ret_20d - sector.pct_change(20)) * 100.0
    else:
        out["Col_RelStrengthVsSector_20d"] = np.nan

    if spy_for_beta is not None:
        mkt_ret = spy_for_beta.pct_change()
        cov = stock_ret_1d.rolling(60).cov(mkt_ret)
        var = mkt_ret.rolling(60).var()
        out["Col_Beta60d"] = _safe_div(cov, var)
        out["Col_CorrToSPY_10d"] = stock_ret_1d.rolling(10).corr(mkt_ret)
    else:
        out["Col_Beta60d"] = np.nan
        out["Col_CorrToSPY_10d"] = np.nan

    return out


# ====================== SIMPLE TIME / SESSION ======================
def add_time_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)

    out["Col_EntryTime_HourNumeric"] = out.index.hour + (out.index.minute / 60.0)
    out["Col_DayOfWeek"] = out.index.dayofweek

    if _is_intraday(out.index):
        hhmm = out.index.hour + (out.index.minute / 60.0)
        out["Col_SessionFlag"] = np.select(
            [
                (hhmm >= 4.0) & (hhmm < 9.5),   # premarket
                (hhmm >= 9.5) & (hhmm < 16.0),  # regular session
                (hhmm >= 16.0) & (hhmm < 20.0), # postmarket
            ],
            [1, 2, 3],
            default=0,
        )
    else:
        out["Col_SessionFlag"] = 2
    mins_raw = (out.index.hour - 9) * 60 + np.asarray(out.index.minute)
    out["Col_MinutesSinceOpen"] = np.maximum(mins_raw, 0)
    return out


# ====================== MASTER ORCHESTRATOR ======================
def add_all_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = _copy_if_needed(df, inplace)
    _validate_ohlcv_df(out)

    out = add_volatility_columns(out, inplace=True)
    out = add_trend_momentum_columns(out, inplace=True)
    out = add_oscillator_columns(out, inplace=True)
    out = add_volume_vwap_columns(out, inplace=True)
    out = add_price_action_columns(out, inplace=True)
    out = add_gaps_columns(out, inplace=True)
    out = add_key_levels_columns(out, inplace=True)
    out = add_market_context_columns(out, inplace=True)
    out = add_time_columns(out, inplace=True)
    return out


# ====================== PUBLIC METADATA API ======================
def get_all_column_names() -> List[str]:
    names: List[str] = []
    for group_cols in _COLUMN_GROUPS.values():
        names.extend(group_cols)
    return names


def get_column_groups() -> Dict[str, List[str]]:
    return {k: v.copy() for k, v in _COLUMN_GROUPS.items()}


# ====================== 78 MISSING INDICATORS (TA-Lib non-CDL + Backtrader uniques) ======================
def _pick_ohlcv_columns(df: pd.DataFrame) -> tuple:
    """Return (open, high, low, close, volume) column names. Accepts both long and single-ticker formats."""
    o = "open" if "open" in df.columns else ("Open" if "Open" in df.columns else None)
    h = "high" if "high" in df.columns else ("High" if "High" in df.columns else None)
    l = "low" if "low" in df.columns else ("Low" if "Low" in df.columns else None)
    c = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    v = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    return o, h, l, c, v


def _pick_indicator_col(ind_df: Optional[pd.DataFrame], index: pd.Index, prefixes: List[str]) -> pd.Series:
    """Extract first matching column from pandas_ta output by prefix."""
    if ind_df is None or ind_df.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    for col in ind_df.columns:
        c = str(col)
        if any(c.startswith(p) for p in prefixes):
            s = ind_df[col].reindex(index)
            return s.astype(float)
    return pd.Series(np.nan, index=index, dtype=float)


def add_all_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 103 TA-Lib / Backtrader / pandas_ta indicators in one pass (78 + 25).

    Works on long-format minute-bar DataFrames with columns: Ticker, datetime,
    open, high, low, close, volume. Also supports single-ticker with DatetimeIndex
    and capitalized Open, High, Low, Close, Volume.
    """
    df = df.copy()
    o_col, h_col, l_col, c_col, v_col = _pick_ohlcv_columns(df)
    if not all([o_col, h_col, l_col, c_col]):
        raise ValueError("DataFrame must have open/high/low/close (or Open/High/Low/Close)")

    has_ticker = "Ticker" in df.columns
    has_datetime_col = "datetime" in df.columns

    if has_ticker:
        sort_cols = ["Ticker"]
        if has_datetime_col:
            sort_cols.append("datetime")
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values(sort_cols).reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    def _add_indicators_to_group(g: pd.DataFrame) -> pd.DataFrame:
        ohlc = g[[o_col, h_col, l_col, c_col]].copy()
        ohlc.columns = ["open", "high", "low", "close"]
        if v_col and v_col in g.columns:
            ohlc["volume"] = g[v_col]
        idx = g.index

        def _col(prefixes: List[str], ind_df: Optional[pd.DataFrame]) -> pd.Series:
            return _pick_indicator_col(ind_df, idx, prefixes)

        # === Overlap / MAs ===
        g["Col_DEMA20"] = ta.dema(ohlc["close"], length=20)
        g["Col_TEMA20"] = ta.tema(ohlc["close"], length=20)
        g["Col_KAMA20"] = ta.kama(ohlc["close"], length=20)
        mama_df = ta.mama(ohlc["close"])
        g["Col_MAMA"] = _col(["MAMA"], mama_df)
        g["Col_FAMA"] = _col(["FAMA"], mama_df)
        g["Col_TRIMA20"] = ta.trima(ohlc["close"], length=20)
        g["Col_WMA20"] = ta.wma(ohlc["close"], length=20)

        # === Volatility / SAR ===
        psar_df = ta.psar(ohlc["high"], ohlc["low"], ohlc["close"])
        if psar_df is not None:
            psar_long = _col(["PSARl"], psar_df)
            psar_short = _col(["PSARs"], psar_df)
            g["Col_SAR"] = psar_long.fillna(psar_short)
        else:
            g["Col_SAR"] = np.nan
        psar_ext = ta.psar(ohlc["high"], ohlc["low"], ohlc["close"], af0=0.01, max_af=0.25)
        if psar_ext is not None:
            pe_long = _col(["PSARl"], psar_ext)
            pe_short = _col(["PSARs"], psar_ext)
            g["Col_SAREXT"] = pe_long.fillna(pe_short)
        else:
            g["Col_SAREXT"] = np.nan
        natr_ser = ta.natr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
        g["Col_NATR14"] = natr_ser if natr_ser is not None else pd.Series(np.nan, index=idx)

        # === Momentum / Oscillators ===
        adx_df = ta.adx(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
        g["Col_ADXR14"] = _col(["ADXR"], adx_df)
        dmp = _col(["DMP"], adx_df)
        dmn = _col(["DMN"], adx_df)
        denom = dmp + dmn
        g["Col_DX14"] = np.where(denom > 0, 100 * _safe_div((dmp - dmn).abs(), denom), np.nan)
        atr14 = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
        atr_safe = atr14.replace(0, np.nan) if atr14 is not None else pd.Series(np.nan, index=idx)
        g["Col_PLUS_DI14"] = (100 * _safe_div(dmp, atr_safe)) if atr14 is not None else pd.Series(np.nan, index=idx)
        g["Col_MINUS_DI14"] = (100 * _safe_div(dmn, atr_safe)) if atr14 is not None else pd.Series(np.nan, index=idx)
        g["Col_APO"] = ta.apo(ohlc["close"], fast=12, slow=26)
        aroon_df = ta.aroon(ohlc["high"], ohlc["low"], length=14)
        g["Col_AROON_UP"] = _col(["AROONU"], aroon_df)
        g["Col_AROON_DOWN"] = _col(["AROOND"], aroon_df)
        g["Col_AROONOSC"] = _col(["AROONOSC"], aroon_df)
        g["Col_BOP"] = ta.bop(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        g["Col_CMO14"] = ta.cmo(ohlc["close"], length=14)
        if v_col and "volume" in ohlc.columns:
            g["Col_MFI14"] = ta.mfi(ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"], length=14)
        else:
            g["Col_MFI14"] = np.nan
        g["Col_MINUS_DM14"] = dmn
        g["Col_MOM10"] = ta.mom(ohlc["close"], length=10)
        g["Col_PLUS_DM14"] = dmp
        ppo_df = ta.ppo(ohlc["close"], fast=12, slow=26)
        g["Col_PPO"] = _col(["PPO_"], ppo_df) if ppo_df is not None else np.nan
        g["Col_ROCP10"] = ta.roc(ohlc["close"], length=10)
        shifted = ohlc["close"].shift(10)
        g["Col_ROCR10"] = _safe_div(ohlc["close"], shifted)
        g["Col_ROCR10010"] = 100 * _safe_div(ohlc["close"], shifted)
        stochf_df = ta.stochf(ohlc["high"], ohlc["low"], ohlc["close"], k=5, d=3)
        g["Col_STOCHF_K"] = _col(["STOCHFk"], stochf_df)
        g["Col_STOCHF_D"] = _col(["STOCHFd"], stochf_df)
        stochrsi_df = ta.stochrsi(ohlc["close"], length=14)
        g["Col_STOCHRSI_K"] = _col(["STOCHRSIk"], stochrsi_df)
        g["Col_STOCHRSI_D"] = _col(["STOCHRSId"], stochrsi_df)
        trix_df = ta.trix(ohlc["close"], length=15)
        g["Col_TRIX15"] = _col(["TRIX"], trix_df) if trix_df is not None else np.nan
        ultosc_ser = ta.uo(ohlc["high"], ohlc["low"], ohlc["close"])
        g["Col_ULTOSC"] = ultosc_ser if ultosc_ser is not None else np.nan
        g["Col_WILLR14"] = ta.willr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)

        # === Price transforms ===
        g["Col_AVGPRICE"] = ta.ohlc4(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        g["Col_MEDPRICE"] = (ohlc["high"] + ohlc["low"]) / 2
        g["Col_TYPPRICE"] = ta.hlc3(ohlc["high"], ohlc["low"], ohlc["close"])
        g["Col_WCLPRICE"] = (ohlc["high"] + ohlc["low"] + 2 * ohlc["close"]) / 4

        # === Hilbert Transform (pandas_ta has ht_trendline; others from TA-Lib if available) ===
        try:
            g["Col_HT_TRENDLINE"] = ta.ht_trendline(ohlc["close"])
        except Exception:
            g["Col_HT_TRENDLINE"] = np.nan
        for col, fn, args in [
            ("Col_HT_DCPERIOD", "ht_dcperiod", ()),
            ("Col_HT_DCPHASE", "ht_dcphase", ()),
            ("Col_HT_TRENDMODE", "ht_trendmode", ()),
        ]:
            fn_obj = getattr(ta, fn, None)
            if fn_obj is not None:
                try:
                    r = fn_obj(ohlc["close"], *args)
                    g[col] = r if r is not None else np.nan
                except Exception:
                    g[col] = np.nan
            else:
                g[col] = np.nan
        phasor_fn = getattr(ta, "ht_phasor", None)
        if phasor_fn is not None:
            try:
                phasor_df = phasor_fn(ohlc["close"])
                g["Col_HT_PHASOR_INPHASE"] = _col(["INPHASE"], phasor_df)
                g["Col_HT_PHASOR_QUADRATURE"] = _col(["QUADRATURE"], phasor_df)
            except Exception:
                g["Col_HT_PHASOR_INPHASE"] = g["Col_HT_PHASOR_QUADRATURE"] = np.nan
        else:
            g["Col_HT_PHASOR_INPHASE"] = g["Col_HT_PHASOR_QUADRATURE"] = np.nan
        sine_fn = getattr(ta, "ht_sine", None)
        if sine_fn is not None:
            try:
                sine_df = sine_fn(ohlc["close"])
                g["Col_HT_SINE_SINE"] = _col(["SINE"], sine_df)
                g["Col_HT_SINE_LEADSINE"] = _col(["LEADSINE"], sine_df)
            except Exception:
                g["Col_HT_SINE_SINE"] = g["Col_HT_SINE_LEADSINE"] = np.nan
        else:
            g["Col_HT_SINE_SINE"] = g["Col_HT_SINE_LEADSINE"] = np.nan

        # === Linear Regression ===
        g["Col_LINEARREG20"] = ta.linreg(ohlc["close"], length=20)
        lra = ta.linreg(ohlc["close"], length=20, angle=True)
        g["Col_LINEARREG_ANGLE20"] = lra if lra is not None else np.nan
        lri = ta.linreg(ohlc["close"], length=20, intercept=True)
        g["Col_LINEARREG_INTERCEPT20"] = lri if lri is not None else np.nan
        lrs = ta.linreg(ohlc["close"], length=20, slope=True)
        g["Col_LINEARREG_SLOPE20"] = lrs if lrs is not None else np.nan
        g["Col_STDDEV20"] = ta.stdev(ohlc["close"], length=20)
        g["Col_VAR20"] = ta.variance(ohlc["close"], length=20)

        # === Hull / Heikin Ashi / Ichimoku ===
        g["Col_HullMA20"] = ta.hma(ohlc["close"], length=20)
        ha_df = ta.ha(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        g["Col_HeikinAshi_Open"] = _col(["HA_open"], ha_df)
        g["Col_HeikinAshi_High"] = _col(["HA_high"], ha_df)
        g["Col_HeikinAshi_Low"] = _col(["HA_low"], ha_df)
        g["Col_HeikinAshi_Close"] = _col(["HA_close"], ha_df)

        ichi_result = ta.ichimoku(ohlc["high"], ohlc["low"], ohlc["close"], include_chikou=True, lookahead=False)
        if ichi_result and ichi_result[0] is not None:
            main_ichi = ichi_result[0]
            g["Col_Ichimoku_Tenkan"] = _col(["ITS"], main_ichi)
            g["Col_Ichimoku_Kijun"] = _col(["IKS"], main_ichi)
            g["Col_Ichimoku_SenkouA"] = _col(["ISA"], main_ichi)
            g["Col_Ichimoku_SenkouB"] = _col(["ISB"], main_ichi)
            g["Col_Ichimoku_Chikou"] = _col(["ICS"], main_ichi)
        else:
            for col in ["Col_Ichimoku_Tenkan", "Col_Ichimoku_Kijun", "Col_Ichimoku_SenkouA", "Col_Ichimoku_SenkouB", "Col_Ichimoku_Chikou"]:
                g[col] = np.nan

        g["Col_PivotPoint"] = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3

        ao_ser = ta.ao(ohlc["high"], ohlc["low"])
        g["Col_AwesomeOscillator"] = ao_ser if ao_ser is not None else np.nan
        if ao_ser is not None:
            ao_sma = ta.sma(ao_ser, length=5)
            g["Col_AccelerationDeceleration"] = ao_ser - ao_sma if ao_sma is not None else ao_ser
        else:
            g["Col_AccelerationDeceleration"] = np.nan

        g["Col_DPO"] = ta.dpo(ohlc["close"], length=20)
        kst_df = ta.kst(ohlc["close"])
        g["Col_KST"] = _col(["KST_"], kst_df) if kst_df is not None else np.nan

        g["Col_PercentRank20"] = ohlc["close"].rolling(20).rank(pct=True) * 100

        vortex_df = ta.vortex(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
        g["Col_Vortex_Plus"] = _col(["VTXP"], vortex_df)
        g["Col_Vortex_Minus"] = _col(["VTXM"], vortex_df)

        # === Final 25 (ALMA, FWMA, HWMA, PWMA, ZLMA, SuperTrend, Keltner, Donchian, etc.) ===
        g["Col_ALMA_10_6_0.85"] = ta.alma(ohlc["close"], length=10, sigma=6.0, offset=0.85)
        g["Col_FWMA_10"] = ta.fwma(ohlc["close"], length=10)
        g["Col_HWMA_10"] = ta.hwma(ohlc["close"], length=10)
        g["Col_PWMA_10"] = ta.pwma(ohlc["close"], length=10)
        g["Col_ZLMA_10"] = ta.zlma(ohlc["close"], length=10)
        st_df = ta.supertrend(ohlc["high"], ohlc["low"], ohlc["close"], length=10, multiplier=3.0)
        g["Col_SuperTrend"] = _col(["SUPERT_"], st_df) if st_df is not None else pd.Series(np.nan, index=idx)
        g["Col_SuperTrend_Direction"] = _col(["SUPERTd"], st_df) if st_df is not None else pd.Series(np.nan, index=idx)
        kc_df = ta.kc(ohlc["high"], ohlc["low"], ohlc["close"], length=20, scalar=2.0)
        kcu = _col(["KCU"], kc_df)
        kcl = _col(["KCL"], kc_df)
        g["Col_Keltner_Upper"] = kcu
        g["Col_Keltner_Lower"] = kcl
        g["Col_Keltner_Width_ATR"] = _safe_div(kcu - kcl, atr_safe)
        dc_df = ta.donchian(ohlc["high"], ohlc["low"], length=20)
        g["Col_Donchian_Upper"] = _col(["DCU"], dc_df) if dc_df is not None else pd.Series(np.nan, index=idx)
        g["Col_Donchian_Lower"] = _col(["DCL"], dc_df) if dc_df is not None else pd.Series(np.nan, index=idx)
        massi_ser = ta.massi(ohlc["high"], ohlc["low"], length=25)
        g["Col_MassIndex_25"] = massi_ser if massi_ser is not None else pd.Series(np.nan, index=idx)
        g["Col_UlcerIndex_14"] = ta.ui(ohlc["close"], length=14)
        g["Col_ChoppinessIndex_14"] = ta.chop(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
        if "volume" in ohlc.columns:
            g["Col_ChaikinMoneyFlow_20"] = ta.cmf(ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"], length=20)
            g["Col_ElderForceIndex_13"] = ta.efi(ohlc["close"], ohlc["volume"], length=13)
            eom_ser = ta.eom(ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"], length=14)
            g["Col_EaseOfMovement_14"] = eom_ser if eom_ser is not None else pd.Series(np.nan, index=idx)
            nvi_out = ta.nvi(ohlc["close"], ohlc["volume"])
            g["Col_NVI"] = _col(["NVI"], nvi_out) if isinstance(nvi_out, pd.DataFrame) else (nvi_out if nvi_out is not None else pd.Series(np.nan, index=idx))
            pvi_out = ta.pvi(ohlc["close"], ohlc["volume"])
            g["Col_PVI"] = _col(["PVI"], pvi_out) if isinstance(pvi_out, pd.DataFrame) else (pvi_out if pvi_out is not None else pd.Series(np.nan, index=idx))
        else:
            for col in ["Col_ChaikinMoneyFlow_20", "Col_ElderForceIndex_13", "Col_EaseOfMovement_14", "Col_NVI", "Col_PVI"]:
                g[col] = pd.Series(np.nan, index=idx)
        psar_final_df = ta.psar(ohlc["high"], ohlc["low"], ohlc["close"])
        if psar_final_df is not None:
            pl = _col(["PSARl"], psar_final_df)
            ps = _col(["PSARs"], psar_final_df)
            g["Col_PSAR"] = pl.fillna(ps)
        else:
            g["Col_PSAR"] = pd.Series(np.nan, index=idx)
        g["Col_QStick_10"] = ta.qstick(ohlc["open"], ohlc["close"], length=10)
        rvi_ser = ta.rvi(ohlc["close"], ohlc["high"], ohlc["low"], length=14)
        g["Col_RelativeVigorIndex"] = rvi_ser if rvi_ser is not None else pd.Series(np.nan, index=idx)
        rvi_sma = ta.sma(rvi_ser, length=4) if rvi_ser is not None else None
        g["Col_RVISignal"] = rvi_sma if rvi_sma is not None else pd.Series(np.nan, index=idx)
        stc_df = ta.stc(ohlc["close"])
        g["Col_SchaffTrendCycle"] = _col(["STC_"], stc_df) if stc_df is not None and not stc_df.empty else pd.Series(np.nan, index=idx)

        return g

    if has_ticker:
        group_col = "Ticker"
        pieces = []
        for name, grp in df.groupby(group_col, group_keys=False):
            out = _add_indicators_to_group(grp.drop(columns=[group_col]))
            out[group_col] = name
            pieces.append(out)
        df = pd.concat(pieces, ignore_index=True)
    else:
        df = _add_indicators_to_group(df)

    return df.reset_index(drop=True) if has_ticker else df


def add_final_22_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper: the 25 extra indicators (ALMA, SuperTrend, Keltner,
    Donchian, Mass Index, Ulcer, Choppiness, CMF, EFI, EOM, NVI, PVI, PSAR, QStick,
    RVI/RVISignal, Schaff Trend Cycle) are now included in add_all_missing_indicators.
    This function simply calls add_all_missing_indicators so one call adds all 103.
    """
    return add_all_missing_indicators(df)


# ====================== CRUNCHER CONTEXT (entry-only, optimization-ready) ======================
def add_cruncher_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 25 cruncher-context columns (gap, ORB, extension from 9 EMA, volume, volatility,
    interaction terms). Entry-only, lookahead-free. Supports long format (Ticker, datetime,
    open/high/low/close/volume) and single-ticker (DatetimeIndex, Open/High/Low/Close/Volume).
    Uses _safe_div and _atr_normalize; computes Col_ATR14 in-group if not present.
    """
    df = df.copy()
    o_col, h_col, l_col, c_col, v_col = _pick_ohlcv_columns(df)
    if not all([o_col, h_col, l_col, c_col]):
        raise ValueError("DataFrame must have open/high/low/close (or Open/High/Low/Close)")

    has_ticker = "Ticker" in df.columns
    has_datetime_col = "datetime" in df.columns

    if has_ticker:
        sort_cols = ["Ticker"]
        if has_datetime_col:
            sort_cols.append("datetime")
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values(sort_cols).reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    def _add_cruncher_to_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) == 0:
            return g
        o = g[o_col]
        h = g[h_col]
        l_ = g[l_col]
        c = g[c_col]
        v = g[v_col] if v_col and v_col in g.columns else None
        idx = g.index

        if "datetime" in g.columns:
            time = pd.to_datetime(g["datetime"])
            session_start = pd.Series(time.dt.normalize().values, index=idx)
        else:
            time = g.index
            session_start = pd.Series(time.normalize(), index=idx)

        if "Col_ATR14" not in g.columns:
            atr_ser = ta.atr(h, l_, c, length=14)
            g = g.copy()
            g["Col_ATR14"] = atr_ser if atr_ser is not None else pd.Series(np.nan, index=idx)

        prev_close = c.shift(1).fillna(c.iloc[0]) if len(c) else c.shift(1)
        gap_pct = _safe_div(c - prev_close, prev_close) * 100
        g["Col_Gap_Pct"] = gap_pct

        # Gap fill 15 min: one value per session, then map to rows
        def _gap_fill_15(sess):
            mask = (time - pd.Timestamp(sess)) <= pd.Timedelta(minutes=15)
            if not mask.any():
                return np.nan
            gp0 = gap_pct.loc[mask].iloc[0]
            if pd.isna(gp0) or gp0 <= 0:
                return 0.0 if not pd.isna(gp0) else np.nan
            c_max = c.loc[mask].max()
            c0 = c.loc[mask].iloc[0]
            return float(_safe_div(pd.Series([c_max - c0]), pd.Series([gp0])).iloc[0]) * 100

        sess_uniq = session_start.unique()
        gap_fill_by_sess = pd.Series({s: _gap_fill_15(s) for s in sess_uniq})
        g["Col_GapFill_15min_Pct"] = session_start.map(gap_fill_by_sess)
        gap_fill_ser = gap_fill_by_sess.sort_index()
        roll_mean = gap_fill_ser.rolling(20, min_periods=1).mean()
        roll_std = gap_fill_ser.rolling(20, min_periods=1).std().replace(0, np.nan)
        zscore = _safe_div(gap_fill_ser - roll_mean, roll_std)
        g["Col_GapFill_15min_Zscore"] = session_start.map(zscore)

        # ORB: per-session high/low for first N minutes, then broadcast
        for mins in [5, 15, 30, 60]:
            def _orb_high(s):
                m = (time - pd.Timestamp(s)) <= pd.Timedelta(minutes=mins)
                return h.loc[m].max() if m.any() else np.nan

            def _orb_low(s):
                m = (time - pd.Timestamp(s)) <= pd.Timedelta(minutes=mins)
                return l_.loc[m].min() if m.any() else np.nan

            orb_high_s = pd.Series({s: _orb_high(s) for s in sess_uniq})
            orb_low_s = pd.Series({s: _orb_low(s) for s in sess_uniq})
            orb_high_map = orb_high_s.reindex(session_start).set_axis(idx)
            orb_low_map = orb_low_s.reindex(session_start).set_axis(idx)
            g[f"Col_ORB_{mins}min_BreakHigh"] = (c > orb_high_map).astype(int)
            g[f"Col_ORB_{mins}min_BreakLow"] = (c < orb_low_map).astype(int)
            g[f"Col_ORB_{mins}min_DistHigh_ATR"] = _atr_normalize(g, c - orb_high_map)

        # Extension from daily 9 EMA
        ema9 = ta.ema(c, length=9)
        if ema9 is not None:
            last_ema9 = ema9.groupby(session_start).last()
            ext_9ema = c - session_start.map(last_ema9)
            g["Col_ExtensionFromDaily9EMA_ATR"] = _atr_normalize(g, ext_9ema)
        else:
            g["Col_ExtensionFromDaily9EMA_ATR"] = pd.Series(np.nan, index=idx)
        g["Col_ExtensionFromDaily9EMA_Rank"] = g["Col_ExtensionFromDaily9EMA_ATR"].groupby(session_start).rank(pct=True) * 100

        # Multi-day slope (5-day slope of 10-period daily SMA)
        daily_close = c.groupby(session_start).last()
        sma10 = daily_close.rolling(10, min_periods=1).mean()
        slope_5d = (sma10 - sma10.shift(5)) / 5
        g["Col_MultiDaySlope_5d"] = session_start.map(slope_5d)

        # Inside day
        prev_high = h.groupby(session_start).max().shift(1)
        prev_low = l_.groupby(session_start).min().shift(1)
        g["Col_InsideDay"] = ((h <= session_start.map(prev_high).values) & (l_ >= session_start.map(prev_low).values)).astype(int)

        # Volume / liquidity
        if v is not None:
            avg_price = (h + l_ + c) / 3
            daily_dollar = (avg_price * v).groupby(session_start).sum()
            g["Col_DollarVolume_20dAvg"] = session_start.map(daily_dollar.rolling(20, min_periods=1).mean()).values

            def _vol_30(s):
                m = (time - pd.Timestamp(s)) <= pd.Timedelta(minutes=30)
                return v.loc[m].sum() if m.any() else np.nan

            vol_30_s = pd.Series({s: _vol_30(s) for s in sess_uniq})
            avg_30 = v.groupby(session_start).sum().rolling(20, min_periods=1).mean()
            rel_vol_30 = _safe_div(vol_30_s.reindex(session_start), avg_30.reindex(session_start))
            g["Col_RelativeVolume_First30min"] = rel_vol_30.values
            rank_rv = pd.Series(rel_vol_30.values, index=idx).groupby(session_start).rank(pct=True) * 100
            g["Col_RelativeVolume_First30min_Rank"] = rank_rv.values

            g["Col_VolumeSurge_1min_Ratio"] = _safe_div(v, v.rolling(20, min_periods=1).mean())
        else:
            g["Col_DollarVolume_20dAvg"] = np.nan
            g["Col_RelativeVolume_First30min"] = np.nan
            g["Col_RelativeVolume_First30min_Rank"] = np.nan
            g["Col_VolumeSurge_1min_Ratio"] = np.nan

        # Volatility
        range_15 = (h - l_).rolling(15, min_periods=1).max()
        g["Col_IntradayATR_Ratio"] = _safe_div(range_15, g["Col_ATR14"])
        g["Col_StdDev_Last10Bars_ATR"] = _atr_normalize(g, c.rolling(10, min_periods=1).std())

        # Tier 3: VWAP slope, RSI divergence, session VWAP distance
        typical = (h + l_ + c) / 3
        if v is not None:
            pv = typical * v
            session_pv = pv.groupby(session_start).cumsum()
            session_vol = v.groupby(session_start).cumsum()
            session_vwap = _safe_div(session_pv, session_vol)
            vwap_slope = session_vwap.rolling(10, min_periods=1).mean().diff()
            g["Col_DistToVWAP_Slope10_ATR"] = _atr_normalize(g, vwap_slope)
            g["Col_DistFromSessionVWAP_ATR"] = _atr_normalize(g, c - session_vwap)
        else:
            g["Col_DistToVWAP_Slope10_ATR"] = pd.Series(np.nan, index=idx)
            g["Col_DistFromSessionVWAP_ATR"] = pd.Series(np.nan, index=idx)
        rsi = g["Col_RSI14"] if "Col_RSI14" in g.columns else ta.rsi(c, length=14)
        if rsi is not None:
            price_change = c.pct_change().replace([np.inf, -np.inf], np.nan)
            rsi_change = rsi.pct_change().replace([np.inf, -np.inf], np.nan)
            g["Col_MomentumDivergence_RSI"] = (rsi_change - price_change).replace([np.inf, -np.inf], np.nan)
        else:
            g["Col_MomentumDivergence_RSI"] = pd.Series(np.nan, index=idx)

        # Interaction terms
        g["Col_GapPct_x_RelVol30"] = gap_pct * g.get("Col_RelativeVolume_First30min", pd.Series(1.0, index=idx))
        g["Col_GapPct_x_ExtensionATR"] = gap_pct * g.get("Col_ExtensionFromDaily9EMA_ATR", pd.Series(1.0, index=idx))

        return g

    if has_ticker:
        group_col = "Ticker"
        pieces = []
        for name, grp in df.groupby(group_col, group_keys=False):
            out = _add_cruncher_to_group(grp)
            out[group_col] = name
            pieces.append(out)
        df = pd.concat(pieces, ignore_index=True)
    else:
        df = _add_cruncher_to_group(df)

    return df.reset_index(drop=True) if has_ticker else df


# ====================== CONTINUOUS INTRATRADE TRACKING ======================
# Columns to track every minute while position is open (Entry / Exit / Max / Min / At30min / At60min)
CONTINUOUS_TRACKING_COLUMNS: List[str] = [
    "Col_ExtensionFromDaily9EMA_ATR",
    "Col_VWAP_Deviation_ATR",
    "Col_DistToVWAP_Slope10_ATR",
    "Col_DistFromSessionVWAP_ATR",
    "Col_MomentumDivergence_RSI",
    "Col_RSI14",
    "Col_StochK_14_3",
    "Col_MACD_Hist",
    "Col_BollingerPctB",
    "Col_ORB_15min_DistHigh_ATR",
    "Col_StdDev_Last10Bars_ATR",
    "Col_CCI20",
    "Col_VolumeSurge_1min_Ratio",
    "Col_IntradayATR_Ratio",
    "Col_ChoppinessIndex_14",
    "Col_SuperTrend",
    "Col_Keltner_Upper",
    "Col_Keltner_Lower",
    "Col_RelativeVigorIndex",
    "Col_SchaffTrendCycle",
]


def add_continuous_tracking(
    df_enriched: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    entry_time_col: str = "EntryTime",
    exit_time_col: str = "ExitTime",
    ticker_col: str = "Ticker",
    datetime_col: str = "datetime",
    columns: Optional[List[str]] = None,
    at_minutes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add continuous intra-trade tracking for selected Col_* columns.

    For each column we add: _Entry, _Exit, _Max, _Min, and _At{N}min for each N in at_minutes.
    df_enriched must have Ticker and datetime (or your names via ticker_col/datetime_col)
    and the Col_* columns. trades must have ticker, entry time, and exit time.

    Usage:
        df = add_all_missing_indicators(df)
        df = add_cruncher_context_columns(df)
        result.trades = add_continuous_tracking(df, result.trades)
    """
    trades = trades.copy()
    if columns is None:
        columns = CONTINUOUS_TRACKING_COLUMNS
    if at_minutes is None:
        at_minutes = [30, 60]

    # Work with columns; reset MultiIndex so we have Ticker and datetime as columns
    df_work = df_enriched.reset_index() if isinstance(df_enriched.index, pd.MultiIndex) else df_enriched.copy()
    if ticker_col not in df_work.columns or datetime_col not in df_work.columns:
        raise ValueError(f"df_enriched must have {ticker_col} and {datetime_col} (or reset MultiIndex)")

    for col in columns:
        if col not in df_work.columns:
            continue
        # Entry / Exit lookup: (ticker, time) -> value (last wins if duplicate keys)
        keys = list(zip(df_work[ticker_col], pd.to_datetime(df_work[datetime_col])))
        col_dict = dict(zip(keys, df_work[col].values))

        entry_keys = list(zip(trades[ticker_col], pd.to_datetime(trades[entry_time_col])))
        exit_keys = list(zip(trades[ticker_col], pd.to_datetime(trades[exit_time_col])))

        trades[f"{col}_Entry"] = [col_dict.get(k, np.nan) for k in entry_keys]
        trades[f"{col}_Exit"] = [col_dict.get(k, np.nan) for k in exit_keys]

        # Max / Min over [entry, exit] per trade
        def _max_min(row: pd.Series, agg: str) -> float:
            t = row[ticker_col]
            entry_ts = pd.Timestamp(row[entry_time_col])
            exit_ts = pd.Timestamp(row[exit_time_col])
            mask = (
                (df_work[ticker_col] == t)
                & (pd.to_datetime(df_work[datetime_col]) >= entry_ts)
                & (pd.to_datetime(df_work[datetime_col]) <= exit_ts)
            )
            if not mask.any():
                return np.nan
            s = df_work.loc[mask, col]
            return s.max() if agg == "max" else s.min()

        trades[f"{col}_Max"] = trades.apply(lambda row: _max_min(row, "max"), axis=1)
        trades[f"{col}_Min"] = trades.apply(lambda row: _max_min(row, "min"), axis=1)

        for minutes in at_minutes:
            target = pd.to_datetime(trades[entry_time_col]) + pd.Timedelta(minutes=minutes)
            at_keys = list(zip(trades[ticker_col], target))
            trades[f"{col}_At{minutes}min"] = [col_dict.get(k, np.nan) for k in at_keys]

    return trades


def get_minute_by_minute_tracking(
    df_enriched: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    entry_time_col: str = "EntryTime",
    exit_time_col: str = "ExitTime",
    ticker_col: str = "Ticker",
    datetime_col: str = "datetime",
    columns: Optional[List[str]] = None,
    trade_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return minute-by-minute tracking for selected Col_* columns through each trade.

    Output: one row per (trade, minute bar) with datetime, minute_offset from entry,
    and the value of each tracked column at that minute. Use for exit discovery,
    plotting evolution, or ML features.

    Usage:
        df = add_all_missing_indicators(df)
        df = add_cruncher_context_columns(df)
        mbm = get_minute_by_minute_tracking(df, result.trades)
    """
    if columns is None:
        columns = [c for c in CONTINUOUS_TRACKING_COLUMNS if c in df_enriched.columns]

    df_work = df_enriched.reset_index() if isinstance(df_enriched.index, pd.MultiIndex) else df_enriched.copy()
    if ticker_col not in df_work.columns or datetime_col not in df_work.columns:
        raise ValueError(f"df_enriched must have {ticker_col} and {datetime_col} (or reset MultiIndex)")

    dt_series = pd.to_datetime(df_work[datetime_col])
    rows = []

    for trade_idx, row in trades.iterrows():
        t = row[ticker_col]
        entry_ts = pd.Timestamp(row[entry_time_col])
        exit_ts = pd.Timestamp(row[exit_time_col])
        mask = (
            (df_work[ticker_col] == t)
            & (dt_series >= entry_ts)
            & (dt_series <= exit_ts)
        )
        if not mask.any():
            continue
        slice_df = df_work.loc[mask].copy()
        slice_df = slice_df.sort_values(datetime_col)
        slice_df["minute_offset"] = (pd.to_datetime(slice_df[datetime_col]) - entry_ts).dt.total_seconds() / 60
        slice_df["trade_idx"] = trade_idx
        if trade_id_col and trade_id_col in trades.columns:
            slice_df["trade_id"] = row[trade_id_col]
        rows.append(slice_df)

    if not rows:
        return pd.DataFrame(
            columns=[ticker_col, datetime_col, "trade_idx", "minute_offset"]
            + [c for c in columns if c in df_work.columns]
        )

    out = pd.concat(rows, ignore_index=True)
    return out

