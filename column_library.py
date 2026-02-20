"""
===========================================================================
Portable Column Library (single-file, reusable across backtesting projects)
===========================================================================

Dependencies:
    pip install pandas numpy pandas_ta

Usage example:
    import pandas as pd
    from column_library import (
        add_all_columns,
        add_volatility_columns,
        add_volume_vwap_columns,
        get_all_column_names,
        get_column_groups,
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
    out["Col_RangeExpansionToday_Pct"] = _safe_div(daily_range, daily_range.rolling(5).mean()) * 100.0
    atr5 = ta.atr(out["High"], out["Low"], out["Close"], length=5)
    atr20 = ta.atr(out["High"], out["Low"], out["Close"], length=20)
    out["Col_VolatilityRatio_20_5"] = _safe_div(atr5, atr20)
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
    out["Col_ExtensionFromOpen_ATR"] = _atr_normalize(out, out["Close"] - out["Open"])
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
        bar_index = out.groupby(session).cumcount() + 1
        expected_cum = today_cum_vol.groupby(bar_index).transform(lambda s: s.rolling(20, min_periods=5).mean())
        out["Col_CumulativeVol_vs_Avg_Pct"] = _safe_div(today_cum_vol, expected_cum) * 100.0
    else:
        out["Col_TodayVol_vs_YestVol"] = _safe_div(out["Volume"], out["Volume"].shift(1))
        out["Col_CumulativeVol_vs_Avg_Pct"] = _safe_div(out["Volume"], out["Volume"].rolling(20).mean()) * 100.0
    out["Col_VolumeSurge_15min_Pct"] = _safe_div(out["Volume"], out["Volume"].rolling(15).mean()) * 100.0

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
    if intraday:
        hhmm = out.index.hour + out.index.minute / 60.0
        premarket = out["Volume"].where((hhmm >= 4.0) & (hhmm < 9.5), 0.0)
        premarket_by_session = premarket.groupby(session).sum()
        avg_premarket = premarket_by_session.rolling(20, min_periods=5).mean()
        out["Col_PreMarketVolume_Ratio"] = _safe_div(session.map(premarket_by_session), session.map(avg_premarket))
    else:
        out["Col_PreMarketVolume_Ratio"] = np.nan
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
    out["Col_MomentumScore_5min"] = _atr_normalize(out, out["Close"] - out["Close"].shift(5))
    up_bars = (out["Close"] > out["Open"]).astype(int)
    up_groups = (up_bars == 0).cumsum()
    out["Col_ConsecutiveUpBars"] = up_bars.groupby(up_groups).cumsum()
    out["Col_BodyToRangeRatio"] = _safe_div((out["Close"] - out["Open"]).abs(), out["High"] - out["Low"])
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
    out["Col_GapFillProbability_Proxy"] = _safe_div((day_open - prev_close).abs(), out["Col_ATR14"])
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
    if _is_intraday(out.index):
        minutes = out.index.hour * 60 + out.index.minute
        out["Col_MinutesSinceOpen"] = np.maximum(minutes - 570, 0)
    else:
        out["Col_MinutesSinceOpen"] = np.nan
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

