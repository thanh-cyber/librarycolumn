# column_library.py — Deep Logic & Bug Audit (Updated)

**Date:** 2026-02-19  
**Scope:** Full logic and bug audit after long-format refactor

---

## Executive Summary

The library supports both long format and single-ticker. Several **critical lookahead bugs** remain in cruncher columns. New issues from the long-format refactor include pandas_ta None handling and VWAP fallback semantics. One **logic bug** in Inside Day. Minor robustness issues noted.

---

## Critical — Lookahead (Invalid for Backtesting)

### 1. Col_ExtensionFromDaily9EMA_ATR — uses end-of-session EMA

**Location:** `_add_cruncher_to_group`, lines 1347–1353

**Issue:** Uses `ema9.groupby(session_start).last()` — the **last** (end-of-session) EMA value mapped to every bar. At 10:00 AM, the bar gets the 4:00 PM EMA.

**Fix:** Use the bar's own EMA:
```python
ext_9ema = c - ema9
g["Col_ExtensionFromDaily9EMA_ATR"] = _atr_normalize(g, ext_9ema)
```
Col_ExtensionFromDaily9EMA_Rank inherits this lookahead.

---

### 2. Col_MultiDaySlope_5d — uses today's closing price intraday

**Location:** `_add_cruncher_to_group`, lines 1356–1359

**Issue:** `daily_close = c.groupby(session_start).last()` — at 10:00 AM on day D, `slope_5d` uses `daily_close[D]` (today's closing price), which is not known until EOD.

**Fix:** Use only prior sessions:
```python
daily_close = c.groupby(session_start).last()
sma10 = daily_close.rolling(10, min_periods=1).mean()
slope_5d = (sma10.shift(1) - sma10.shift(6)) / 5
g["Col_MultiDaySlope_5d"] = session_start.map(slope_5d)
```

---

## Logic Bugs

### 3. Col_InsideDay — wrong definition for minute bars

**Location:** `_add_cruncher_to_group`, lines 1362–1364

**Issue:** Uses `h` and `l_` (current bar's high/low), not the day's high/low so far. For minute bars, need cumulative day range.

**Fix:**
```python
day_high_so_far = h.groupby(session_start).cummax()
day_low_so_far = l_.groupby(session_start).cummin()
prev_high = h.groupby(session_start).max().shift(1)
prev_low = l_.groupby(session_start).min().shift(1)
g["Col_InsideDay"] = ((day_high_so_far <= session_start.map(prev_high).values)
                      & (day_low_so_far >= session_start.map(prev_low).values)).astype(int)
```

---

## Medium — Robustness / Edge Cases

### 4. _add_cruncher_to_group — no defensive copy

**Location:** `_add_cruncher_to_group`, line 1282

**Issue:** `g` from groupby may be a view; assigning new columns can trigger SettingWithCopyWarning.

**Fix:** Add `g = g.copy()` at the start (after `len(g) == 0` check).

---

### 5. _add_volatility_to_group — ta.atr / ta.true_range can return None

**Location:** lines 363, 369

**Issue:** `ta.atr` and `ta.true_range` can return None for edge inputs (empty series, bad data). Would cause TypeError or wrong behavior.

**Fix:** Add None checks:
```python
atr_ser = ta.atr(h, l_, c, length=14)
g["Col_ATR14"] = atr_ser if atr_ser is not None else pd.Series(np.nan, index=g.index)
# ...
true_range = ta.true_range(h, l_, c)
if true_range is None:
    true_range = pd.Series(np.nan, index=g.index)
g["Col_TrueRange_vs_ATR"] = _safe_div(true_range, g["Col_ATR14"])
```

---

### 6. _add_oscillator_to_group — ta.rsi / ta.bbands can return None

**Location:** lines 499, 511

**Issue:** `g["Col_RSI14"] = ta.rsi(...)` — if ta.rsi returns None, column gets None. `bb = ta.bbands(...)` — if None, `_indicator_col(bb, ...)` handles it, but bb_low/bb_up from None would fail before that. `_indicator_col` returns Series for None/empty input, so we're safe for bb. For RSI:
```python
rsi_ser = ta.rsi(c, length=14)
g["Col_RSI14"] = rsi_ser if rsi_ser is not None else pd.Series(np.nan, index=g.index)
```

---

### 7. _add_volume_vwap_to_group — VWAP fallback is not per-session

**Location:** lines 590–594

**Issue:** When `ta.vwap` returns None (e.g. long format with RangeIndex), fallback uses `(typical * v).cumsum() / v.cumsum()`, which is global cumsum, not per-session. For intraday data with multiple sessions, this is wrong.

**Fix:** Use session-weighted fallback:
```python
if daily_vwap is None:
    session = _session_labels_for_group(g)
    cum_pv = (typical * v).groupby(session).cumsum()
    cum_v = v.groupby(session).cumsum()
    daily_vwap = _safe_div(cum_pv, cum_v)
```

---

### 8. Col_CumulativeVol_vs_Avg_Pct — formula quirks

**Location:** `_add_volume_vwap_to_group`, lines 577–578

**Issues:**
- `v.cumsum()` is unbounded across sessions; for multi-day data this grows without reset.
- `rolling(390)` assumes 390 bars per session; may not match actual frequency.
- `minutes_idx` uses (hour-9)*60+minute+1; at 9:30 this is 31, not 0.

**Recommendation:** Document or add session-level reset and configurable session length.

---

### 9. Empty groups — no explicit handling

**Location:** All `_add_*_to_group` functions

**Issue:** If a ticker has 0 rows, `grp` is empty. `ta.atr` on empty series may return None or empty. Most operations on empty g would yield empty results. Likely OK but untested.

---

## Minor — Documentation / Consistency

### 10. Col_ConsecutiveUpBars — off-by-one semantics

**Location:** `_add_price_action_to_group`, line 583

**Observation:** `cumcount()` is 0-based. First up bar in a streak yields 0. If "consecutive up bars" should include current (e.g. 3 = 3 bars), use `cumcount() + 1`.

---

### 11. Col_MomentumDivergence_RSI — scale mismatch

**Location:** cruncher, lines 1409–1411

**Observation:** `rsi_change - price_change` mixes scales (RSI pct vs price pct). Sign is correct for divergence; magnitude is not normalized.

---

### 12. Col_MinutesSinceOpen — assumes 9:00 open

**Location:** `_add_time_to_group`, lines 883, 886

**Observation:** Formula `(hour - 9) * 60 + minute` gives 30 at 9:30. US markets often open at 9:30. Document or make configurable.

---

### 13. pct_change FutureWarning

**Location:** cruncher Col_MomentumDivergence_RSI, line 1410

**Issue:** `rsi.pct_change()` triggers FutureWarning about default `fill_method='pad'`. Use `pct_change(fill_method=None)` or fill NaN before calling.

---

## Long-Format Refactor — Verification

| Area | Status | Notes |
|------|--------|-------|
| _pick_ohlcv_columns | OK | Handles open/Open, etc. |
| _session_labels_for_group | OK | Works with datetime col or index |
| _is_intraday_for_group | OK | Duplicated dates check |
| _session_prev_map(session=) | OK | Optional session param |
| _add_volatility_to_group | OK | Uses col names |
| _add_trend_momentum_to_group | OK | None handling for sma/linreg |
| _add_oscillator_to_group | OK | Uses col names |
| _add_volume_vwap_to_group | OK | VWAP fallback issue noted |
| _add_price_action_to_group | OK | session for prev map |
| _add_gaps_to_group | OK | session for day_open |
| _add_key_levels_to_group | OK | Simple |
| _add_market_context_to_group | OK | Per-group |
| _add_time_to_group | OK | dt.dt.hour for long format |

---

## add_all_missing_indicators — Potential Issues

- **g["Col_PPO"]** (line 966): `_col(..., ppo_df) if ppo_df is not None else np.nan` — assigns scalar `np.nan` to entire column; should be `pd.Series(np.nan, index=idx)` for consistency.
- **Col_LINEARREG_ANGLE20 etc.** (lines 1142–1145): Same pattern — `np.nan` scalar. May cause dtype issues.
- **add_all_missing_indicators long format**: When `has_ticker` and no `datetime`, `sort_cols = ["Ticker"]` only. `df.sort_values(["Ticker"])` without datetime can scramble bar order within a ticker if rows aren't already sorted. If `datetime` is missing in long format, bar order is undefined.

---

## add_continuous_tracking — Verification

- Entry/Exit lookup uses `(ticker, datetime)` keys; correct.
- Max/Min over [entry, exit] inclusive; correct.
- At30min/At60min: exact timestamp match — may miss if bar doesn't exist at that exact minute. Design choice (nearest bar would need different logic).

---

## Summary of Recommended Fixes

| Priority | Item | Action |
|----------|------|--------|
| **Critical** | ExtensionFromDaily9EMA | Use bar-level ema9, not session .last() |
| **Critical** | MultiDaySlope_5d | Use only prior-session closes |
| **Logic** | InsideDay | Use cummax/cummin for day range so far |
| **Medium** | _add_cruncher_to_group | Add g.copy() at start |
| **Medium** | _add_volatility_to_group | Handle ta.atr, ta.true_range None |
| **Medium** | _add_oscillator_to_group | Handle ta.rsi None |
| **Medium** | VWAP fallback | Use session-weighted cumsum when ta.vwap returns None |
| **Minor** | pct_change | Use fill_method=None to silence FutureWarning |
| **Minor** | ConsecutiveUpBars | Document or use cumcount()+1 |
| **Minor** | Col_CumulativeVol_vs_Avg_Pct | Document formula |
| **Minor** | Col_MinutesSinceOpen | Document 9:00 assumption |

---

## Test Recommendations

1. Unit test: no cruncher column uses future data.
2. Edge cases: empty groups, single-row groups, len(g) &lt; 14 for ATR.
3. Long format: verify session boundaries and groupby correctness.
4. ta.* None returns: test with degenerate inputs (all NaN, single row).
