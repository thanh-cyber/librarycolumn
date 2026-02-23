# column_library.py — Deep Logic & Bug Audit

**Date:** 2026-02-19  
**Scope:** Full logic and bug audit of `column_library.py`

---

## Executive Summary

The library is generally well-structured. Several **critical lookahead bugs** exist in cruncher columns that would invalidate backtests. One **logic bug** in Inside Day. Minor robustness and documentation issues noted.

---

## Critical — Lookahead (Invalid for Backtesting)

These columns use future information and must be fixed for any live/backtest use.

### 1. Col_ExtensionFromDaily9EMA_ATR — uses end-of-session EMA

**Location:** `_add_cruncher_to_group`, lines 1049–1053

**Issue:** Uses `ema9.groupby(session_start).last()` — the **last** (end-of-session) EMA value — mapped to every bar in the session. At 10:00 AM, the bar gets the 4:00 PM EMA, which is not known yet.

**Fix:** Use the bar’s own EMA, not the session’s last:

```python
ext_9ema = c - ema9
g["Col_ExtensionFromDaily9EMA_ATR"] = _atr_normalize(g, ext_9ema)
```

Col_ExtensionFromDaily9EMA_Rank inherits this lookahead.

---

### 2. Col_MultiDaySlope_5d — uses today’s closing price intraday

**Location:** `_add_cruncher_to_group`, lines 1059–1063

**Issue:** `daily_close = c.groupby(session_start).last()` — last close per session. At 10:00 AM on day D, `slope_5d` still uses `daily_close[D]`, which is today’s closing price. That value is not known until end of day.

**Fix:** For intraday data, use only **prior** sessions. For example:

```python
daily_close = c.groupby(session_start).last()
sma10 = daily_close.rolling(10, min_periods=1).mean()
slope_5d = (sma10.shift(1) - sma10.shift(6)) / 5  # use prior-day SMA only
g["Col_MultiDaySlope_5d"] = session_start.map(slope_5d)
```

Or require daily data for this column and document accordingly.

---

## Logic Bugs

### 3. Col_InsideDay — wrong definition for minute bars

**Location:** `_add_cruncher_to_group`, lines 1066–1068

**Issue:** “Inside day” = today’s range inside yesterday’s range. Current code uses `h` and `l_`, which are the **current bar’s** high/low, not the day’s high/low so far. For minute bars, each bar has its own H/L; “today’s high so far” should use a cumulative max.

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

**Location:** `_add_cruncher_to_group`, line 981

**Issue:** `g` from groupby may be a view; assigning new columns can trigger SettingWithCopyWarning and can affect the source DataFrame depending on pandas version/usage.

**Fix:** Add `g = g.copy()` at the start of the function (or right after the `len(g) == 0` check).

---

### 5. Col_MomentumDivergence_RSI — scale mismatch (design choice)

**Location:** lines 1114–1117

**Observation:** `price_change` is typically small (e.g. 0.01 for 1%), while `rsi_change` for RSI 50→51 is 0.02. The formula `rsi_change - price_change` mixes scales. Sign/direction is correct for divergence; magnitude is not normalized. Consider documenting or normalizing if magnitudes matter.

---

### 6. Col_CumulativeVol_vs_Avg_Pct — unusual formula

**Location:** `add_volume_vwap_columns`, lines 333–334

```python
minutes_idx = out.index.hour * 60 + out.index.minute + 1
out["Col_CumulativeVol_vs_Avg_Pct"] = _safe_div(out["Volume"].cumsum(), out["Volume"].rolling(390).mean() * minutes_idx) * 100.0
```

**Issue:**  
- `Volume.cumsum()` is unbounded across sessions; for multi-day intraday data this grows without reset.  
- `minutes_idx` starts at 1 at 9:00 and increases; at 9:30 it’s 31. That might not match intended “minutes since open” (e.g. 9:30 open → 0 at 9:30).  
- `rolling(390)` assumes 390 bars per session; may not match actual intraday frequency (e.g. 1-min).

**Recommendation:** Confirm intended semantics and add session-level resets and session-aware window if needed.

---

## Minor — Documentation / Consistency

### 7. Col_ConsecutiveUpBars — off‑by‑one semantics

**Location:** `add_price_action_columns`, line 376

```python
out["Col_ConsecutiveUpBars"] = up_bars.groupby((up_bars.diff().ne(0).cumsum())).cumcount()
```

**Observation:** `cumcount()` is 0-based. The first up bar in a streak yields 0, the second 1, etc. If “consecutive up bars” is meant to include the current bar (e.g. 3 consecutive = 3), this is off by one. Consider `cumcount() + 1` and document.

---

### 8. Col_MinutesSinceOpen — assumes 9:00 open

**Location:** `add_time_columns`, lines 461–462

```python
mins_raw = (out.index.hour - 9) * 60 + np.asarray(out.index.minute)
out["Col_MinutesSinceOpen"] = np.maximum(mins_raw, 0)
```

At 9:30, this gives 30. Many US markets open at 9:30, where “minutes since open” should be 0. Worth documenting or making the open time configurable.

---

### 9. Col_GapFill_15min_Pct — minor edge

**Location:** lines 1012–1016

```python
if pd.isna(gp0) or gp0 <= 0:
    return 0.0 if not pd.isna(gp0) else np.nan
```

If `gp0` is exactly 0 (flat open), this returns 0.0. If `gp0` is NaN, it returns NaN. Correct, but the `return 0.0 if not pd.isna(gp0)` is equivalent to `return 0.0` in the `gp0 <= 0` branch when `gp0` is not NaN. Logic is fine; could be simplified for readability.

---

## Tier 3 Columns — Logic Check

### Col_DistToVWAP_Slope10_ATR

- Session VWAP: `(typical * vol).groupby(session_start).cumsum() / vol.groupby(session_start).cumsum()` — correct.  
- Slope: `session_vwap.rolling(10).mean().diff()` — rolling mean then diff; acceptable momentum measure.  
- No lookahead; ATR normalization handled via `_safe_div`.

### Col_MomentumDivergence_RSI

- `rsi_change - price_change`; inf/nan handling applied.  
- No lookahead; scale mismatch noted above.

### Col_DistFromSessionVWAP_ATR

- Distance from session VWAP, ATR-normalized; no lookahead.  
- Correct when volume is missing (returns NaN).

---

## Summary of Recommended Fixes

| Priority | Item | Action |
|----------|------|--------|
| **Critical** | ExtensionFromDaily9EMA | Use bar-level `ema9`, not session `.last()` |
| **Critical** | MultiDaySlope_5d | Use only prior-session closes for intraday |
| **Logic** | InsideDay | Use `cummax`/`cummin` for day range so far |
| **Medium** | _add_cruncher_to_group | Add `g = g.copy()` at start |
| **Minor** | ConsecutiveUpBars | Consider `cumcount() + 1` and doc |
| **Minor** | Col_CumulativeVol_vs_Avg_Pct | Review and document formula |
| **Minor** | Col_MinutesSinceOpen | Document 9:00 assumption or make configurable |

---

## Test Recommendations

1. Unit test: verify no cruncher column uses future data (e.g. compare with explicitly shifted series).  
2. Sanity test: ExtensionFromDaily9EMA at 10:00 vs end-of-day EMA to confirm fix.  
3. Sanity test: InsideDay on minute data with known inside-day days.  
4. Edge cases: empty groups, single-bar groups, missing volume, NaN in OHLCV.
