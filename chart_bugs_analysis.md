# Chart Bugs Analysis - Code Skeptic Report

## Executive Summary

Analyzed the chart functionality issue where "when opening chart second time, chart shows data for previously opened symbol, but OHLC, LTP and other things in the top of chart is correctly loaded."

---

## PRIMARY BUG: Chart Not Refreshing on Same Ticker

### Symptom
When opening the chart for the **same symbol a second time**, the chart visual (candlesticks) shows data from the **previously opened symbol**, but OHLC, LTP, and header values are correctly updated.

### Root Cause Analysis

**Location**: `static/js/app.js` lines 302-320

The bug is in the `renderChart()` function:

```javascript
const incomingTicker = (data.ticker || "").toUpperCase();
const isNewTicker = window.lastChartTicker !== incomingTicker;

if (isNewTicker) {
    // Only clears data when ticker CHANGES
    Object.values(appState.series).forEach(s => { if(s && s.setData) s.setData([]); });
    if (appState.series.candles) appState.series.candles.setMarkers([]);
    window.lastChartTicker = incomingTicker;
    window.isFirstTickerLoad = true;
}
```

**The Problem**:
1. User opens ticker A (e.g., RELIANCE), `window.lastChartTicker` = "RELIANCE.NS"
2. User closes chart and opens the **same ticker A again**
3. `incomingTicker === window.lastChartTicker` → `isNewTicker` = **FALSE**
4. Old candle data is **NOT cleared**
5. New data gets MERGED/APPENDED to existing data (LightweightCharts default behavior)
6. Result: Mixed/stale data displayed

**Why OHLC/LTP Work Correctly**:
Lines 352-354 unconditionally update the header:
```javascript
document.getElementById('legend-ltp').textContent = `LTP: ₹${last.close.toFixed(2)}`;
document.getElementById('legend-ohlc').textContent = `O: ${last.open.toFixed(2)} H: ${last.high.toFixed(2)} L: ${last.low.toFixed(2)} C: ${last.close.toFixed(2)}`;
```

This runs regardless of ticker change, which is why header updates but chart visual doesn't.

### Fix Required

In `openChart()` function at line 72, add chart clearing BEFORE requesting new data:

```javascript
window.openChart = (ticker) => {
    if (!ticker) return;
    
    // CRITICAL FIX: Clear chart state BEFORE requesting new data
    if (appState.chartInstance) {
        Object.values(appState.series).forEach(s => { 
            if(s && s.setData) s.setData([]); 
        });
        if (appState.series.candles) appState.series.candles.setMarkers([]);
    }
    window.lastChartTicker = "";  // Force isNewTicker = true
    
    const modal = document.getElementById('chart-modal');
    if (modal) modal.style.display = 'flex';
    const title = document.getElementById('chart-title');
    if (title) title.textContent = ticker.replace('.NS', '');
    
    appState.currentChartKey = ticker;
    window.isFirstTickerLoad = true;
    window.safeWSSend({ action: "get_chart_data", ticker });
};
```

---

## SECONDARY ISSUES FOUND

### Issue #2: No Loading State
- **Location**: `openChart()` function
- **Problem**: No visual feedback while waiting for WebSocket response
- **Impact**: User sees stale data until new data arrives, contributes to confusion

### Issue #3: Chart Not Properly Cleaned on Close
- **Location**: `static/js/app.js` lines 84-89
- **Problem**: Only sets references to null but doesn't call `.destroy()` on LightweightCharts instances
- **Impact**: Memory leaks and zombie event listeners

```javascript
window.closeChart = () => {
    const modal = document.getElementById('chart-modal');
    if (modal) modal.style.display = 'none';
    appState.chartInstance = null;
    appState.adxChart = null;
};
```

### Issue #4: Ticker Normalization Inconsistency
- **Locations**: Multiple places - `openChart()`, `renderChart()`, backend handler
- **Problem**: 
  - User input: "RELIANCE"
  - Some places add ".NS": "RELIANCE.NS"  
  - Comparison in `renderChart()` uses `.toUpperCase()` but doesn't ensure consistent ".NS"
- **Impact**: Comparison fails if backend returns "RELIANCE" but `lastChartTicker` is "RELIANCE.NS"

### Issue #5: Global State Usage
- **Location**: `static/js/app.js` line 30
- **Problem**: Uses `window.lastChartTicker` instead of `appState`
- **Impact**: Variables persist across sessions, unexpected behavior

### Issue #6: Interval Switching May Not Clear Properly
- **Location**: `static/js/app.js` lines 167-185
- **Problem**: When switching intervals, doesn't set `window.lastChartTicker = ""`
- **Impact**: Chart data may be appended instead of replaced

### Issue #7: yfinance Cache - No Size Limit
- **Location**: `services/technical_analysis.py` lines 105-125
- **Problem**: Cache never expires (only 5-minute window), no size limit
- **Impact**: Memory grows unbounded in long-running apps

### Issue #8: No Error Boundary for Chart
- **Location**: `renderChart()` function
- **Problem**: No try-catch, failures are silent
- **Impact**: Chart can fail without user feedback

### Issue #9: Live Tick Race Condition
- **Location**: `static/js/app.js` lines 432-459
- **Problem**: If chart data hasn't loaded but live ticks arrive, errors occur
- **Impact**: Console errors

### Issue #10: ADX Chart Resize Not Synced
- **Location**: `static/js/app.js` lines 296-299
- **Problem**: ResizeObserver only watches main container, not adxContainer
- **Impact**: ADX chart may not resize properly

---

## Summary Table

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | Chart not refreshing on same ticker | **CRITICAL** | app.js:302-320 |
| 2 | No loading state | HIGH | openChart() |
| 3 | Chart not destroyed on close | MEDIUM | app.js:84-89 |
| 4 | Ticker normalization | MEDIUM | Multiple |
| 5 | Global state usage | LOW | app.js:30 |
| 6 | Interval switch issues | MEDIUM | app.js:167-185 |
| 7 | yfinance cache | LOW | technical_analysis.py:105 |
| 8 | No error boundary | MEDIUM | renderChart() |
| 9 | Live tick race | LOW | app.js:432-459 |
| 10 | ADX resize | LOW | app.js:296-299 |

---

## Recommended Fix Priority

1. **IMMEDIATE**: Fix Issue #1 - Add chart clearing in `openChart()` 
2. **SOON**: Add loading states (#2) and proper cleanup (#3)
3. **LATER**: Address normalization (#4) and error boundaries (#8)
