/**
 * V3 SuperNova Dashboard Engine
 * --------------------------------------------------
 * RESTORATION, ENHANCEMENT & DE-DUPLICATION (Update 16)
 */

// ─── Global State & Exports ──────────────────────────────────────────
const appState = {
    openTrades: [], closedTrades: [], globalContext: {}, marketPhase: {},
    aiAdvisor: null, aiScansToday: [], actionTimeline: [], aiCallsToday: 0, aiCallsLimit: 7,
    chartInstance: null, adxChart: null, currentChartData: null,
    currentChartKey: null, currentInterval: '5m',
    // Rate limit tracking
    isRateLimited: false,
    rateLimitCooldown: 0,
    lastDataUpdate: null,
    isDataStale: false,
    series: { candles: null, adxLine: null, rsiLine: null, stLine: null, knnUp: null, knnDown: null, projLine: null, upperBand: null, lowerBand: null, vwap: null },
    indicatorSettings: {
        st: { atr_period: 10, factor: 3.0, training_len: 100, p_low: 0.25, p_med: 0.5, p_high: 0.75 },
        lz: { k: 8, lookback: 2000, threshold: 0.5, use_volatility: true },
        knn: { k: 5, sequence_length: 15, window: 200 }
    }
};
window.appState = appState;
window.openChart = openChart;
window.closeChart = closeChart;
window.changeChartInterval = changeChartInterval;
window.toggleIndicator = toggleIndicator;
window.toggleIndicatorSettings = toggleIndicatorSettings;
window.saveIndicatorSettings = saveIndicatorSettings;
window.resetIndicatorSettings = resetIndicatorSettings;

let chartRefreshTimer = null;

const DEFAULT_INDICATOR_SETTINGS = {
    st: { atr_period: 10, factor: 3.0, training_len: 100, p_low: 0.25, p_med: 0.5, p_high: 0.75 },
    lz: { k: 8, lookback: 2000, threshold: 0.5, use_volatility: true },
    knn: { k: 5, sequence_length: 15, window: 200 }
};

function resetIndicatorSettings(indicator) {
    if (DEFAULT_INDICATOR_SETTINGS[indicator]) {
        appState.indicatorSettings[indicator] = JSON.parse(JSON.stringify(DEFAULT_INDICATOR_SETTINGS[indicator]));
        toggleIndicatorSettings(indicator); // Re-populate UI
        saveIndicatorSettings(indicator); // Apply and refresh
        showToast(`${indicator.toUpperCase()} settings reset to defaults`, 'info');
    }
}

// ─── AI Models Config ────────────────────────────────────────────────
const AI_MODELS = {
    google: [
        {value: "gemini-3.1-pro", label: "Gemini 3.1 Pro (Deep Reasoning)"},
        {value: "gemini-3-flash", label: "Gemini 3 Flash (Fast & Balanced)"},
        {value: "gemini-2.5-pro", label: "Gemini 2.5 Pro (Large Context)"}
    ],
    groq: [
        {value: "deepseek-r1-distill-llama-70b", label: "DeepSeek R1 (Complex Logic / Fast)"},
        {value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B (High Precision)"},
        {value: "llama-3.1-8b-instant", label: "Llama 3.1 8B (High Freq / Cheap)"}
    ],
    sambanova: [
        {value: "Meta-Llama-3.3-70B-Instruct", label: "Llama 3.3 70B (Deep Scan)"},
        {value: "Meta-Llama-3.1-405B-Instruct", label: "Llama 3.1 405B (Ultra Accuracy)"},
        {value: "Meta-Llama-3.1-8B-Instruct", label: "Llama 3.1 8B (Fast Analysis)"}
    ]
};

function updateModelDropdown() {
    const provider = document.getElementById('ai-provider-input')?.value;
    const modelSelect = document.getElementById('ai-model-input');
    if (provider && modelSelect && AI_MODELS[provider]) {
        modelSelect.innerHTML = AI_MODELS[provider].map(m => `<option value="${m.value}">${m.label}</option>`).join('');
    }
}

// ─── Utility Functions ──────────────────────────────────────────────
function showToast(msg, level='info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const t = document.createElement('div');
    t.className = `toast toast-${level}`;
    t.textContent = msg;
    container.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function signalClass(signal) {
    const s = (signal || '').toUpperCase();
    if (s.includes('STRONG BUY')) return 'signal-strong-buy';
    if (s.includes('BUY')) return 'signal-buy';
    if (s.includes('STRONG SHORT')) return 'signal-strong-short';
    if (s.includes('SHORT')) return 'signal-short';
    return 'signal-neutral';
}

function rsiColor(rsi) {
    if (rsi >= 70) return '#ef4444'; // Overbought
    if (rsi <= 30) return '#22c55e'; // Oversold
    return '#8b949e';
}

function fmt(val) {
    if (val === undefined || val === null) return '--';
    return Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function sentimentClass(label) {
    const l = (label || '').toLowerCase();
    if (l.includes('bullish')) return 'positive';
    if (l.includes('bearish')) return 'negative';
    return 'neutral';
}

/**
 * Updates the data provider status badges in the settings panel.
 */
function updateDataProviderStatus(provider, isUsingFallback) {
    const activeBadge = document.getElementById('provider-active-badge');
    const fallbackBadge = document.getElementById('provider-fallback-badge');
    
    if (activeBadge) {
        activeBadge.textContent = '● ' + (provider === 'upstox' ? 'Upstox' : 'YFinance');
    }
    
    if (fallbackBadge) {
        if (isUsingFallback) {
            fallbackBadge.style.display = 'inline-block';
            fallbackBadge.textContent = '↻ Using Fallback';
        } else {
            fallbackBadge.style.display = 'none';
        }
    }
}

/**
 * Updates the data provider quota display in the settings panel.
 */
function updateDataProviderQuota(quota) {
    const upstoxQuotaEl = document.getElementById('upstox-quota');
    const yfinanceQuotaEl = document.getElementById('yfinance-quota');
    
    if (upstoxQuotaEl && quota.upstox) {
        upstoxQuotaEl.textContent = '∞';
    }
    
    if (yfinanceQuotaEl && quota.yfinance) {
        const yf = quota.yfinance;
        const used = yf.limit_rpd - yf.remaining_rpd;
        const pct = yf.used_rpd_pct || 0;
        yfinanceQuotaEl.textContent = `${used}/${yf.limit_rpd}`;
        
        // Apply color classes based on usage
        yfinanceQuotaEl.classList.remove('warning', 'danger');
        if (pct >= 90) {
            yfinanceQuotaEl.classList.add('danger');
        } else if (pct >= 70) {
            yfinanceQuotaEl.classList.add('warning');
        }
    }
}

/**
 * Updates rate limit status display in the UI.
 * Shows a warning badge when rate limited with countdown timer.
 */
function updateRateLimitStatus(status) {
    appState.isRateLimited = status.is_rate_limited || false;
    appState.rateLimitCooldown = status.remaining_cooldown || 0;
    
    // Find or create rate limit indicator
    let rateLimitEl = document.getElementById('rate-limit-indicator');
    if (!rateLimitEl) {
        // Create the element if it doesn't exist
        const header = document.querySelector('.header-actions') || document.querySelector('.top-bar');
        if (header) {
            rateLimitEl = document.createElement('div');
            rateLimitEl.id = 'rate-limit-indicator';
            rateLimitEl.className = 'rate-limit-badge';
            rateLimitEl.style.cssText = 'display: none; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-left: 10px;';
            header.appendChild(rateLimitEl);
        }
    }
    
    if (rateLimitEl) {
        if (appState.isRateLimited) {
            rateLimitEl.style.display = 'inline-block';
            rateLimitEl.style.background = 'rgba(239, 68, 68, 0.2)';
            rateLimitEl.style.color = '#ef4444';
            rateLimitEl.textContent = `⏳ Rate Limited (${appState.rateLimitCooldown}s)`;
            
            // Add stale data indicator to chart if visible
            const chartContainer = document.getElementById('chart-container');
            if (chartContainer) {
                let staleBadge = document.getElementById('stale-data-badge');
                if (!staleBadge) {
                    staleBadge = document.createElement('div');
                    staleBadge.id = 'stale-data-badge';
                    staleBadge.className = 'stale-badge';
                    staleBadge.style.cssText = 'position: absolute; top: 10px; right: 10px; padding: 4px 8px; background: rgba(251, 191, 36, 0.2); color: #fbbf24; border-radius: 4px; font-size: 11px; font-weight: 600; z-index: 10;';
                    staleBadge.textContent = '📊 Cached Data';
                    chartContainer.parentElement.style.position = 'relative';
                    chartContainer.parentElement.appendChild(staleBadge);
                }
                staleBadge.style.display = 'block';
            }
        } else {
            rateLimitEl.style.display = 'none';
            
            // Hide stale data badge
            const staleBadge = document.getElementById('stale-data-badge');
            if (staleBadge) {
                staleBadge.style.display = 'none';
            }
        }
    }
    
    // Update app state for data freshness
    appState.isDataStale = appState.isRateLimited;
}

/**
 * Updates the data freshness indicator based on last update time.
 */
function updateDataFreshness(lastUpdate) {
    if (!lastUpdate) {
        appState.lastDataUpdate = null;
        return;
    }
    
    appState.lastDataUpdate = lastUpdate;
    const lastUpdateEl = document.getElementById('last-update-time');
    if (lastUpdateEl) {
        const date = new Date(lastUpdate);
        lastUpdateEl.textContent = date.toLocaleTimeString('en-IN');
    }
}

// ─── WebSocket Initialization ───────────────────────────────────────
const ws = new WebSocket(`ws://${window.location.host}/ws`);

/**
 * Opens the chart modal for a specific ticker and requests data.
 * Requirement 1: Core charting logic at the top.
 * Requirement 10: id='chart-title' used.
 */
function openChart(ticker) {
    if (!ticker) return;
    
    // CRITICAL FIX: Clear chart data BEFORE requesting new data
    // This fixes the bug where reopening same ticker shows old data
    if (appState.chartInstance) {
        Object.values(appState.series).forEach(s => { 
            if(s && s.setData) s.setData([]); 
        });
        if (appState.series.candles) appState.series.candles.setMarkers([]);
    }
    // Also clear the global lastChartTicker to force fresh load
    window.lastChartTicker = "";
    
    // SKEPTIC: Robust ticker cleaning for Search Lookups
    let cleanTicker = ticker.toUpperCase();
    if (!cleanTicker.includes('|') && !cleanTicker.startsWith('^') && !cleanTicker.endsWith('.NS')) {
        cleanTicker += '.NS';
    }
    appState.currentChartKey = cleanTicker;
    
    // Reset state for new chart
    appState.currentChartData = { ohlc: [] };
    appState.isFirstLoad = true; // For zoom handling
    
    const modal = document.getElementById('chart-modal');
    if (modal) {
        modal.style.display = 'flex';
        // Add animation class if desired
        modal.classList.add('active');
    }

    // Update title immediately
    const titleEl = document.getElementById('chart-title');
    if (titleEl) {
        titleEl.innerHTML = `📈 <span id="ticker-display">${cleanTicker.replace('.NS', '')}</span> <span class="v3-tag">V3 Vision</span>`;
    }

    // Clear previous legend values
    ['legend-ltp', 'legend-ohlc', 'legend-adx', 'legend-rsi', 'legend-st', 'legend-lz', 'legend-knn', 'legend-vwap'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = id.split('-')[1].toUpperCase() + ': --';
    });

    // Initial data request
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'get_chart_data',
            ticker: cleanTicker,
            interval: appState.currentInterval,
            params: appState.indicatorSettings
        }));
    }

    // Setup auto-refresh for the chart (every 60s)
    if (chartRefreshTimer) clearInterval(chartRefreshTimer);
    chartRefreshTimer = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN && appState.currentChartKey) {
            ws.send(JSON.stringify({
                action: 'get_chart_data',
                ticker: appState.currentChartKey,
                interval: appState.currentInterval,
                params: appState.indicatorSettings
            }));
        }
    }, 60000);
}

/**
 * Closes the chart modal and cleans up resources.
 */
function closeChart() {
    const modal = document.getElementById('chart-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.classList.remove('active');
    }
    
    const container = document.getElementById('chart-container');
    const adxContainer = document.getElementById('adx-container');
    if (container) container.innerHTML = '';
    if (adxContainer) adxContainer.innerHTML = '';
    
    // Clear references
    appState.chartInstance = null;
    appState.adxChart = null;
    appState.series = {
        candles: null, adxLine: null, rsiLine: null, 
        stUp: null, stDown: null, knnUp: null, knnDown: null,
        projLine: null, upperBand: null, lowerBand: null
    };
    
    // Stop refresh timer
    if (chartRefreshTimer) {
        clearInterval(chartRefreshTimer);
        chartRefreshTimer = null;
    }
}

/**
 * Changes the chart timeframe/interval.
 */
function changeChartInterval(interval) {
    // SKEPTIC: Lock zoom IMMEDIATELY to prevent fitContent race
    appState.isFirstLoad = false;
    appState.isIntervalSwitching = true;
    appState.expectedInterval = interval;
    appState.currentInterval = interval;
    
    // Update UI buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
        if (btn.getAttribute('data-interval') === interval) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    if (appState.currentChartKey && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'get_chart_data',
            ticker: appState.currentChartKey,
            interval: interval,
            params: appState.indicatorSettings
        }));
    }
}

/**
 * Toggles indicator visibility and triggers a re-render if needed.
 * Requirement 7: toggleIndicator re-renders the chart.
 */
function toggleIndicator(indicator, show) {
    if (appState.currentChartData) {
        renderChart(appState.currentChartData);
    }
}

/**
 * Open settings modal for specific ML indicators.
 */
function toggleIndicatorSettings(indicator) {
    const modal = document.getElementById(`${indicator}-settings`);
    if (modal) {
        modal.style.display = 'flex';
        // Populate inputs from state
        if (indicator === 'st') {
            document.getElementById('st-atr-len').value = appState.indicatorSettings.st.atr_period;
            document.getElementById('st-factor').value = appState.indicatorSettings.st.factor;
            document.getElementById('st-train-len').value = appState.indicatorSettings.st.training_len;
            document.getElementById('st-p-low').value = appState.indicatorSettings.st.p_low;
            document.getElementById('st-p-med').value = appState.indicatorSettings.st.p_med;
            document.getElementById('st-p-high').value = appState.indicatorSettings.st.p_high;
        } else if (indicator === 'lz') {
            document.getElementById('lz-k').value = appState.indicatorSettings.lz.k;
            document.getElementById('lz-lookback').value = appState.indicatorSettings.lz.lookback;
            document.getElementById('lz-threshold').value = appState.indicatorSettings.lz.threshold;
            document.getElementById('lz-use-vol').checked = appState.indicatorSettings.lz.use_volatility;
        } else if (indicator === 'knn') {
            document.getElementById('knn-k').value = appState.indicatorSettings.knn.k;
            document.getElementById('knn-seq-len').value = appState.indicatorSettings.knn.sequence_length;
            document.getElementById('knn-window').value = appState.indicatorSettings.knn.window;
        }
    } else {
        showToast(`${indicator.toUpperCase()} settings configuration coming in next update.`, 'info');
    }
}

function saveIndicatorSettings(indicator) {
    if (indicator === 'st') {
        appState.indicatorSettings.st = {
            atr_period: parseInt(document.getElementById('st-atr-len').value),
            factor: parseFloat(document.getElementById('st-factor').value),
            training_len: parseInt(document.getElementById('st-train-len').value),
            p_low: parseFloat(document.getElementById('st-p-low').value),
            p_med: parseFloat(document.getElementById('st-p-med').value),
            p_high: parseFloat(document.getElementById('st-p-high').value)
        };
    } else if (indicator === 'lz') {
        appState.indicatorSettings.lz = {
            k: parseInt(document.getElementById('lz-k').value),
            lookback: parseInt(document.getElementById('lz-lookback').value),
            threshold: parseFloat(document.getElementById('lz-threshold').value),
            use_volatility: document.getElementById('lz-use-vol').checked
        };
    } else if (indicator === 'knn') {
        appState.indicatorSettings.knn = {
            k: parseInt(document.getElementById('knn-k').value),
            sequence_length: parseInt(document.getElementById('knn-seq-len').value),
            window: parseInt(document.getElementById('knn-window').value)
        };
    }

    document.getElementById(`${indicator}-settings`).style.display = 'none';
    showToast(`${indicator.toUpperCase()} settings applied`, 'success');

    // Trigger refresh
    if (appState.currentChartKey && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'get_chart_data',
            ticker: appState.currentChartKey,
            interval: appState.currentInterval,
            params: appState.indicatorSettings
        }));
    }
}

/**
 * Core rendering engine for Lightweight Charts.
 * Requirement 6: Markers sorted by time.
 */
function initChartInstance(container, adxContainer) {
    const chartOptions = {
        width: container.clientWidth,
        height: 450,
        layout: {
            background: { type: 'solid', color: '#0b0e11' },
            textColor: '#8b949e',
            fontSize: 12,
            fontFamily: 'Inter, sans-serif'
        },
        grid: {
            vertLines: { color: 'rgba(139,148,158,0.06)' },
            horzLines: { color: 'rgba(139,148,158,0.06)' }
        },
        crosshair: {
            mode: 1,
            vertLine: { labelVisible: true, color: 'rgba(139,148,158,0.5)', style: 2 },
            horzLine: { labelVisible: true, color: 'rgba(139,148,158,0.5)', style: 2 }
        },
        rightPriceScale: { borderColor: 'rgba(139,148,158,0.2)', autoScale: true },
        timeScale: {
            borderColor: 'rgba(139,148,158,0.2)',
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 50, // 35% right margin (approx 50 bars)
            barSpacing: 6,
            shiftVisibleRangeOnNewBar: false, // MANDATORY: Prevent chart from jumping to latest bar on new tick
            lockVisibleTimeRangeOnResize: true
        }
    };

    appState.chartInstance = LightweightCharts.createChart(container, chartOptions);

    appState.series.candles = appState.chartInstance.addCandlestickSeries({
        upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
        wickUpColor: '#22c55e', wickDownColor: '#ef4444',
        autoscaleInfoProvider: (orig) => {
            if (appState.isIntervalSwitching) return null;
            return orig();
        }
    });
    
    // Auxiliary Series (SKEPTIC: Explicitly hide Y-axis labels to prevent clutter)
    const auxOptions = {
        lineWidth: 2, title: '', 
        priceLineVisible: false, lastValueVisible: false,
        axisLabelVisible: false
    };

    appState.series.stLine = appState.chartInstance.addLineSeries({ 
        lineWidth: 2, 
        priceLineVisible: false, 
        lastValueVisible: false,
        axisLabelVisible: false,
        title: 'ST' 
    });
    appState.series.knnUp = appState.chartInstance.addLineSeries({ ...auxOptions, color: '#a855f7' });
    appState.series.knnDown = appState.chartInstance.addLineSeries({ ...auxOptions, color: '#f97316' });
    appState.series.vwap = appState.chartInstance.addLineSeries({ 
        color: 'rgba(255, 193, 7, 0.8)', 
        lineWidth: 1, 
        lineStyle: 3, 
        title: 'VWAP',
        lastValueVisible: false,
        priceLineVisible: false,
        axisLabelVisible: false,
        autoscaleInfoProvider: () => null
    });


    appState.adxChart = LightweightCharts.createChart(adxContainer, {
        ...chartOptions, height: 150,
        crosshair: { ...chartOptions.crosshair },
        timeScale: { 
            ...chartOptions.timeScale, 
            visible: true, 
            borderVisible: false
        }
    });
    
    // SKEPTIC: Ensure timeScale is visible for tooltips with minimalist labels
    appState.adxChart.applyOptions({
        timeScale: {
            visible: true,
            ticksVisible: true,
            borderVisible: false,
            shiftVisibleRangeOnNewBar: true
        }
    });

    appState.series.adxLine = appState.adxChart.addLineSeries({ color: '#facc15', lineWidth: 2, title: 'ADX' });
    appState.series.rsiLine = appState.adxChart.addLineSeries({ color: '#60a5fa', lineWidth: 1, title: 'RSI' });

    // Sync TimeScales (Scrolling)
    appState.chartInstance.timeScale().subscribeVisibleTimeRangeChange(range => {
        if (range && range.from && range.to && appState.adxChart) {
            try {
                appState.adxChart.timeScale().setVisibleRange(range);
            } catch(e) { /* Defensive catch for sync jitter */ }
        }
    });
    appState.adxChart.timeScale().subscribeVisibleTimeRangeChange(range => {
        if (range && range.from && range.to && appState.chartInstance) {
            try {
                appState.chartInstance.timeScale().setVisibleRange(range);
            } catch(e) { /* Defensive catch for sync jitter */ }
        }
    });

    // Sync Crosshairs (Bidirectional)
    appState.chartInstance.subscribeCrosshairMove(param => {
        if (!appState.adxChart) return;
        if (!param.time || !param.point) {
            appState.adxChart.setCrosshairPosition(undefined, undefined, appState.series.adxLine);
            return;
        }
        appState.adxChart.setCrosshairPosition(undefined, param.time, appState.series.adxLine);
    });
    appState.adxChart.subscribeCrosshairMove(param => {
        if (!appState.chartInstance) return;
        if (!param.time || !param.point) {
            appState.chartInstance.setCrosshairPosition(undefined, undefined, appState.series.candles);
            return;
        }
        appState.chartInstance.setCrosshairPosition(undefined, param.time, appState.series.candles);
    });

    if (!window.chartResizeObserver) {
        window.chartResizeObserver = new ResizeObserver(() => {
            const container = document.getElementById('chart-container');
            if (container && appState.chartInstance) {
                appState.chartInstance.applyOptions({ width: container.clientWidth });
            }
            if (container && appState.adxChart) {
                appState.adxChart.applyOptions({ width: container.clientWidth });
            }
        });
        window.chartResizeObserver.observe(container);
    }
}

function verifySignalAccuracy(ohlc, markers) {
    if (!ohlc || ohlc.length < 20 || !markers || markers.length === 0) return null;
    
    let total = 0;
    let successful = 0;
    const horizon = 5; // 5 bars lookahead
    
    markers.forEach(m => {
        if (!m.text || (!m.text.includes('BUY') && !m.text.includes('SELL'))) return;
        
        const idx = ohlc.findIndex(c => c.time === m.time);
        if (idx === -1 || idx + horizon >= ohlc.length) return;
        
        const entryPrice = ohlc[idx].close;
        const exitPrice = ohlc[idx + horizon].close;
        
        total++;
        if (m.text.includes('BUY')) {
            if (exitPrice > entryPrice) successful++;
        } else if (m.text.includes('SELL')) {
            if (exitPrice < entryPrice) successful++;
        }
    });
    
    if (total === 0) return null;
    const accuracy = (successful / total) * 100;
    console.log(`SKEPTIC: Signal Accuracy Audit -> ${accuracy.toFixed(1)}% (${successful}/${total} successful)`);
    return accuracy;
}

function renderChart(data) {
    if (!data || data.error) {
        showToast(data?.error || "Chart data unavailable", "error");
        return;
    }
    
    // Handle stale data indicator from rate limiting
    if (data.is_stale_data) {
        const staleBadge = document.getElementById('stale-data-badge');
        if (staleBadge) {
            staleBadge.style.display = 'block';
        }
        appState.isDataStale = true;
    }

    const container = document.getElementById('chart-container');
    const adxContainer = document.getElementById('adx-container');
    if (!container || !adxContainer) return;

    appState.currentChartData = data;

    if (!appState.chartInstance) {
        initChartInstance(container, adxContainer);
    }

    // Process OHLC Data (Shifted to IST)
    let ohlc = (data.ohlc || []).map(c => ({
        time: Number(c.time) + 19800, // UTC to IST
        open: parseFloat(c.open),
        high: parseFloat(c.high),
        low: parseFloat(c.low),
        close: parseFloat(c.close)
    })).filter(c => c.time > 19800 && c.open > 0).sort((a, b) => a.time - b.time);

    // De-duplicate time points
    const uniqueOhlc = [];
    let lastTime = -1;
    for (const c of ohlc) {
        if (c.time > lastTime) {
            uniqueOhlc.push(c);
            lastTime = c.time;
        }
    }
    ohlc = uniqueOhlc;

    if (ohlc.length > 0) {
        // SKEPTIC: Manually preserve bar count (Logical Range) if not first load
        const timeScale = appState.chartInstance.timeScale();
        const oldLogicalRange = (!appState.isFirstLoad) ? timeScale.getVisibleLogicalRange() : null;

        appState.series.candles.setData(ohlc);
        
        // SKEPTIC: Restore logical range immediately after data set
        if (oldLogicalRange) {
            setTimeout(() => {
                try {
                    // Force the logical range (number of bars) ONCE
                    timeScale.setVisibleLogicalRange(oldLogicalRange);
                    if (appState.adxChart) appState.adxChart.timeScale().setVisibleLogicalRange(oldLogicalRange);

                    if (data.interval === appState.expectedInterval) {
                        appState.isIntervalSwitching = false;
                    }
                } catch(e) { /* Defensive */ }
            }, 50);
        } else {            if (data.interval === appState.expectedInterval) {
                appState.isIntervalSwitching = false;
            }
        }

        const last = ohlc[ohlc.length - 1];
        document.getElementById('legend-ltp').textContent = `LTP: ₹${last.close.toFixed(2)}`;
        document.getElementById('legend-ohlc').textContent = `O: ${last.open.toFixed(2)} H: ${last.high.toFixed(2)} L: ${last.low.toFixed(2)} C: ${last.close.toFixed(2)}`;
    }

    // Indicator Checkboxes
    const showLZ = document.getElementById('toggle-lz')?.checked !== false;
    const showST = document.getElementById('toggle-st')?.checked !== false;
    const showKNN = document.getElementById('toggle-knn')?.checked !== false;

    // ADX & RSI
    if (data.adx_series) {
        const adx = data.adx_series.map(p => ({ time: Number(p.time) + 19800, value: parseFloat(p.value) }))
            .filter(p => p.time > 19800).sort((a, b) => a.time - b.time);
        appState.series.adxLine.setData(adx);
        if (adx.length > 0) document.getElementById('legend-adx').textContent = `ADX: ${adx[adx.length - 1].value.toFixed(1)}`;
    }
    if (data.rsi_series) {
        const rsi = data.rsi_series.map(p => ({ time: Number(p.time) + 19800, value: parseFloat(p.value) }))
            .filter(p => p.time > 19800).sort((a, b) => a.time - b.time);
        appState.series.rsiLine.setData(rsi);
        if (rsi.length > 0) document.getElementById('legend-rsi').textContent = `RSI: ${rsi[rsi.length - 1].value.toFixed(1)}`;
    }

    // Indicators & Markers
    const combinedMarkers = [];
    const validTimes = new Set(ohlc.map(c => c.time));

    // 1. Lorentzian Signals
    if (showLZ && data.ml_lorentzian) {
        data.ml_lorentzian.filter(l => l.signal !== 0).forEach(l => {
            const time = Number(l.time) + 19800;
            if (validTimes.has(time)) {
                combinedMarkers.push({
                    time,
                    position: l.signal === 1 ? 'belowBar' : 'aboveBar',
                    color: l.signal === 1 ? '#22c55e' : '#ef4444',
                    shape: l.signal === 1 ? 'arrowUp' : 'arrowDown',
                    text: l.signal === 1 ? 'L-BUY' : 'L-SELL'
                });
            }
        });
        if (data.ml_lorentzian.length > 0) {
            document.getElementById('legend-lz').textContent = `LZ: ${parseFloat(data.ml_lorentzian[data.ml_lorentzian.length - 1].score).toFixed(2)}`;
        }
    }

    // 2. Adaptive SuperTrend (AlgoAlpha Style: Single line with dynamic coloring)
    if (data.ml_adaptive_st && data.ml_adaptive_st.time) {
        const trend = data.ml_adaptive_st.trend;
        const stTimes = data.ml_adaptive_st.time;
        const stVals = data.ml_adaptive_st.value;
        const regimes = data.ml_adaptive_st.regime || [];
        
        const stPoints = [];
        for (let i = 0; i < stTimes.length; i++) {
            const time = Number(stTimes[i]) + 19800;
            const tr = trend[i];
            const val = parseFloat(stVals[i]);
            
            stPoints.push({ 
                time, 
                value: val,
                color: tr === 1 ? '#22c55e' : '#ef4444' 
            });

            // Regime Markers (1, 2, 3) - Show at flips or every 30 bars
            if (showST && (i === 0 || trend[i] !== trend[i - 1] || i % 30 === 0)) {
                if (validTimes.has(time)) {
                    combinedMarkers.push({
                        time,
                        position: tr === 1 ? 'belowBar' : 'aboveBar',
                        color: tr === 1 ? '#22c55e' : '#ef4444',
                        shape: tr === 1 ? 'arrowUp' : 'arrowDown',
                        text: regimes[i] ? regimes[i].toString() : ''
                    });
                }
            }
        }
        
        if (appState.series.stLine) {
            appState.series.stLine.setData(stPoints.sort((a, b) => a.time - b.time));
            appState.series.stLine.applyOptions({ visible: showST });
        }

        // Update Legend
        if (stVals.length > 0) {
            const lastST = stVals[stVals.length - 1];
            const stEl = document.getElementById('legend-st');
            if (stEl) stEl.textContent = `ST: ${parseFloat(lastST).toFixed(1)}`;
        }
    }

    // 3. KNN Trend (TV Parity: Color-changing line based on trend)
    if (data.ml_knn) {
        console.log(`SKEPTIC: KNN Data Arrival -> ${data.ml_knn.length} points`);
        console.log(`SKEPTIC: KNN Latest Sample ->`, data.ml_knn.slice(-1)[0]);
        const knnData = data.ml_knn.map(p => ({
            time: Number(p.time) + 19800,
            value: parseFloat(p.value),
            trend: parseInt(p.trend),
            marker: parseInt(p.marker)
        })).sort((a, b) => a.time - b.time);

        const upPoints = [];
        const downPoints = [];

        for (let i = 0; i < knnData.length; i++) {
            const pt = knnData[i];
            
            // Collect Markers (K-BUY / K-SELL)
            if (showKNN && pt.marker !== 0 && !isNaN(pt.marker)) {
                if (validTimes.has(pt.time)) {
                    combinedMarkers.push({
                        time: pt.time,
                        position: pt.marker === 1 ? 'belowBar' : 'aboveBar',
                        color: pt.marker === 1 ? '#a855f7' : '#f97316',
                        shape: pt.marker === 1 ? 'arrowUp' : 'arrowDown',
                        text: pt.marker === 1 ? 'K-BUY' : 'K-SELL'
                    });
                }
            }

            if (pt.trend === 1) {
                upPoints.push({ time: pt.time, value: pt.value });
                if (i > 0 && knnData[i - 1].trend === -1) {
                    downPoints.push({ time: pt.time, value: pt.value }); // Transition anchor
                }
            } else if (pt.trend === -1) {
                downPoints.push({ time: pt.time, value: pt.value });
                if (i > 0 && knnData[i - 1].trend === 1) {
                    upPoints.push({ time: pt.time, value: pt.value }); // Transition anchor
                }
            }
        }

        if (appState.series.knnUp) {
            appState.series.knnUp.setData(upPoints.sort((a, b) => a.time - b.time));
            appState.series.knnUp.applyOptions({ visible: showKNN, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
        }
        if (appState.series.knnDown) {
            appState.series.knnDown.setData(downPoints.sort((a, b) => a.time - b.time));
            appState.series.knnDown.applyOptions({ visible: showKNN, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
        }

        // Update Legend
        if (knnData.length > 0) {
            const last = knnData[knnData.length - 1];
            const knnEl = document.getElementById('legend-knn');
            if (knnEl) {
                const signal = last.trend === 1 ? 'BUY' : (last.trend === -1 ? 'SELL' : 'NEUTRAL');
                knnEl.textContent = `KNN: ${signal}`;
                console.log(`SKEPTIC: KNN Legend Updated -> ${knnEl.textContent}`);
            }
        }
    } else {
        console.warn("SKEPTIC: No ml_knn data in payload");
    }

    // 4. 3PM Price Projection
    if (data.projection && data.proj_timestamps) {
        const proj = data.proj_timestamps.map((t, i) => ({
            time: Number(t) + 19800,
            value: parseFloat(data.projection[i])
        })).sort((a, b) => a.time - b.time);
        
        if (!appState.series.projLine) {
            appState.series.projLine = appState.chartInstance.addLineSeries({ 
                color: '#6366f1', lineWidth: 2, lineStyle: 2, title: '',
                autoscaleInfoProvider: () => ({
                    priceRange: { min: Math.min(...data.projection), max: Math.max(...data.projection) },
                    margins: { above: 10, below: 10 }
                }),
                lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false
            });
        }
        appState.series.projLine.setData(proj);
        appState.series.projLine.applyOptions({ visible: true });

        if (data.upper_band) {
            const upper = data.proj_timestamps.map((t, i) => ({ time: Number(t) + 19800, value: parseFloat(data.upper_band[i]) })).sort((a, b) => a.time - b.time);
            if (!appState.series.upperBand) appState.series.upperBand = appState.chartInstance.addLineSeries({ color: 'rgba(99,102,241,0.1)', lineWidth: 1, lineStyle: 3, autoscaleInfoProvider: () => null, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
            appState.series.upperBand.setData(upper);
            appState.series.upperBand.applyOptions({ visible: true, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
        }
        if (data.lower_band) {
            const lower = data.proj_timestamps.map((t, i) => ({ time: Number(t) + 19800, value: parseFloat(data.lower_band[i]) })).sort((a, b) => a.time - b.time);
            if (!appState.series.lowerBand) appState.series.lowerBand = appState.chartInstance.addLineSeries({ color: 'rgba(99,102,241,0.1)', lineWidth: 1, lineStyle: 3, autoscaleInfoProvider: () => null, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
            appState.series.lowerBand.setData(lower);
            appState.series.lowerBand.applyOptions({ visible: true, lastValueVisible: false, priceLineVisible: false, axisLabelVisible: false });
        }
    }

    // Ensure markers are sorted by time.
    combinedMarkers.sort((a, b) => a.time - b.time);

    // Filter duplicates at the same time point
    const uniqueMarkers = [];
    let lastMT = -1;
    for (const m of combinedMarkers) {
        if (m.time > lastMT) {
            uniqueMarkers.push({ ...m, text: String(m.text || '') });
            lastMT = m.time;
        } else if (m.time === lastMT && uniqueMarkers.length > 0) {
            const last = uniqueMarkers[uniqueMarkers.length - 1];
            if (m.text) {
                last.text = last.text ? `${last.text} | ${m.text}` : String(m.text);
            }
        }
    }

    // SKEPTIC: Manually preserve range if not first load
    const timeScale = appState.chartInstance.timeScale();
    const oldRange = !appState.isFirstLoad ? timeScale.getVisibleRange() : null;

    appState.series.candles.setMarkers(uniqueMarkers);
    
    // SKEPTIC: Restore range immediately after marker set
    if (oldRange && oldRange.from && oldRange.to) {
        try {
            timeScale.setVisibleRange(oldRange);
        } catch(e) { /* Defensive */ }
    }
    
    // SKEPTIC: Accuracy Audit
    verifySignalAccuracy(ohlc, uniqueMarkers);

    // VWAP Legend
    if (data.vwap) {
        document.getElementById('legend-vwap').textContent = `VWAP: ₹${parseFloat(data.vwap).toFixed(2)}`;
        if (appState.series.vwap && ohlc.length > 0) {
            // Draw a horizontal line from start to end of OHLC
            const vwapVal = parseFloat(data.vwap);
            appState.series.vwap.setData([
                { time: ohlc[0].time, value: vwapVal },
                { time: ohlc[ohlc.length - 1].time, value: vwapVal }
            ]);
        }
    }

    // SKEPTIC: Strict Zoom handling. Only fitContent on the absolute first data arrival for this ticker.
    const incomingTicker = data.ticker ? data.ticker.toUpperCase() : "";
    const activeTicker = appState.currentChartKey ? appState.currentChartKey.toUpperCase() : "";
    
    if (appState.isFirstLoad && ohlc.length > 0 && 
        incomingTicker === activeTicker) {
        
        // SKEPTIC: Focus on TODAY only (9:15 AM IST onwards)
        const now = new Date();
        const today915 = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 9, 15, 0).getTime() / 1000;
        const ist915 = today915 + 19800; // Local IST epoch
        
        const lastTime = ohlc[ohlc.length - 1].time;
        
        // If we have data today, focus from 9:15 to latest
        if (lastTime > ist915) {
            console.log(`SKEPTIC: Focusing today from 9:15 AM IST`);
            appState.chartInstance.timeScale().setVisibleRange({
                from: ist915,
                to: lastTime + (60 * 300) // 5 hours buffer for projections and scrolling mandate
            });
        } else {
            appState.chartInstance.timeScale().fitContent();
        }
        
        appState.isFirstLoad = false; // LOCK IT
    }
}

// ─── UI Rendering Functions ──────────────────────────────────────────

/**
 * Renders the open trades into the dashboard position grid.
 */
function renderPositions() {
    const container = document.getElementById('positions-container');
    const totalPnlBadge = document.getElementById('total-pnl');
    if (!container) return;
    
    if (appState.openTrades.length === 0) {
        container.innerHTML = '<div class="empty-state">No open positions. Use AI Advisor or Quick Log.</div>';
        if (totalPnlBadge) {
            totalPnlBadge.textContent = '₹0.00';
            totalPnlBadge.className = 'pnl-badge';
        }
        return;
    }

    let html = '';
    let totalOpenPnl = 0;

    appState.openTrades.forEach(t => {
        const pnl = t.pnl || 0;
        totalOpenPnl += pnl;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const typeClass = t.action === 'SHORT SELL' ? 'short' : 'buy';
        const advice = t.risk_advice || {};
        const proj = t.projections || {};
        
        // P&L % calculation
        const entryValue = t.entry_price * t.quantity;
        const pnlPct = entryValue > 0 ? ((pnl / entryValue) * 100).toFixed(2) : '0.00';

        html += `
            <div class="position-card ${typeClass}">
                <div class="pos-header">
                    <div class="pos-title">
                        <span class="pos-ticker clickable-ticker" onclick="openChart('${t.ticker}')">${t.ticker.replace('.NS', '')}</span>
                        <span class="badge-${typeClass}">${t.action}</span>
                        ${advice.advice ? `<span class="advice-pill ${advice.advice.toLowerCase().includes('exit') ? 'exit' : 'hold'}">${advice.advice}</span>` : ''}
                    </div>
                    <div class="pos-pnl-group">
                        <span class="pos-pnl-pct ${pnlClass}">${pnl >= 0 ? '+' : ''}${pnlPct}%</span>
                        <span class="pos-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</span>
                    </div>
                </div>
                <div class="pos-grid">
                    <div class="pos-stat"><label>Entry</label><span>₹${t.entry_price.toFixed(2)}</span></div>
                    <div class="pos-stat"><label>LTP</label><span class="${pnlClass}">₹${(t.current_price || t.entry_price).toFixed(2)}</span></div>
                    <div class="pos-stat"><label>SL</label><span class="sl-val">₹${t.stop_loss.toFixed(2)}</span></div>
                    <div class="pos-stat"><label>Trail</label><span class="trail-val">₹${(t.trailing_sl || t.stop_loss).toFixed(2)}</span></div>
                    <div class="pos-stat"><label>T1</label><span>₹${t.target_1.toFixed(2)}</span></div>
                    <div class="pos-stat"><label>Qty</label><span>${t.quantity}</span></div>
                </div>
                ${proj.ensemble_target ? `
                    <div class="pos-projections">
                        <span class="proj-chip" title="AI Ensemble Projection">🎯 EOD Target: ₹${proj.ensemble_target.toFixed(2)}</span>
                    </div>
                ` : ''}
                ${advice.reason ? `<div class="pos-advice-reason">${advice.reason}</div>` : ''}
                <div class="pos-actions">
                    <button class="btn secondary sm" onclick="openChart('${t.ticker}')">📈 Chart</button>
                    <button class="btn close-btn sm" onclick="closeTrade('${t.id}', ${t.current_price || t.entry_price})">Close</button>
                </div>
            </div>`;
    });

    container.innerHTML = html;
    
    // Update global P&L badge
    if (totalPnlBadge) {
        totalPnlBadge.textContent = (totalOpenPnl >= 0 ? '+' : '') + `₹${totalOpenPnl.toFixed(2)}`;
        totalPnlBadge.className = `pnl-badge ${totalOpenPnl >= 0 ? 'positive' : 'negative'}`;
    }
}

/**
 * Updates the top summary bar with day performance and AI quota.
 */
function updateDaySummary() {
    let totalPnl = 0;
    let wins = 0, losses = 0;
    
    appState.openTrades.forEach(t => totalPnl += (t.pnl || 0));
    appState.closedTrades.forEach(t => {
        const pnl = t.pnl || 0;
        totalPnl += pnl;
        if (pnl > 0) wins++; else if (pnl < 0) losses++;
    });

    // P&L
    const pnlEl = document.getElementById('ds-pnl');
    if (pnlEl) {
        pnlEl.textContent = `₹${totalPnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
        pnlEl.className = `ds-value ${totalPnl >= 0 ? 'stat-up' : 'stat-down'}`;
    }

    // Trades & Win Rate
    const totalTrades = appState.openTrades.length + appState.closedTrades.length;
    document.getElementById('ds-trades').textContent = totalTrades;
    document.getElementById('ds-wl').textContent = `${wins} / ${losses}`;
    
    const wr = appState.closedTrades.length > 0 ? ((wins / appState.closedTrades.length) * 100).toFixed(1) : '--';
    document.getElementById('ds-winrate').textContent = wr === '--' ? '--' : wr + '%';

    // AI Quota
    const used = appState.aiCallsToday || 0;
    const limit = appState.aiCallsLimit || 7;
    const quotaFill = document.getElementById('ds-quota-fill');
    const quotaText = document.getElementById('ds-quota-text');
    if (quotaFill && quotaText) {
        const pct = Math.min(100, (used / limit) * 100);
        quotaFill.style.width = `${pct}%`;
        quotaFill.style.background = pct > 80 ? '#ef4444' : pct > 50 ? '#fbbf24' : '#3b82f6';
        quotaText.textContent = `${used}/${limit}`;
    }
}

/**
 * Updates the market pulse tickers with India and Global indices.
 * Requirement 5: Ensure GIFT NIFTY and global indices are correctly ordered.
 */
function updateMarketPulse(ctx) {
    if (!ctx) return;
    
    // VIX
    const vix = ctx.vix || {};
    const vixValEl = document.getElementById('vix-val');
    if (vixValEl) {
        const sign = (vix.change_pct || 0) >= 0 ? '+' : '';
        vixValEl.textContent = (vix.value !== undefined && vix.value !== null) ? `${vix.value} (${sign}${vix.change_pct}%)` : '--';
        vixValEl.className = (vix.change_pct || 0) > 2 ? 'negative' : (vix.change_pct || 0) < -2 ? 'positive' : '';
    }

    const india = ctx.india || {};
    const global = ctx.global || {};

    // Indian Indices Order
    const inOrder = ["GIFT Nifty", "Nifty 50", "Sensex", "Bank Nifty", "FinNifty", "Nifty Midcap 100", "Nifty Smallcap 100"];
    let htmlIn = '';
    inOrder.forEach(name => {
        const d = india[name] || global[name];
        if (d && d.value !== undefined) {
            const cls = d.change_pct >= 0 ? 'positive' : 'negative';
            const sign = d.change_pct >= 0 ? '+' : '';
            htmlIn += `<span class="pulse-item ${cls}">${name}: ${d.value.toLocaleString('en-IN')} (${sign}${d.change_pct.toFixed(2)}%)</span>`;
        }
    });
    const pulseIndia = document.getElementById('market-pulse-india');
    if (pulseIndia) pulseIndia.innerHTML = htmlIn || '<span class="pulse-item">Waiting for Indian market data...</span>';

    // Global Indices Order
    const glOrder = ["USD/INR", "WTI Crude", "Brent Crude", "Gold", "Silver", "S&P 500", "Nasdaq"];
    let htmlGl = '';
    glOrder.forEach(name => {
        const d = global[name];
        if (d && d.value !== undefined) {
            const cls = d.change_pct >= 0 ? 'positive' : 'negative';
            const sign = d.change_pct >= 0 ? '+' : '';
            htmlGl += `<span class="pulse-item ${cls}">${name}: ${d.value.toLocaleString()} (${sign}${d.change_pct.toFixed(2)}%)</span>`;
        }
    });
    const pulseGlobal = document.getElementById('market-pulse-global');
    if (pulseGlobal) pulseGlobal.innerHTML = htmlGl || '<span class="pulse-item">Waiting for Global indices...</span>';
}

/**
 * Updates the current market phase and associated guidance.
 */
function updateMarketPhase(phase) {
    if (!phase) return;
    const badge = document.getElementById('market-phase');
    if (badge) {
        badge.textContent = phase.phase_label || phase.phase;
        badge.className = `phase-badge phase-${(phase.phase || '').toLowerCase()}`;
    }

    const guidance = document.getElementById('phase-guidance');
    if (guidance) {
        guidance.innerHTML = `
            <span class="advisor-icon">💡</span>
            <p>${phase.guidance || 'Market monitoring in progress...'}</p>
            ${phase.mins_to_close > 0 ? `<span class="time-left">${phase.mins_to_close}m to close</span>` : ''}
        `;
    }
}

/**
 * Renders the action timeline of events (trades, scans, etc.).
 */
function renderTimeline() {
    const container = document.getElementById('timeline-container');
    if (!container) return;
    
    if (!appState.actionTimeline || appState.actionTimeline.length === 0) {
        container.innerHTML = '<div class="empty-state">Events appear here as the day progresses.</div>';
        return;
    }

    const iconMap = {
        'TRADE_OPEN': '🟢',
        'TRADE_CLOSE': '🔴',
        'AI_SCAN': '🤖',
        'AI_POSITION_REVIEW': '📋',
        'AI_EXIT_GUIDANCE': '⚡',
        'SENTINEL_ALERT': '🛡️'
    };

    let html = '';
    // Show newest events first
    [...appState.actionTimeline].reverse().forEach(e => {
        html += `
            <div class="timeline-event">
                <span class="tl-time">${e.time || ''}</span>
                <span class="tl-icon">${iconMap[e.type] || '📌'}</span>
                <span class="tl-message">${e.message || ''}</span>
            </div>`;
    });
    container.innerHTML = html;
}

/**
 * Renders historical scans for the history page.
 */
function renderHistoryScans(scans) {
    const container = document.getElementById('history-scans-container');
    if (!container) return;

    if (!scans || scans.length === 0) {
        container.innerHTML = '<div class="empty-state">No AI scans found for the selected period.</div>';
        return;
    }

    let html = '';
    scans.forEach(scan => {
        const timeStr = new Date(scan.timestamp).toLocaleString('en-IN', {
            day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
        });
        let resultArr = [];
        try {
            resultArr = typeof scan.output_json === 'string' ? JSON.parse(scan.output_json) : scan.output_json;
        } catch (e) { console.error("History parse error", e); }

        if (!Array.isArray(resultArr)) resultArr = [];

        html += `
            <div class="history-scan-card glass-panel">
                <div class="hsc-header">
                    <span class="hsc-time">🕒 ${timeStr}</span>
                    <span class="hsc-model-badge">${scan.model_used || 'AI'}</span>
                </div>
                <div class="hsc-results-grid">`;

        if (resultArr.length === 0) {
            html += `<div class="hsc-empty">No candidates identified in this scan cycle.</div>`;
        } else {
            resultArr.forEach(r => {
                const action = (r.action || '').toUpperCase();
                const cls = action === 'BUY' ? 'buy' : 'short';
                const ticker = (r.ticker || '').replace('.NS','');
                const fullTicker = (r.ticker || '').endsWith('.NS') ? r.ticker : `${r.ticker}.NS`;
                const conf = r.confidence || 0;

                html += `
                    <div class="hsc-item card-${cls}" onclick="openChart('${fullTicker}')">
                        <div class="hsc-item-top">
                            <span class="hsc-ticker">${ticker}</span>
                            <span class="hsc-badge badge-${cls}">${action}</span>
                        </div>
                        <div class="hsc-item-bottom">
                            <span class="hsc-conf-label">Confidence</span>
                            <span class="hsc-conf-val">${conf}%</span>
                        </div>
                    </div>`;
            });
        }
        html += `</div></div>`;
    });
    container.innerHTML = html;
}
/**
 * Renders today's scans into the discovery column.
 */
function renderIntradayAIHistory() {
    const container = document.getElementById('ai-history-body');
    if (!container) return;
    
    if (!appState.aiScansToday || appState.aiScansToday.length === 0) {
        container.innerHTML = '<div class="empty-state">Scans will appear here after execution.</div>';
        return;
    }

    let html = '';
    [...appState.aiScansToday].reverse().forEach(scan => {
        const time = scan.timestamp || '00:00';
        const results = Array.isArray(scan.result) ? scan.result : [];
        if (results.length === 0) return;

        html += `
            <div class="intraday-scan-card glass-panel">
                <div class="isc-header">
                    <span class="isc-time">🕒 ${time}</span>
                    <span class="isc-count">${results.length} Picks</span>
                </div>
                <div class="isc-grid">`;
        
        results.forEach(r => {
            const action = (r.action || '').toUpperCase();
            const cls = action === 'BUY' ? 'buy' : 'short';
            const ticker = (r.ticker || '').replace('.NS','');
            const fullTicker = (r.ticker || '').endsWith('.NS') ? r.ticker : `${r.ticker}.NS`;
            const conf = r.confidence !== undefined ? r.confidence : '--';
            
            html += `
                <div class="isc-item item-${cls}" onclick="openChart('${fullTicker}')">
                    <span class="isc-ticker">${ticker}</span>
                    <span class="isc-badge badge-${cls}">${action}</span>
                    <span class="isc-conf">${conf}%</span>
                </div>`;
        });
        html += `</div></div>`;
    });
    container.innerHTML = html;
}

/**
 * Renders detailed backtest results.
 */
function handleBacktestResults(results) {
    const btn = document.getElementById('run-backtest-btn');
    if (btn) {
        btn.disabled = false;
        btn.textContent = '🚀 Run Backtest';
    }

    const container = document.getElementById('backtest-results-container');
    if (!container) return;

    if (results.error) {
        container.innerHTML = `<div class="error-msg">⚠️ Backtest failed: ${results.error}</div>`;
        return;
    }

    const pnlClass = results.net_profit >= 0 ? 'positive' : 'negative';

    container.innerHTML = `
        <div class="bt-results-grid">
            <div class="bt-summary-glass glass-panel">
                <h3 style="margin-top:0; color:var(--primary)">📊 Result Summary</h3>
                <div class="bt-stat-card">
                    <label>Net Profit</label>
                    <strong class="${pnlClass}">₹${results.net_profit.toLocaleString()}</strong>
                </div>
                <div class="bt-stat-card">
                    <label>Win Rate</label>
                    <strong>${results.win_rate}%</strong>
                </div>
                <div class="bt-stat-card">
                    <label>Total Trades</label>
                    <strong>${results.total_trades}</strong>
                </div>
                <div class="bt-stat-card">
                    <label>Max Drawdown</label>
                    <strong style="color:#f87171">${results.max_drawdown_pct}%</strong>
                </div>
            </div>

            <div class="bt-log-panel glass-panel">
                <h3 style="margin:20px 25px 10px; color:var(--primary)">📝 Strategy Execution Log</h3>
                <div class="bt-log-scroll" style="padding: 0 25px 25px;">
                    <table class="history-table">
                        <thead style="position:sticky; top:0; background: #1a2234; z-index:2">
                            <tr><th>Type</th><th>Entry</th><th>Exit</th><th>P&L %</th></tr>
                        </thead>
                        <tbody>
                            ${(results.trade_log || []).map(t => `
                                <tr>
                                    <td><span class="badge-${t.type.includes('BUY') ? 'buy' : 'short'}">${t.type}</span></td>
                                    <td>₹${t.entry_price.toFixed(2)}</td>
                                    <td>₹${t.exit_price.toFixed(2)}</td>
                                    <td class="${t.pnl_pct >= 0 ? 'positive' : 'negative'}">
                                        <b>${(t.pnl_pct * 100).toFixed(2)}%</b>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}
/**
 * Renders new AI Advisor results (Rich Stock Cards).
 */
function handleAIAdvisorUpdate(data) {
    const section = document.getElementById('ai-result-section');
    const body = document.getElementById('ai-result-body');
    if (!section || !body) return;

    if (!data) {
        appState.aiAdvisor = null;
        body.innerHTML = '<div class="ai-empty">No active recommendations.</div>';
        return;
    }

    appState.aiAdvisor = data;
    section.style.display = 'block';
    const result = data.result;

    if (Array.isArray(result)) {        if (result.length === 0) {
            body.innerHTML = '<div class="ai-empty">No high-probability setups identified in this scan. Efficiency over frequency. ✅</div>';
            return;
        }

        let html = '';
        result.forEach(r => {
            const cls = (r.action || '').toUpperCase() === 'BUY' ? 'buy' : 'short';
            const ticker = (r.ticker || '').replace('.NS', '');
            const fullTicker = (r.ticker || '').endsWith('.NS') ? r.ticker : `${r.ticker}.NS`;
            const ta = r.technicals || {};
            const rl = r.risk_levels || {};
            const fund = r.fundamentals || {};
            const sent = r.sentiment || {};
            const signal = r.signal || r.action || 'NEUTRAL';
            const conf = r.confidence !== undefined ? r.confidence : '--';

            html += `
                <div class="stock-card ${cls}">
                    <div class="sc-header">
                        <div class="sc-title">
                            <span class="sc-ticker aih-ticker" onclick="openChart('${fullTicker}')">${ticker}</span>
                            <span class="sc-signal ${signalClass(signal)}">${signal}</span>
                        </div>
                        <div class="sc-price-block">
                            <span class="sc-price sc-live-price">₹${fmt(r.live_price)}</span>
                            <div class="sc-confidence" title="AI Consistency Score">
                                <div class="conf-bar" style="width:${conf === '--' ? 0 : conf}%; background: ${conf > 75 ? '#22c55e' : conf > 50 ? '#fbbf24' : '#ef4444'}"></div>
                                <span>${conf}%</span>
                            </div>
                        </div>
                    </div>

                    <div class="sc-reasoning">${r.reasoning || r.reason || 'Momentum setup based on TA convergence.'}</div>

                    <div class="sc-ta-row">
                        <div class="ta-chip" title="RSI Indicator"><span class="ta-label">RSI</span><span class="ta-value" style="color:${rsiColor(ta.rsi_14)}">${ta.rsi_14 || '--'}</span></div>
                        <div class="ta-chip" title="MACD Convergence"><span class="ta-label">MACD</span><span class="ta-value">${ta.macd_hist || '--'}</span></div>
                        <div class="ta-chip" title="ADX Trend Strength"><span class="ta-label">ADX</span><span class="ta-value">${ta.adx_14 || '--'}</span></div>
                        <div class="ta-chip" title="Volume Relative to 20-day Avg"><span class="ta-label">VOL</span><span class="ta-value">${ta.vol_surge || 1}x</span></div>
                        <div class="ta-chip" title="Lorentzian AI Score"><span class="ta-label">LZ AI</span><span class="ta-value">${ta.lz_score || '--'}</span></div>
                    </div>

                    <div class="sc-levels-row">
                        <div class="level-chip entry"><span>Entry</span><span>₹${fmt(r.live_price)}</span></div>
                        <div class="level-chip sl"><span>SL</span><span>₹${fmt(rl.stop_loss)}</span></div>
                        <div class="level-chip t1"><span>T1</span><span>₹${fmt(rl.target_1)}</span></div>
                        <div class="level-chip qty"><span>Qty</span><span>${rl.quantity || '--'}</span></div>
                    </div>

                    <div class="sc-meta-row">
                        ${fund.sector ? `<span class="meta-tag">Sector: ${fund.sector}</span>` : ''}
                        <span class="meta-tag ${sentimentClass(sent.label)}">${sent.label || 'Neutral'} Sentiment (${sent.headline_count || 0} news)</span>
                    </div>

                    <div class="sc-actions">
                        <button class="btn primary sm" onclick="quickLogFromScan('${fullTicker}', '${r.action}')">📝 Log</button>
                        <button class="btn secondary sm" onclick="openChart('${fullTicker}')">📈 Chart</button>
                    </div>
                </div>`;
        });
        body.innerHTML = html;
    } else if (result && typeof result === 'object') {
        // Handle position review or exit guidance
        const msg = result.message || result.reasoning || 'Reviewing open positions for risk...';
        body.innerHTML = `
            <div class="ai-guidance-card glass-panel">
                <div class="agc-header">💡 AI Strategy Insight</div>
                <div class="agc-body">${msg}</div>
            </div>`;
    }
}

/**
 * Renders the recent trades table.
 */
function renderTradeHistory() {
    const body = document.getElementById('trades-body');
    if (!body) return;
    
    let html = '';
    // Show open first, then closed
    appState.openTrades.forEach(t => {
        html += `
            <tr class="row-open">
                <td><span class="clickable-ticker" onclick="openChart('${t.ticker}')">${t.ticker.replace('.NS', '')}</span></td>
                <td><span class="badge-${t.action === 'BUY' ? 'buy' : 'short'}">${t.action}</span></td>
                <td>₹${t.entry_price.toFixed(2)}</td>
                <td><span class="badge-open">OPEN</span></td>
                <td class="${(t.pnl || 0) >= 0 ? 'positive' : 'negative'}">₹${(t.pnl || 0).toFixed(2)}</td>
            </tr>`;
    });
    
    appState.closedTrades.slice(0, 10).forEach(t => {
        html += `
            <tr class="row-closed">
                <td>${t.ticker.replace('.NS', '')}</td>
                <td><span class="badge-${t.action === 'BUY' ? 'buy' : 'short'}">${t.action}</span></td>
                <td>₹${t.entry_price.toFixed(2)}</td>
                <td>₹${(t.exit_price || 0).toFixed(2)}</td>
                <td class="${(t.pnl || 0) >= 0 ? 'positive' : 'negative'}">₹${(t.pnl || 0).toFixed(2)}</td>
            </tr>`;
    });
    
    body.innerHTML = html || '<tr><td colspan="5" class="empty-state">No recent trades.</td></tr>';
}

// ─── State & Message Handlers ────────────────────────────────────────

/**
 * Handles 'state_update' messages from the server.
 * Requirement 4: Use strict undefined checks for capital/risk settings.
 */
function handleStateUpdate(msg) {
    // 1. Core Data
    appState.openTrades = msg.open_trades || [];
    appState.closedTrades = msg.closed_trades || [];
    appState.actionTimeline = msg.action_timeline || [];
    appState.aiCallsToday = msg.ai_calls_today || 0;
    appState.aiCallsLimit = msg.ai_calls_limit || 7;
    
    // 2. Market Context
    if (msg.market_phase) updateMarketPhase(msg.market_phase);
    if (msg.global_context) updateMarketPulse(msg.global_context);
    if (msg.ai_advisor) handleAIAdvisorUpdate(msg.ai_advisor);
    if (msg.ai_scans_today) {
        appState.aiScansToday = msg.ai_scans_today;
        renderIntradayAIHistory();
    }

    // 6. Holiday Alerts - Show in banner below P&L section
    const holidayBanner = document.getElementById('holiday-alert-banner');
    const holidayMessage = document.getElementById('holiday-alert-message');
    
    if (msg.holiday_info) {
        const info = msg.holiday_info;
        if (info.is_holiday) {
            if (holidayBanner && holidayMessage) {
                holidayMessage.textContent = `Today is ${info.holiday_name || 'a holiday'}. Markets are closed.`;
                holidayBanner.style.display = 'flex';
            }
        } else if (info.upcoming) {
            if (holidayBanner && holidayMessage) {
                holidayMessage.textContent = `Markets will be closed on ${info.upcoming.date} for ${info.upcoming.name} (${info.upcoming.days_away} days away).`;
                holidayBanner.style.display = 'flex';
            }
        } else {
            // No holiday - hide banner
            if (holidayBanner) {
                holidayBanner.style.display = 'none';
            }
        }
    }

    // 3. Risk & Settings (Requirement 4: Strict Checks)
    if (msg.capital !== undefined && msg.capital !== null) {
        const capInput = document.getElementById('capital-input');
        if (capInput) capInput.value = msg.capital;
    }
    if (msg.max_loss !== undefined && msg.max_loss !== null) {
        const riskInput = document.getElementById('risk-input');
        if (riskInput) riskInput.value = msg.max_loss;
    }
    
    // 4. Provider Settings
    if (msg.search_engine) {
        const se = document.getElementById('search-engine-input');
        if (se) se.value = msg.search_engine;
    }
    if (msg.data_provider) {
        const dp = document.getElementById('data-provider-input');
        if (dp) dp.value = msg.data_provider;
        
        // Update provider status badges
        updateDataProviderStatus(msg.data_provider, msg.is_using_fallback);
    }
    
    // Update data provider quota display
    if (msg.data_provider_quota) {
        updateDataProviderQuota(msg.data_provider_quota);
    }
    
    // Update rate limit status
    if (msg.rate_limit_status) {
        updateRateLimitStatus(msg.rate_limit_status);
    }
    
    // Update data freshness status
    if (msg.is_data_stale !== undefined) {
        appState.isDataStale = msg.is_data_stale;
    }
    if (msg.last_data_update) {
        updateDataFreshness(msg.last_data_update);
    }
    if (msg.ai_provider) {
        const ap = document.getElementById('ai-provider-input');
        if (ap) {
            ap.value = msg.ai_provider;
            updateModelDropdown();
            // Delayed model selection to wait for list_models response
            if (msg.ai_model) {
                setTimeout(() => {
                    const am = document.getElementById('ai-model-input');
                    if (am) am.value = msg.ai_model;
                }, 800);
            }
        }
    }

    // 5. Fallback Settings
    if (msg.fallback_data !== undefined) {
        const fd = document.getElementById('fallback-data-input');
        if (fd) fd.checked = msg.fallback_data;
    }
    if (msg.fallback_search !== undefined) {
        const fs = document.getElementById('fallback-search-input');
        if (fs) fs.checked = msg.fallback_search;
    }
    if (msg.fallback_ai !== undefined) {
        const fa = document.getElementById('fallback-ai-input');
        if (fa) fa.checked = msg.fallback_ai;
    }

    if (msg.connection_status) {
        const upxPill = document.getElementById('upx-status');
        if (upxPill) {
            const conn = msg.connection_status?.upstox?.connected;
            upxPill.className = `health-pill ${conn ? 'active' : 'disconnected'}`;
            upxPill.title = `Upstox: ${conn ? 'Connected' : 'Disconnected'}`;
        }
    }

    // 6. Refresh UI
    renderPositions();
    renderTimeline();
    updateDaySummary();
    renderTradeHistory();
    renderIntradayAIHistory();
    if (msg.ai_advisor !== undefined) handleAIAdvisorUpdate(msg.ai_advisor);
}

/**
 * Shows a system-wide alert modal.
 */
function showAlert(title, message) {
    const modal = document.getElementById('alert-modal');
    const titleEl = document.getElementById('alert-title');
    const msgEl = document.getElementById('alert-message');
    if (modal && titleEl && msgEl) {
        titleEl.textContent = title;
        msgEl.textContent = message;
        modal.style.display = 'flex';
    }
}

/**
 * Handles real-time price ticks.
 */
function handleLiveTick(tick) {
    if (!tick) return;
    
    // SKEPTIC: Map Upstox 'key' to our internal ticker representation
    const tickKey = tick.key || tick.symbol || "";
    const cleanKey = tickKey.replace("NSE_EQ|", "").replace("NSE_INDEX|", "");
    const tickerWithNS = cleanKey.endsWith(".NS") ? cleanKey : (cleanKey + ".NS");
    
    // 1. Update matching open trades
    appState.openTrades.forEach(t => {
        if (t.ticker === tickerWithNS || t.ticker === cleanKey) {
            t.current_price = tick.ltp;
            // Recalculate P&L
            if (t.action === 'BUY') {
                t.pnl = (tick.ltp - t.entry_price) * t.quantity;
            } else {
                t.pnl = (t.entry_price - tick.ltp) * t.quantity;
            }
        }
    });
    
    // 2. Update UI
    renderPositions();
    updateDaySummary();
    renderTradeHistory();

    // 3. Update active chart if it matches
    if (appState.currentChartKey === tickerWithNS || appState.currentChartKey === cleanKey) {
        if (appState.series.candles) {
            const tickTime = (tick.ltt ? Math.floor(tick.ltt / 1000) : Math.floor(Date.now() / 1000)) + 19800;
            appState.series.candles.update({
                time: tickTime,
                open: tick.ltp,
                high: tick.ltp,
                low: tick.ltp,
                close: tick.ltp
            });
            // Update Legend LTP
            const ltpEl = document.getElementById('legend-ltp');
            if (ltpEl) ltpEl.textContent = `LTP: ₹${tick.ltp.toFixed(2)}`;
        }
    }
}

/**
 * Central WebSocket message dispatcher.
 * Requirement 2: handleServerMessage/onmessage with no duplications.
 */
ws.onopen = () => {
    console.log("WebSocket connected");
    ws.send(JSON.stringify({ action: "get_status" }));
};

ws.onmessage = (event) => {
    try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
            case "state_update":
                handleStateUpdate(msg);
                break;
            case "trades_update":
                appState.openTrades = msg.open_trades || [];
                appState.closedTrades = msg.closed_trades || [];
                renderPositions();
                updateDaySummary();
                renderTradeHistory();
                break;
            case "chart_data":
                renderChart(msg.data);
                break;
            case "tick":
                handleLiveTick(msg.data);
                break;
            case "backtest_results":
                handleBacktestResults(msg.data);
                break;
            case "ai_history_update":
                renderHistoryScans(msg.data);
                break;
            case "trade_history_90d":
                const historyContainer = document.getElementById('history-body');
                if (!historyContainer) return;
                
                if (!msg.data || msg.data.length === 0) {
                    historyContainer.innerHTML = '<div class="empty-state">No trade history found for the last 90 days.</div>';
                    return;
                }

                historyContainer.innerHTML = `
                    <div class="history-grid">
                        ${msg.data.map(t => {
                            const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                            const fullTicker = t.ticker.endsWith('.NS') ? t.ticker : `${t.ticker}.NS`;
                            const cleanTicker = t.ticker.replace('.NS', '');
                            return `
                                <div class="history-card glass-panel">
                                    <div class="hc-header">
                                        <span class="hc-ticker">${cleanTicker}</span>
                                        <span class="hc-date">${t.close_time}</span>
                                    </div>
                                    <div class="hc-body">
                                        <div class="hc-stat"><label>Action</label><span class="badge-${t.action === 'BUY' ? 'buy' : 'short'}">${t.action}</span></div>
                                        <div class="hc-stat"><label>Qty</label><span>${t.quantity}</span></div>
                                        <div class="hc-stat"><label>Entry</label><span>₹${t.entry_price.toFixed(2)}</span></div>
                                        <div class="hc-stat"><label>Exit</label><span>₹${t.exit_price.toFixed(2)}</span></div>
                                        <div class="hc-pnl ${pnlClass}">
                                            <small>Profit/Loss</small><br>
                                            ₹${t.pnl.toFixed(2)}
                                        </div>
                                    </div>
                                    <div style="margin-top:15px; text-align:right">
                                        <button class="btn secondary sm" onclick="openChart('${fullTicker}')">📈 View Chart</button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
                break;
            case "scan_results":
                handleAIAdvisorUpdate({ result: msg.data, timestamp: new Date().toLocaleTimeString('en-IN') });
                break;
            case "model_list":
                const modelSelect = document.getElementById('ai-model-input');
                if (modelSelect) {
                    modelSelect.innerHTML = (msg.data || []).map(m => `<option value="${m.value}">${m.label}</option>`).join('');
                }
                break;
            case "notification":
                showToast(msg.message, msg.level || 'info');
                break;
            case "connection_status":
                const upxPill = document.getElementById('upx-status');
                if (upxPill) {
                    const conn = msg.status?.upstox?.connected;
                    upxPill.className = `health-pill ${conn ? 'active' : 'disconnected'}`;
                    upxPill.title = `Upstox: ${conn ? 'Connected' : 'Disconnected'}`;
                }
                break;
            case "sentinel_alert":
                showToast(`🚨 ${msg.data.ticker}: ${msg.data.title}`, 'warning');
                break;
            case "news_update":
                // Requirement: Monitor news for shocks.
                if (msg.data && msg.data.headlines) {
                    const timeline = document.getElementById('timeline-container');
                    const feed = document.getElementById('sentinel-feed');
                    
                    msg.data.headlines.forEach(h => {
                        const eventTicker = h.ticker || msg.data.ticker || 'NEWS';
                        // 1. Update Timeline
                        if (timeline) {
                            const event = document.createElement('div');
                            event.className = 'timeline-event info';
                            event.innerHTML = `<span class="time">${new Date().toLocaleTimeString()}</span> <span class="msg"><b>${eventTicker}</b>: ${h.title}</span>`;
                            timeline.prepend(event);
                        }
                        
                        // 2. Update Sentinel Feed (Sidebar)
                        if (feed) {
                            // Clear empty state if present
                            if (feed.querySelector('.empty-state')) feed.innerHTML = '';
                            
                            const newsItem = document.createElement('div');
                            const sentiment = h.sentiment || 'Neutral';
                            newsItem.className = `news-item ${sentiment.toLowerCase()}`; // Ensure class name is lowercase for CSS (e.g. 'negative', 'bullish')
                            newsItem.innerHTML = `
                                <span class="news-title">${h.title}</span>
                                <div class="news-meta">
                                    <span>${eventTicker}</span>
                                    <span>${h.time || 'Just now'}</span>
                                </div>
                            `;
                            // Link if available
                            if (h.url && h.url !== '#') {
                                newsItem.style.cursor = 'pointer';
                                newsItem.onclick = () => window.open(h.url, '_blank');
                            }
                            feed.prepend(newsItem);
                            // Keep last 15 items
                            while (feed.children.length > 15) feed.removeChild(feed.lastChild);
                        }
                    });
                }
                break;
            case "market_data":
                // Handle market data updates from ws_handler (global indices)
                if (msg.global_context) {
                    updateMarketPulse(msg.global_context);
                }
                break;
        }
    } catch (e) {
        console.error("Message handling error", e);
    }
};

ws.onopen = () => {
    const indicator = document.getElementById('ws-status');
    if (indicator) {
        indicator.className = 'status-indicator connected';
        indicator.textContent = 'Live';
    }
    showToast("SuperNova Engine Link Established", "success");
    updateModelDropdown();
};

ws.onclose = () => {
    const indicator = document.getElementById('ws-status');
    if (indicator) indicator.className = 'status-indicator disconnected';
    showToast("SuperNova Engine Disconnected. Retrying...", "error");
    setTimeout(() => location.reload(), 5000);
};

// ─── Event Listeners ────────────────────────────────────────────────

/**
 * Requirement 8: Include all DOMContentLoaded event listeners.
 */
document.addEventListener('DOMContentLoaded', () => {
    // 1. Navigation
    const navs = {
        dashboard: document.getElementById('nav-dashboard-btn'),
        history: document.getElementById('nav-history-btn'),
        backtest: document.getElementById('nav-backtest-btn')
    };
    const pages = {
        dashboard: document.getElementById('page-dashboard'),
        history: document.getElementById('page-history'),
        backtest: document.getElementById('page-backtest')
    };

    Object.keys(navs).forEach(key => {
        navs[key]?.addEventListener('click', () => {
            // Hide all pages
            Object.values(pages).forEach(p => { if (p) { p.style.display = 'none'; p.classList.remove('active'); } });
            // Deactivate all buttons
            Object.values(navs).forEach(n => n.classList.remove('active'));
            
            // Show target
            if (pages[key]) {
                pages[key].style.display = 'block';
                pages[key].classList.add('active');
            }
            navs[key].classList.add('active');
            
            // Special triggers per page
            if (key === 'history') {
                ws.send(JSON.stringify({ action: 'get_trade_history', days: 90 }));
                // Default to trades view
                document.getElementById('history-trades-view').style.display = 'block';
                document.getElementById('history-scans-view').style.display = 'none';
            }
        });
    });

    // History View Toggles
    document.getElementById('view-trades-btn')?.addEventListener('click', () => {
        document.getElementById('history-trades-view').style.display = 'block';
        document.getElementById('history-scans-view').style.display = 'none';
    });
    document.getElementById('view-scans-btn')?.addEventListener('click', () => {
        document.getElementById('history-trades-view').style.display = 'none';
        document.getElementById('history-scans-view').style.display = 'block';
        ws.send(JSON.stringify({ action: 'get_ai_history' }));
    });

    // 2. Global Search
    const searchBtn = document.getElementById('global-chart-btn');
    const searchInput = document.getElementById('global-chart-input');
    const triggerSearch = () => {
        const val = searchInput.value.trim().toUpperCase();
        if (val) openChart(val);
    };
    searchBtn?.addEventListener('click', triggerSearch);
    searchInput?.addEventListener('keypress', (e) => { if (e.key === 'Enter') triggerSearch(); });

    // 3. Chart Controls (Timeframe & Indicators)
    document.getElementById('chart-close-btn')?.addEventListener('click', closeChart);

    // Upstox Connect Button
    document.getElementById('upstox-connect-btn')?.addEventListener('click', () => {
        window.open('/upstox/connect', '_blank');
    });
    
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const interval = btn.getAttribute('data-interval');
            if (interval) {
                appState.currentInterval = interval;
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                ws.send(JSON.stringify({ 
                    action: 'get_chart_data', 
                    ticker: appState.currentChartKey, 
                    interval: interval 
                }));
            }
        });
    });

    ['toggle-lz', 'toggle-st', 'toggle-knn'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => {
            if (appState.currentChartData) renderChart(appState.currentChartData);
        });
    });

    // 4. Scan Trigger
    document.getElementById('scan-btn')?.addEventListener('click', () => {
        const btn = document.getElementById('scan-btn');
        btn.disabled = true;
        btn.textContent = 'Scanning...';
        ws.send(JSON.stringify({ action: 'trigger_scan' }));
        // Cooldown UI
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = '⚡ Scan Market Now';
        }, 60000);
    });

    // 5. Truncate Data
    document.getElementById('truncate-data-btn')?.addEventListener('click', () => {
        if (confirm("⚠️ CRITICAL: This will PERMANENTLY DELETE all trade history, AI scans, and timeline events. System settings will be preserved. Proceed?")) {
            ws.send(JSON.stringify({ action: "truncate_test_data" }));
        }
    });

    // 6. Quick Log Trade
    document.getElementById('qt-submit')?.addEventListener('click', () => {
        const ticker = document.getElementById('qt-ticker').value.trim().toUpperCase();
        const entry = document.getElementById('qt-entry').value;
        if (!ticker || !entry) {
            showToast("Enter Ticker and Price", "warning");
            return;
        }
        logQuickTrade(ticker, 'BUY', entry);
        document.getElementById('qt-ticker').value = '';
        document.getElementById('qt-entry').value = '';
    });

    const aiProvInput = document.getElementById('ai-provider-input');
    if (aiProvInput) {
        aiProvInput.addEventListener('change', updateModelDropdown);
    }

    // 5. Settings Save
    document.getElementById('save-settings-btn')?.addEventListener('click', () => {
        const settings = {
            action: 'update_settings',
            capital: parseFloat(document.getElementById('capital-input').value),
            max_loss: parseFloat(document.getElementById('risk-input').value),
            search_engine: document.getElementById('search-engine-input').value,
            data_provider: document.getElementById('data-provider-input').value,
            ai_provider: document.getElementById('ai-provider-input').value,
            ai_model: document.getElementById('ai-model-input').value,
            fallback_data: document.getElementById('fallback-data-input').checked,
            fallback_search: document.getElementById('fallback-search-input').checked,
            fallback_ai: document.getElementById('fallback-ai-input').checked
        };
        ws.send(JSON.stringify(settings));
        // Show "saving" indicator - actual confirmation comes from server
        showToast("Saving settings to database...", "info");
    });

    // 6. Backtest Runner
    document.getElementById('run-backtest-btn')?.addEventListener('click', () => {
        const btn = document.getElementById('run-backtest-btn');
        btn.disabled = true;
        btn.textContent = '🧬 Processing...';
        
        ws.send(JSON.stringify({
            action: 'run_backtest',
            ticker: document.getElementById('bt-ticker').value.toUpperCase(),
            period: document.getElementById('bt-period').value,
            params: {
                ema_fast: parseInt(document.getElementById('bt-ema-fast').value),
                ema_slow: parseInt(document.getElementById('bt-ema-slow').value),
                rsi_len: parseInt(document.getElementById('bt-rsi-len').value),
                rsi_buy_threshold: parseInt(document.getElementById('bt-rsi-buy').value),
                rsi_short_threshold: parseInt(document.getElementById('bt-rsi-short').value),
                sl_pct: parseFloat(document.getElementById('bt-sl').value) / 100,
                tp_pct: parseFloat(document.getElementById('bt-tp').value) / 100
            }
        }));
    });

    // 7. Clock IST
    setInterval(() => {
        const clock = document.getElementById('ist-clock');
        if (clock) {
            clock.textContent = new Date().toLocaleTimeString('en-IN', {
                timeZone: 'Asia/Kolkata',
                hour12: false
            });
        }
        
        // Update rate limit countdown
        if (appState.isRateLimited && appState.rateLimitCooldown > 0) {
            appState.rateLimitCooldown--;
            const rateLimitEl = document.getElementById('rate-limit-indicator');
            if (rateLimitEl && rateLimitEl.style.display !== 'none') {
                rateLimitEl.textContent = `⏳ Rate Limited (${appState.rateLimitCooldown}s)`;
            }
            
            // Auto-refresh when cooldown expires
            if (appState.rateLimitCooldown <= 0) {
                appState.isRateLimited = false;
                updateRateLimitStatus({ is_rate_limited: false, remaining_cooldown: 0 });
                showToast('Rate limit expired. Data will refresh.', 'info');
            }
        }
    }, 1000);
});

// ─── Trade Execution Helpers ─────────────────────────────────────────

/**
 * Closes a trade at the specified price.
 */
function closeTrade(tradeId, exitPrice) {
    if (!confirm(`Confirm exit for trade ${tradeId} at ₹${exitPrice}?`)) return;
    ws.send(JSON.stringify({
        action: 'close_trade',
        trade_id: String(tradeId),
        exit_price: parseFloat(exitPrice)
    }));
    // SKEPTIC: Immediate UI cleanup for better UX (backend will confirm via trades_update)
    appState.openTrades = appState.openTrades.filter(t => String(t.id) !== String(tradeId));
    renderPositions();
    updateDaySummary();
}

/**
 * Logs a trade manually or from scan.
 */
function logQuickTrade(ticker, action, price) {
    const cleanTicker = ticker.endsWith('.NS') ? ticker : ticker + '.NS';
    ws.send(JSON.stringify({
        action: 'log_trade',
        ticker: cleanTicker,
        trade_action: action.includes('BUY') ? 'BUY' : 'SHORT SELL',
        entry_price: parseFloat(price) || 0
    }));
}

/**
 * Logs a trade specifically from a scan result card.
 */
function quickLogFromScan(ticker, action) {
    showToast(`Initializing ${action} for ${ticker.replace('.NS','')}`, 'info');
    logQuickTrade(ticker, action, 0); // 0 triggers live price lookup in backend
}

// ─────────────────────────────────────────────────────────────────────
// END OF SUPERNOVA APP ENGINE (V3)
// ─────────────────────────────────────────────────────────────────────
