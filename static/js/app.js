/**
 * V3 SuperNova Dashboard Engine
 * --------------------------------------------------
 * ARCHITECT-LEVEL RESTORATION
 * - Fixed: Candle time-series alignment (IST +19800 offset)
 * - Fixed: Timeline data flow and rendering
 * - Restored: Multi-indicator rendering (LZ, ST, KNN)
 * - Restored: Dashboard P&L context
 */

const ws = new WebSocket(`ws://${window.location.host}/ws`);

const appState = { 
    openTrades: [], closedTrades: [], globalContext: {}, marketPhase: {}, 
    aiAdvisor: null, aiScansToday: [], actionTimeline: [], aiCallsToday: 0, aiCallsLimit: 7,
    chartInstance: null, adxChart: null, currentChartData: null, 
    currentChartKey: null, currentInterval: '5m',
    series: { candles: null, lzMarkers: null, stLines: [], stUp: null, stDown: null, adxLine: null, rsiLine: null, stLine: null }
};
window.appState = appState;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initClock();
    initEventListeners();
    
    // Check for ticker in URL
    const urlParams = new URLSearchParams(window.location.search);
    const ticker = urlParams.get('ticker');
    if (ticker) {
        setTimeout(() => openChart(ticker.endsWith('.NS') ? ticker : ticker + '.NS'), 1000);
    }
});

function initNavigation() {
    const dashboardBtn = document.getElementById('nav-dashboard-btn');
    const historyBtn = document.getElementById('nav-history-btn');
    const backtestBtn = document.getElementById('nav-backtest-btn');
    const pages = {
        dashboard: document.getElementById('page-dashboard'),
        history: document.getElementById('page-history'),
        backtest: document.getElementById('page-backtest')
    };

    const switchPage = (activeBtn, activePage) => {
        Object.values(pages).forEach(p => p && (p.style.display = 'none'));
        [dashboardBtn, historyBtn, backtestBtn].forEach(b => b?.classList.remove('active'));
        activeBtn.classList.add('active');
        activePage.style.display = 'block';
    };

    dashboardBtn?.addEventListener('click', () => switchPage(dashboardBtn, pages.dashboard));
    historyBtn?.addEventListener('click', () => { switchPage(historyBtn, pages.history); renderFullHistory(); });
    backtestBtn?.addEventListener('click', () => switchPage(backtestBtn, pages.backtest));
}

function initEventListeners() {
    // Settings & Fallbacks
    document.getElementById('save-settings-btn')?.addEventListener('click', () => {
        ws.send(JSON.stringify({
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
        }));
        showToast('Settings Saved', 'success');
    });

    document.getElementById('ai-provider-input')?.addEventListener('change', updateModelDropdown);

    // Risk Modal Fix
    document.getElementById('risk-accept')?.addEventListener('click', () => {
        document.getElementById('risk-modal').style.display = 'none';
        updateRiskWarningUI();
    });
    document.getElementById('risk-decline')?.addEventListener('click', () => {
        const modal = document.getElementById('risk-modal');
        const targetId = modal.dataset.targetInputId;
        if (targetId) document.getElementById(targetId).checked = true;
        modal.style.display = 'none';
        updateRiskWarningUI();
    });

    ['fallback-data-input', 'fallback-search-input', 'fallback-ai-input'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', (e) => {
            if (!e.target.checked) {
                const modal = document.getElementById('risk-modal');
                modal.style.display = 'flex';
                modal.dataset.targetInputId = id;
            }
            updateRiskWarningUI();
        });
    });
// Scan
const scanBtn = document.getElementById('scan-btn');
if (scanBtn) {
    scanBtn.addEventListener('click', () => {
        scanBtn.disabled = true;
        scanBtn.innerHTML = 'Scanning...';
        ws.send(JSON.stringify({ action: 'trigger_scan' }));
        setTimeout(() => {
            scanBtn.disabled = false;
            scanBtn.innerHTML = '⚡ Scan Market Now';
        }, 60000);
    });
}

    // Global Search
    const searchInput = document.getElementById('global-chart-input');
    const searchBtn = document.getElementById('global-chart-btn');
    const triggerSearch = () => {
        const ticker = searchInput.value.trim().toUpperCase();
        if (ticker) openChart(ticker.endsWith('.NS') ? ticker : ticker + '.NS');
    };
    searchBtn?.addEventListener('click', triggerSearch);
    searchInput?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') triggerSearch();
    });
}

// ─── WebSocket Flow ─────────────────────────────────────────────────
ws.onopen = () => {
    const el = document.getElementById('ws-status');
    if (el) {
        el.className = 'status-indicator connected';
        el.textContent = 'Live';
    }
    updateModelDropdown();
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch(msg.type) {
        case "state_update": handleStateUpdate(msg); break;
        case "ai_advisor_update": handleAIAdvisorUpdate(msg.data); break;
        case "scan_results": handleAIAdvisorUpdate({ type: 'SCAN', result: msg.data }); break;
        case "recommendations": handleAIAdvisorUpdate({ type: 'SCAN', result: msg.data }); break;
        case "news_update": handleNewsUpdate(msg.data); break;
        case "sentinel_alert": handleSentinelAlert(msg.data); break;
        case "trades_update":
            appState.openTrades = msg.open_trades || [];
            appState.closedTrades = msg.closed_trades || [];
            renderPositions();
            renderTradeHistory();
            updateDaySummary();
            break;
        case "ai_history_update":
            renderHistoryScans(msg.data);
            break;
        case "chart_data": renderChart(msg.data); break;
        case "tick": handleLiveTick(msg.data); break;
        case "scan_results": handleAIAdvisorUpdate({ type: 'SCAN', result: msg.data }); break;
        case "backtest_results": handleBacktestResults(msg.data); break;
        case "notification": 
            showToast(msg.message, msg.level);
            appState.actionTimeline.unshift({ time: new Date().toLocaleTimeString(), message: msg.message });
            renderTimeline();
            break;
        case "model_list":
            const dd = document.getElementById('ai-model-input');
            if (dd) dd.innerHTML = msg.data.map(m => `<option value="${m.value}">${m.label}</option>`).join('');
            break;
    }
};

function handleNewsUpdate(data) {
    console.log("📥 News Update Received:", data);
    const container = document.getElementById('sentinel-feed');
    if (!container || !data.headlines) return;
    
    container.innerHTML = data.headlines.map(h => `
        <div class="news-item ${h.sentiment?.toLowerCase() || 'neutral'}">
            <div class="news-meta">
                <span class="news-ticker">${h.ticker || 'MARKET'}</span>
                <span class="news-time">${h.time || 'Now'}</span>
            </div>
            <div class="news-title"><a href="${h.url}" target="_blank">${h.title}</a></div>
        </div>
    `).join('');
}

function handleSentinelAlert(alert) {
    showToast(`🚨 ${alert.title}`, 'error');
    appState.actionTimeline.unshift({ time: 'ALERT', message: alert.title });
    renderTimeline();
}

function handleAIAdvisorUpdate(data) {
    console.log("📥 AI Advisor Update:", data);
    const body = document.getElementById('ai-result-body');
    const section = document.getElementById('ai-result-section');
    if (!body || !section) {
        console.error("❌ AI result containers not found!");
        return;
    }
    section.style.display = 'block';

    const type = data.type || 'SCAN';
    const result = data.result;

    if (!result || (Array.isArray(result) && result.length === 0)) {
        body.innerHTML = '<div class="empty-state">No actionable setups or advice at this time.</div>';
        return;
    }

    body.innerHTML = ''; // Final clear before DOM build

    if (type === 'SCAN') {
        const results = Array.isArray(result) ? result : [result];
        results.forEach(r => {
            const card = document.createElement('div');
            card.className = `stock-card ${r.action?.includes('BUY') ? 'buy' : 'short'}`;

            const header = document.createElement('div');
            header.className = 'sc-header';
            header.innerHTML = `
                <strong class="sc-ticker clickable-ticker" onclick="openChart('${r.ticker}')">${r.ticker?.replace('.NS','')}</strong>
                <span class="badge">${r.signal || r.action}</span>
                <span class="sc-price">${r.live_price ? '₹'+r.live_price : ''}</span>
                <span class="conf">${r.confidence || 0}%</span>
            `;

            const chips = document.createElement('div');
            chips.className = 'sc-ta-chips';
            chips.innerHTML = `
                <div class="ta-chip"><span class="ta-label">RSI</span><span class="ta-value">${r.ta_data?.rsi_14?.toFixed(1) || '--'}</span></div>
                <div class="ta-chip"><span class="ta-label">MACD</span><span class="ta-value">${r.ta_data?.macd_hist?.toFixed(2) || '--'}</span></div>
                <div class="ta-chip"><span class="ta-label">ADX</span><span class="ta-value">${r.ta_data?.adx_14?.toFixed(1) || '--'}</span></div>
                <div class="ta-chip"><span class="ta-label">VOL</span><span class="ta-value">${r.ta_data?.vol_surge?.toFixed(1) || '--'}x</span></div>
                <div class="ta-chip"><span class="ta-label">LZ AI</span><span class="ta-value">${r.ta_data?.lz_score?.toFixed(2) || '--'}</span></div>
            `;

            const reasoning = document.createElement('div');
            reasoning.className = 'sc-reasoning';
            reasoning.textContent = r.reasoning || r.reason || '';

            const actions = document.createElement('div');
            actions.className = 'sc-actions';
            actions.innerHTML = `
                <button class="btn primary btn-sm" onclick="openChart('${r.ticker}')">Chart</button>
                <button class="btn success btn-sm" onclick="logQuickTrade('${r.ticker}', '${r.action}', ${r.live_price || 0})">Log Trade</button>
            `;

            card.appendChild(header);
            card.appendChild(chips);
            card.appendChild(reasoning);
            card.appendChild(actions);
            body.appendChild(card);
        });
    } else if (type === 'POSITION_REVIEW') {
        const results = Array.isArray(result) ? result : [result];
        results.forEach(r => {
            const card = document.createElement('div');
            card.className = 'stock-card';
            card.innerHTML = `
                <div class="sc-header">
                    <strong class="sc-ticker clickable-ticker" onclick="openChart('${r.ticker}')">${r.ticker?.replace('.NS','')}</strong>
                    <span class="badge" style="background:var(--accent);">${r.action}</span>
                    <span class="conf" style="color:var(--warning);">${r.urgency || 'MEDIUM'} URGENCY</span>
                </div>
                <div class="sc-reasoning">${r.reasoning || r.reason || ''}</div>
                <div class="sc-actions">
                    <button class="btn primary btn-sm" onclick="openChart('${r.ticker}')">Chart</button>
                </div>
            `;
            body.appendChild(card);
        });
    } else if (type === 'EXIT_GUIDANCE') {
        const card = document.createElement('div');
        card.className = `stock-card ${result.should_close_all ? 'short' : ''}`;
        card.style.borderColor = 'var(--danger)';
        card.innerHTML = `
            <div class="sc-header">
                <strong>MARKET CLOSE GUIDANCE</strong>
                <span class="badge" style="background:var(--danger);">${result.should_close_all ? 'CLOSE ALL' : 'HOLD SELECT'}</span>
            </div>
            <div class="sc-reasoning">${result.reasoning || ''}</div>
            <div class="sc-positions-list" style="margin-top:10px;">
                ${(result.per_position || []).map(p => `
                    <div style="padding:8px 0; border-top:1px solid var(--border-color);">
                        <strong class="clickable-ticker" onclick="openChart('${p.ticker}')">${p.ticker?.replace('.NS','')}</strong>: ${p.action} - ${p.reason}
                    </div>
                `).join('')}
            </div>
        `;
        body.appendChild(card);
    }
}


function closeTrade(id, price) {
    if (!id || !price) {
        showToast('Invalid trade data for closing', 'error');
        return;
    }
    ws.send(JSON.stringify({ 
        action: 'close_trade', 
        trade_id: id, 
        exit_price: parseFloat(price) 
    }));
}

function logQuickTrade(ticker, action, price) {
    if (!ticker) return;
    const cleanTicker = ticker.endsWith('.NS') ? ticker : `${ticker}.NS`;
    ws.send(JSON.stringify({
        action: 'log_trade',
        ticker: cleanTicker,
        trade_action: action.includes('BUY') ? 'BUY' : 'SHORT SELL',
        entry_price: parseFloat(price) || 0
    }));
}

// ─── UI Rendering ───────────────────────────────────────────────────
function handleStateUpdate(msg) {
    appState.openTrades = msg.open_trades || [];
    appState.closedTrades = msg.closed_trades || [];
    appState.actionTimeline = msg.action_timeline || [];
    appState.aiCallsToday = msg.ai_calls_today || 0;
    
    if (msg.market_phase) {
        appState.marketPhase = msg.market_phase;
        updateMarketPhase(msg.market_phase);
    }
    if (msg.global_context) {
        appState.globalContext = msg.global_context;
        updateMarketPulse(msg.global_context);
    }
    if (msg.ai_advisor) {
        appState.aiAdvisor = msg.ai_advisor;
        handleAIAdvisorUpdate(msg.ai_advisor);
    }
    if (msg.ai_scans_today) {
        appState.aiScansToday = msg.ai_scans_today;
        renderIntradayAIHistory();
    }

    if (msg.capital) document.getElementById('capital-input').value = msg.capital;
    if (msg.max_loss) document.getElementById('risk-input').value = msg.max_loss;
    if (msg.search_engine) document.getElementById('search-engine-input').value = msg.search_engine;
    if (msg.data_provider) document.getElementById('data-provider-input').value = msg.data_provider;
    if (msg.ai_provider) {
        document.getElementById('ai-provider-input').value = msg.ai_provider;
        updateModelDropdown();
    }
    if (msg.ai_model) document.getElementById('ai-model-input').value = msg.ai_model;
    
    if (msg.fallback_data !== undefined) document.getElementById('fallback-data-input').checked = msg.fallback_data;
    if (msg.fallback_search !== undefined) document.getElementById('fallback-search-input').checked = msg.fallback_search;
    if (msg.fallback_ai !== undefined) document.getElementById('fallback-ai-input').checked = msg.fallback_ai;

    renderPositions();
    renderTimeline();
    updateDaySummary();
}

function updateMarketPhase(phase) {
    const badge = document.getElementById('market-phase');
    if (badge) {
        badge.textContent = phase.phase_label || phase.phase;
        badge.className = `phase-badge phase-${(phase.phase || '').toLowerCase()}`;
    }
    const guidance = document.getElementById('phase-guidance');
    if (guidance) {
        guidance.innerHTML = `<span class="advisor-icon">💡</span><p>${phase.guidance || ''}</p>
            ${phase.mins_to_close > 0 ? `<span class="time-left">${phase.mins_to_close}m to close</span>` : ''}`;
    }
}

function renderIntradayAIHistory() {
    const body = document.getElementById('ai-history-body');
    if (!body || !appState.aiScansToday) return;
    
    if (appState.aiScansToday.length === 0) {
        body.innerHTML = '<div class="empty-state">No past recommendations yet.</div>';
        return;
    }

    body.innerHTML = appState.aiScansToday.slice(0, 5).map(scan => {
        const time = scan.timestamp || 'Recent';
        const type = scan.type || 'SCAN';
        const results = Array.isArray(scan.result) ? scan.result : [scan.result];
        
        return `
            <div class="ai-history-item">
                <div class="aih-meta">🕒 ${time} | ${type}</div>
                <div class="aih-results">
                    ${results.map(r => `<span class="aih-ticker clickable-ticker ${r.action?.includes('BUY')?'pos':'neg'}" onclick="openChart('${r.ticker}')">${r.ticker?.replace('.NS','')}</span>`).join('')}
                </div>
            </div>
        `;
    }).join('');
}

function updateMarketPulse(ctx) {
    const vix = ctx.vix || {};
    const vixEl = document.getElementById('vix-val');
    if (vixEl) vixEl.textContent = vix.value ? `${vix.value} (${vix.change_pct >= 0 ? '+' : ''}${vix.change_pct}%)` : '--';

    const india = ctx.india || {};
    const global = ctx.global || {};

    // ─── ROW 1: INDIA (GIFT Nifty, NIFTY, BANKNIFTY, etc) ────────────
    let htmlIndia = '';
    const inOrder = ["GIFT Nifty", "NIFTY 50", "SENSEX", "BANK NIFTY", "NIFTY IT", "NIFTY AUTO", "MIDCAP 100", "SMALLCAP 100"];
    inOrder.forEach(name => {
        const d = india[name] || global[name]; 
        if (d && d.value !== undefined) {
            const cls = d.change_pct >= 0 ? 'positive' : 'negative';
            htmlIndia += `<span class="pulse-item ${cls}">${name}: ${d.value.toLocaleString('en-IN')} (${d.change_pct >= 0 ? '+' : ''}${d.change_pct.toFixed(2)}%)</span>`;
        }
    });
    const inEl = document.getElementById('market-pulse-india');
    if (inEl) inEl.innerHTML = htmlIndia || '<span class="pulse-item">Loading Indian data...</span>';

    // ─── ROW 2: GLOBAL (USDINR, Crude, Gold, Silver leftmost) ────────
    let htmlGlobal = '';
    const glOrder = ["USD/INR", "WTI Crude", "Brent Crude", "Gold", "Silver", "S&P 500", "NASDAQ", "DOW 30", "DAX", "FTSE 100", "NIKKEI"];
    glOrder.forEach(name => {
        const d = global[name];
        if (d && d.value !== undefined) {
            const cls = d.change_pct >= 0 ? 'positive' : 'negative';
            const locale = name === "USD/INR" ? 'en-IN' : 'en-US';
            htmlGlobal += `<span class="pulse-item ${cls}">${name}: ${d.value.toLocaleString(locale)} (${d.change_pct >= 0 ? '+' : ''}${d.change_pct.toFixed(2)}%)</span>`;
        }
    });
    const glEl = document.getElementById('market-pulse-global');
    if (glEl) glEl.innerHTML = htmlGlobal || '<span class="pulse-item">Loading Global data...</span>';
}

function renderTimeline() {
    const container = document.getElementById('timeline-container');
    if (!container) return;
    container.innerHTML = appState.actionTimeline.slice(0, 20).map(item => `
        <div class="timeline-item">
            <span class="tl-time">${item.time || ''}</span>
            <span class="tl-msg">${item.message}</span>
        </div>
    `).join('');
}

function renderChart(data) {
    const container = document.getElementById('chart-container');
    const adxContainer = document.getElementById('adx-container');
    if (!container || !adxContainer) return;

    appState.currentChartData = data;

    if (!appState.chartInstance) {
        const opt = { 
            width: container.clientWidth, height: 450,
            layout: { background: { type: 'solid', color: '#0b0e11' }, textColor: '#8b949e' },
            timeScale: { timeVisible: true, secondsVisible: false }
        };
        appState.chartInstance = LightweightCharts.createChart(container, opt);
        appState.series.candles = appState.chartInstance.addCandlestickSeries({ upColor: '#22c55e', downColor: '#ef4444' });
        appState.adxChart = LightweightCharts.createChart(adxContainer, { ...opt, height: 150 });
        appState.series.adxLine = appState.adxChart.addLineSeries({ color: '#facc15', title: 'ADX' });
        appState.series.rsiLine = appState.adxChart.addLineSeries({ color: '#60a5fa', title: 'RSI', lineWidth: 1 });
        
        appState.series.stLine = appState.chartInstance.addLineSeries({ color: '#34d399', lineWidth: 2 });
        appState.series.stUp = appState.chartInstance.addLineSeries({ color: '#22c55e', lineWidth: 2 }); // Compatibility
        appState.series.stDown = appState.chartInstance.addLineSeries({ color: '#ef4444', lineWidth: 2 }); // Compatibility
        
        // Populate test series with dummy data if needed
        appState.series.stUp.setData([]);
        appState.series.stDown.setData([]);
        
        // Sync Time Scales
        appState.chartInstance.timeScale().subscribeVisibleTimeRangeChange(range => {
            appState.adxChart.timeScale().setVisibleRange(range);
        });
        appState.adxChart.timeScale().subscribeVisibleTimeRangeChange(range => {
            appState.chartInstance.timeScale().setVisibleRange(range);
        });
        
        // IST Offset logic: Data from backend is UTC epoch. Lightweight charts expects local or specific unix.
        // We add 19800 (5.5h) for IST visualization consistency.
    }

    // Fixed: Ensure data is sorted by time and mapped correctly, removing duplicates and NaNs
    let ohlc = (data.ohlc || []).map(c => ({
        time: (c.time || 0) + 19800,
        open: parseFloat(c.open) || 0,
        high: parseFloat(c.high) || 0,
        low: parseFloat(c.low) || 0,
        close: parseFloat(c.close) || 0
    })).filter(c => c.time > 19800 && c.open > 0).sort((a,b) => a.time - b.time);

    // Remove duplicates
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
        try {
            appState.series.candles.setData(ohlc);
            // Update legend
            const last = ohlc[ohlc.length - 1];
            const ltpEl = document.getElementById('legend-ltp');
            const ohlcEl = document.getElementById('legend-ohlc');
            if (ltpEl) ltpEl.textContent = `LTP: ₹${last.close.toFixed(2)}`;
            if (ohlcEl) ohlcEl.textContent = `O: ${last.open.toFixed(2)} H: ${last.high.toFixed(2)} L: ${last.low.toFixed(2)} C: ${last.close.toFixed(2)}`;
        } catch (e) {
            console.error("Candle setData error:", e);
        }
    }

    if (data.adx_series) {
        let adx = data.adx_series.map(p => ({ 
            time: (p.time || 0) + 19800, 
            value: parseFloat(p.value) || 0 
        })).filter(p => p.time > 19800).sort((a,b)=>a.time-b.time);
        
        // Remove duplicates for ADX
        const uAdx = [];
        let lTime = -1;
        for (const p of adx) {
            if (p.time > lTime) { uAdx.push(p); lTime = p.time; }
        }
        
        try {
            appState.series.adxLine.setData(uAdx);
            // Update legend
            if (uAdx.length > 0) {
                const lastAdx = uAdx[uAdx.length - 1];
                const adxEl = document.getElementById('legend-adx');
                if (adxEl) adxEl.textContent = `ADX: ${lastAdx.value.toFixed(1)}`;
            }
        } catch(e) {
            console.error("ADX setData error:", e);
        }
    }

    if (data.rsi_series) {
        let rsi = data.rsi_series.map(p => ({ 
            time: (p.time || 0) + 19800, 
            value: parseFloat(p.value) || 0 
        })).filter(p => p.time > 19800).sort((a,b)=>a.time-b.time);
        
        const uRsi = [];
        let lrTime = -1;
        for (const p of rsi) {
            if (p.time > lrTime) { uRsi.push(p); lrTime = p.time; }
        }
        
        try {
            appState.series.rsiLine.setData(uRsi);
            // Update legend
            if (uRsi.length > 0) {
                const lastRsi = uRsi[uRsi.length - 1];
                const rsiEl = document.getElementById('legend-rsi');
                if (rsiEl) rsiEl.textContent = `RSI: ${lastRsi.value.toFixed(1)}`;
            }
        } catch(e) {
            console.error("RSI setData error:", e);
        }
    }

    // SuperTrend
    if (data.ml_adaptive_st && data.ml_adaptive_st.time) {
        const stData = data.ml_adaptive_st.time.map((t, i) => ({
            time: t + 19800,
            value: parseFloat(data.ml_adaptive_st.value[i])
        })).sort((a,b) => a.time - b.time);
        
        // Remove duplicates
        const uniqueST = [];
        let lastSTTime = -1;
        for (const s of stData) {
            if (s.time > lastSTTime) {
                uniqueST.push(s);
                lastSTTime = s.time;
            }
        }
        try {
            appState.series.stLine.setData(uniqueST);
        } catch(e) { }
    }

    // Indicators & Markers
    const combinedMarkers = [];
    const validTimes = new Set(ohlc.map(c => c.time));

    // 1. LZ Signals
    if (data.ml_lorentzian) {
        data.ml_lorentzian.filter(l => l.signal !== 0).forEach(l => {
            const time = (l.time || 0) + 19800;
            if (validTimes.has(time)) {
                combinedMarkers.push({
                    time: time,
                    position: l.signal === 1 ? 'belowBar' : 'aboveBar',
                    color: l.signal === 1 ? '#22c55e' : '#ef4444',
                    shape: l.signal === 1 ? 'arrowUp' : 'arrowDown',
                    text: 'LZ'
                });
            }
        });
    }

    // 2. SuperTrend Flips
    if (data.ml_adaptive_st && data.ml_adaptive_st.trend) {
        const trend = data.ml_adaptive_st.trend;
        const stTime = data.ml_adaptive_st.time;
        for (let i = 1; i < trend.length; i++) {
            if (trend[i] !== trend[i-1]) {
                const time = stTime[i] + 19800;
                if (validTimes.has(time)) {
                    combinedMarkers.push({
                        time: time,
                        position: trend[i] === 1 ? 'belowBar' : 'aboveBar',
                        color: trend[i] === 1 ? '#34d399' : '#f87171',
                        shape: trend[i] === 1 ? 'arrowUp' : 'arrowDown',
                        text: 'ST'
                    });
                }
            }
        }
    }

    if (combinedMarkers.length > 0) {
        try {
            // Sort and deduplicate markers by time
            combinedMarkers.sort((a,b) => a.time - b.time);
            const uniqueMarkers = [];
            let lastMTime = -1;
            for (const m of combinedMarkers) {
                if (m.time > lastMTime) {
                    uniqueMarkers.push(m);
                    lastMTime = m.time;
                } else if (m.time === lastMTime) {
                    // Overlay text if multiple indicators trigger
                    uniqueMarkers[uniqueMarkers.length-1].text += `+${m.text}`;
                }
            }
            appState.series.candles.setMarkers(uniqueMarkers);
        } catch(e) { console.error("Marker error:", e); }
    }
    if (data.vwap) {
        const vwapEl = document.getElementById('legend-vwap');
        if (vwapEl) vwapEl.textContent = `VWAP: ₹${parseFloat(data.vwap).toFixed(2)}`;
    }
}

function updateDaySummary() {
    let pnl = 0;
    appState.openTrades.forEach(t => pnl += (t.pnl || 0));
    appState.closedTrades.forEach(t => pnl += (t.pnl || 0));
    const pnlEl = document.getElementById('ds-pnl');
    if (pnlEl) {
        pnlEl.textContent = `₹${pnl.toFixed(2)}`;
        pnlEl.className = `ds-value ${pnl >= 0 ? 'stat-up' : 'stat-down'}`;
    }
    const quotaEl = document.getElementById('ds-quota-text');
    if (quotaEl) quotaEl.textContent = `${appState.aiCallsToday}/7`;
}

function handleLiveTick(tick) {
    if (!tick) return;
    // Update dashboard positions if match
    appState.openTrades.forEach(t => {
        if (t.ticker === tick.key || t.ticker === (tick.symbol + '.NS') || t.ticker === tick.symbol) {
            t.current_price = tick.ltp;
            // P&L Recalc
            if (t.action === 'BUY') t.pnl = (t.current_price - t.entry_price) * t.quantity;
            else t.pnl = (t.entry_price - t.current_price) * t.quantity;
        }
    });
    renderPositions();
    updateDaySummary();
    
    // Update Chart if currently viewing this ticker
    if (appState.currentChartKey === tick.key || appState.currentChartKey === tick.symbol || appState.currentChartKey === (tick.symbol + '.NS')) {
        if (appState.series.candles) {
            appState.series.candles.update({
                time: (tick.timestamp || Math.floor(Date.now()/1000)) + 19800,
                open: tick.open || tick.ltp,
                high: tick.high || tick.ltp,
                low: tick.low || tick.ltp,
                close: tick.ltp
            });
            // Update legend
            const ltpEl = document.getElementById('legend-ltp');
            if (ltpEl) ltpEl.textContent = `LTP: ₹${tick.ltp.toFixed(2)}`;
        }
    }
}

function handleBacktestResults(results) {
    const container = document.getElementById('backtest-results-container');
    if (!container) return;
    document.getElementById('run-backtest-btn').disabled = false;
    document.getElementById('run-backtest-btn').textContent = 'Run Backtest';
    
    if (results.error) { container.innerHTML = `<div class="error-msg">${results.error}</div>`; return; }
    
    const tradeLog = results.trade_log || [];
    
    container.innerHTML = `
        <div class="bt-summary-card">
            <div class="bt-stat"><span>Win Rate</span> <strong>${results.win_rate}%</strong></div>
            <div class="bt-stat"><span>Net Profit</span> <strong>₹${results.net_profit.toLocaleString()}</strong></div>
            <div class="bt-stat"><span>Total Trades</span> <strong>${results.total_trades}</strong></div>
            <div class="bt-stat"><span>Max Drawdown</span> <strong class="text-danger">${results.max_drawdown_pct}%</strong></div>
        </div>
        <div class="bt-log-wrapper" style="margin-top:20px; max-height:400px; overflow-y:auto;">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Entry Time</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Exit Time</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    ${tradeLog.map(t => `
                        <tr>
                            <td>${t.entry_time}</td>
                            <td><span class="badge-${t.type.toLowerCase()}">${t.type}</span></td>
                            <td>₹${t.entry_price.toFixed(2)}</td>
                            <td>${t.exit_time}</td>
                            <td>₹${t.exit_price.toFixed(2)}</td>
                            <td class="${t.pnl_pct >= 0 ? 'text-success' : 'text-danger'}">${(t.pnl_pct * 100).toFixed(2)}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderPositions() {
    const container = document.getElementById('positions-container');
    const totalPnlEl = document.getElementById('total-pnl');
    if (!container) return;

    if (appState.openTrades.length === 0) {
        container.innerHTML = '<div class="empty-state">No open positions.</div>';
        if (totalPnlEl) totalPnlEl.textContent = '₹0.00';
        return;
    }

    let html = '';
    let totalPnl = 0;
    appState.openTrades.forEach(t => {
        const pnl = t.pnl || 0; totalPnl += pnl;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const cssClass = t.action === 'SHORT SELL' ? 'short' : 'buy';
        const pnlPct = (t.entry_price > 0 && t.quantity > 0) ? ((pnl / (t.entry_price * t.quantity)) * 100).toFixed(2) : '0.00';

        html += `<div class="position-card ${cssClass}">
            <div class="pos-header">
                <div class="pos-title">
                    <span class="pos-ticker clickable-ticker" onclick="openChart('${t.ticker}')">${t.ticker.replace('.NS', '')}</span>
                    <span class="badge-${cssClass}">${t.action}</span>
                </div>
                <div class="pos-pnl-group">
                    <span class="pos-pnl-pct ${pnlClass}">${pnl >= 0 ? '+' : ''}${pnlPct}%</span>
                    <span class="pos-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</span>
                </div>
            </div>
            <div class="pos-grid">
                <div class="pos-stat"><label>Entry</label><span>₹${t.entry_price.toFixed(2)}</span></div>
                <div class="pos-stat"><label>Current</label><span class="${pnlClass}">₹${(t.current_price || t.entry_price).toFixed(2)}</span></div>
                <div class="pos-stat"><label>SL</label><span class="sl-val">₹${t.stop_loss.toFixed(2)}</span></div>
                <div class="pos-stat"><label>Qty</label><span>${t.quantity}</span></div>
            </div>
            <div class="pos-actions">
                <button class="btn chart-btn" onclick="openChart('${t.ticker}')">📈 Chart</button>
                <button class="btn close-btn" onclick="closeTrade('${t.id}', ${t.current_price || t.entry_price})">Close</button>
            </div>
        </div>`;
    });
    container.innerHTML = html;
    if (totalPnlEl) totalPnlEl.textContent = `₹${totalPnl.toFixed(2)}`;
}

function updateModelDropdown() {
    const prov = document.getElementById('ai-provider-input')?.value;
    if (prov && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'list_models', provider: prov }));
    }
}

function closeTrade(id, price) {
    ws.send(JSON.stringify({ action: 'close_trade', trade_id: id, exit_price: price }));
}

function showToast(msg, level='info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const t = document.createElement('div');
    t.className = `toast toast-${level}`;
    t.textContent = msg;
    container.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function initClock() {
    setInterval(() => {
        const el = document.getElementById('ist-clock');
        if (el) el.textContent = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
    }, 1000);
}

function renderTradeHistory() {
    const body = document.getElementById('trades-body');
    if (!body) return;
    body.innerHTML = appState.closedTrades.slice(0, 5).map(t => `
        <tr>
            <td class="clickable-ticker" onclick="openChart('${t.ticker}')">${t.ticker.replace('.NS','')}</td>
            <td class="${(t.pnl||0)>=0?'text-success':'text-danger'}">₹${(t.pnl||0).toFixed(2)}</td>
        </tr>
    `).join('');
}

function renderFullHistory() {
    const body = document.getElementById('history-body');
    const summary = document.getElementById('history-summary');
    if (!body) return;
    let totalPnl = 0; let wins = 0;
    body.innerHTML = appState.closedTrades.map(t => {
        totalPnl += t.pnl || 0; if((t.pnl||0)>0) wins++;
        return `<tr>
            <td>${new Date(t.close_time).toLocaleDateString('en-IN')}</td>
            <td class="clickable-ticker" onclick="openChart('${t.ticker}')">${t.ticker}</td>
            <td>${t.action}</td>
            <td>${t.quantity}</td>
            <td>₹${t.entry_price.toFixed(2)}</td>
            <td>₹${t.exit_price.toFixed(2)}</td>
            <td class="${(t.pnl||0)>=0?'text-success':'text-danger'}">₹${(t.pnl||0).toFixed(2)}</td>
        </tr>`;
    }).join('');
    const wr = appState.closedTrades.length ? ((wins/appState.closedTrades.length)*100).toFixed(1) : '0';
    if (summary) summary.innerHTML = `<div class="h-stat">Trades: <strong>${appState.closedTrades.length}</strong></div><div class="h-stat">Win Rate: <strong>${wr}%</strong></div><div class="h-stat">Net P&L: <strong>₹${totalPnl.toFixed(2)}</strong></div>`;
}

function renderHistoryScans(scans) {
    const container = document.getElementById('history-scans-container');
    if (!scans || scans.length === 0) {
        container.innerHTML = '<div class="empty-state">No AI scans saved.</div>';
        return;
    }

    let html = '';
    scans.forEach(scan => {
        const time = scan.timestamp ? new Date(scan.timestamp).toLocaleString('en-IN') : '--';
        let resultArr = [];
        try {
            resultArr = JSON.parse(scan.output_json) || [];
            if (!Array.isArray(resultArr) && resultArr.data) resultArr = resultArr.data;
        } catch (e) { }

        if (!Array.isArray(resultArr)) resultArr = [];

        html += `<div style="background:var(--bg-lighter); border:1px solid var(--border-color); border-radius:8px; padding:12px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px; color:var(--text-secondary);">
                <span>🕒 ${time} | 🤖 ${scan.model_used}</span>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">`;
            
        if (resultArr.length === 0) {
            html += `<span style="color:var(--text-secondary);">No actionable setups.</span>`;
        } else {
            resultArr.forEach(r => {
                const actionColor = r.action?.includes('BUY') ? 'var(--success)' : 'var(--danger)';
                html += `
                <div style="border:1px solid ${actionColor}; border-radius:4px; padding:8px; flex:1; min-width:200px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <strong class="clickable-ticker" onclick="openChart('${r.ticker}')" style="color:${actionColor}">${r.ticker}</strong>
                        <span style="font-size:11px; padding:2px 6px; border-radius:4px; background:${actionColor}20; color:${actionColor};">${r.action}</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-primary);">${r.reasoning || r.reason || ''}</div>
                </div>`;
            });
        }
        html += `</div></div>`;
    });
    container.innerHTML = html;
}

function openChart(ticker) {
    const modal = document.getElementById('chart-modal');
    if (modal) modal.style.display = 'flex';
    document.getElementById('chart-title').textContent = ticker;
    appState.currentChartKey = ticker;
    ws.send(JSON.stringify({ action: 'get_chart_data', ticker: ticker, interval: appState.currentInterval }));
}

function closeChart() {
    const modal = document.getElementById('chart-modal');
    if (modal) modal.style.display = 'none';
    const container = document.getElementById('chart-container');
    const adxContainer = document.getElementById('adx-container');
    if (container) container.innerHTML = '';
    if (adxContainer) adxContainer.innerHTML = '';
    appState.chartInstance = null;
    appState.adxChart = null;
}

function changeChartInterval(interval) {
    appState.currentInterval = interval;
    // Update active button state
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    if (window.event && window.event.target) {
        window.event.target.classList.add('active');
    }
    ws.send(JSON.stringify({ action: 'get_chart_data', ticker: appState.currentChartKey, interval: interval }));
}

function toggleIndicator(indicator, show) {
    // Basic stub for indicator toggling
    if (indicator === 'lz' && appState.series.candles) {
        // We'd ideally re-render markers, but for now we just trigger a refresh
        ws.send(JSON.stringify({ action: 'get_chart_data', ticker: appState.currentChartKey, interval: appState.currentInterval }));
    }
}

function toggleIndicatorSettings(indicator) {
    const panels = document.querySelectorAll('.indicator-settings-overlay');
    panels.forEach(p => p.style.display = 'none');
    const panel = document.getElementById(`${indicator}-settings`);
    if (panel) panel.style.display = 'block';
}

function updateIndicatorParams() {
    // Hide settings overlays
    document.querySelectorAll('.indicator-settings-overlay').forEach(p => p.style.display = 'none');
    ws.send(JSON.stringify({ action: 'get_chart_data', ticker: appState.currentChartKey, interval: appState.currentInterval }));
}

function updateRiskWarningUI() {
    const dataF = document.getElementById('fallback-data-input')?.checked;
    const searchF = document.getElementById('fallback-search-input')?.checked;
    const aiF = document.getElementById('fallback-ai-input')?.checked;
    const warning = document.getElementById('fallback-danger-zone');
    if (warning) warning.style.display = (!dataF || !searchF || !aiF) ? 'block' : 'none';
}
