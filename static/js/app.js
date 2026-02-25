const ws = new WebSocket(`ws://${window.location.host}/ws`);

// DOM Elements
const globalIndicesEl = document.getElementById('global-indices');
const vixValEl = document.getElementById('vix-val');
const istClockEl = document.getElementById('ist-clock');
const sessionNameEl = document.getElementById('session-name');
const wsStatusEl = document.getElementById('ws-status');
const lastUpdatedTimeEl = document.getElementById('last-updated-time');
const signalCardsContainer = document.getElementById('signal-cards-container');
const toastContainer = document.getElementById('toast-container');

// Settings & Panel Elements
const capitalInput = document.getElementById('capital-input');
const riskInput = document.getElementById('risk-input');
const searchEngineInput = document.getElementById('search-engine-input');
const dataProviderInput = document.getElementById('data-provider-input');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const triggerScanBtn = document.getElementById('trigger-scan-btn');
const magicScanBtn = document.getElementById('magic-scan-btn');

// P&L Elements
const totalPnlEl = document.getElementById('total-pnl');
const openCountEl = document.getElementById('open-count');
const closedCountEl = document.getElementById('closed-count');
const openTradesList = document.getElementById('open-trades-list');
const closedTradesBody = document.getElementById('closed-trades-body');

// Chart Elements
const chartModal = document.getElementById('chart-modal');
const closeChartBtn = document.getElementById('close-chart-btn');
const chartTitle = document.getElementById('chart-modal-title');
const chartContainer = document.getElementById('tv-chart-container');
let tvChart = null;
let candlestickSeries = null;

let appState = {
    openTrades: [],
    closedTrades: [],
    aiSignals: [],
    globalContext: {},
    customStrategies: [],
    searchEngine: 'gemini',
    dataProvider: 'yfinance',
    lastScanType: 'standard',
    dashboardStocks: new Set()
};

async function fetchDashboardStocks() {
    try {
        const res = await fetch('/api/dashboard/stocks');
        const data = await res.json();
        appState.dashboardStocks = new Set(data.map(t => t.replace('.NS', '')));
    } catch (e) {
        console.error(e);
    }
}
fetchDashboardStocks();

// Start local clock immediately
setInterval(() => {
    const now = new Date();
    istClockEl.textContent = now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
}, 1000);

ws.onopen = () => {
    wsStatusEl.classList.add('connected');
    showToast('Connected to Trading Engine', 'success');
};

ws.onclose = () => {
    wsStatusEl.classList.remove('connected');
    showToast('Connection lost! Reconnecting...', 'error');
    setTimeout(() => window.location.reload(), 5000);
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'state_update') {
        appState.capital = data.capital;
        appState.maxLoss = data.max_loss;
        if (data.search_engine) appState.searchEngine = data.search_engine;
        if (data.data_provider) appState.dataProvider = data.data_provider;

        appState.openTrades = data.open_trades;
        appState.closedTrades = data.closed_trades;
        appState.aiSignals = data.ai_signals;
        appState.globalContext = data.global_context;

        capitalInput.value = appState.capital;
        riskInput.value = appState.maxLoss;
        if (searchEngineInput) searchEngineInput.value = appState.searchEngine;
        if (dataProviderInput) dataProviderInput.value = appState.dataProvider;

        renderGlobalContext();
        renderSignals();
        renderTrades();
    }
    else if (data.type === 'heartbeat') {
        sessionNameEl.textContent = `Session: ${data.session}`;
        appState.globalContext = data.global_context;
        appState.openTrades = data.open_trades;
        renderGlobalContext();
        renderTrades(); // Updates live PNL
    }
    else if (data.type === 'notification') {
        showToast(data.message, data.level);
    }
    else if (data.type === 'trades_update') {
        if (data.open_trades) appState.openTrades = data.open_trades;
        if (data.closed_trades) appState.closedTrades = data.closed_trades;
        renderTrades();
    }
};

// UI Triggers
saveSettingsBtn.addEventListener('click', () => {
    ws.send(JSON.stringify({
        action: 'update_settings',
        capital: capitalInput.value,
        max_loss: riskInput.value,
        search_engine: searchEngineInput ? searchEngineInput.value : 'gemini',
        data_provider: dataProviderInput ? dataProviderInput.value : 'yfinance'
    }));
    showToast('Settings Updated', 'success');
});

triggerScanBtn.addEventListener('click', () => {
    triggerScanBtn.disabled = true;
    triggerScanBtn.innerHTML = '<span class="icon">⌛</span> Scanning NSE...';
    // Clear old signals
    document.getElementById('standard-pane').style.display = 'block';
    document.getElementById('magic-pane').style.display = 'none';
    appState.lastScanType = 'standard';
    signalCardsContainer.innerHTML = '<div class="empty-state massive">Scanning NSE universe and analyzing technicals + news using AI...</div>';

    const strategyRadio = document.querySelector('input[name="strategy"]:checked');
    const strategy = strategyRadio ? strategyRadio.value : 's1';
    let custom_rules = null;
    if (strategy.startsWith('cs_') && strategyRadio.getAttribute('data-rules')) {
        custom_rules = atob(strategyRadio.getAttribute('data-rules'));
    }

    ws.send(JSON.stringify({ action: 'trigger_scan', strategy, custom_rules }));

    // Re-enable after 30s as fallback
    setTimeout(() => {
        triggerScanBtn.disabled = false;
        triggerScanBtn.innerHTML = '<span class="icon">⚡</span> Standard AI Screen';
    }, 60000);
});

magicScanBtn.addEventListener('click', () => {
    magicScanBtn.disabled = true;
    magicScanBtn.innerHTML = '<span class="icon">⌛</span> Running Magic Pipeline...';

    document.getElementById('standard-pane').style.display = 'none';
    document.getElementById('magic-pane').style.display = 'block';
    appState.lastScanType = 'magic';

    const magicAiCards = document.getElementById('magic-ai-cards');
    const magicWlCards = document.getElementById('magic-wl-cards');
    magicAiCards.innerHTML = '<div class="empty-state massive">Vectorizing top 10 stocks. Backtesting 7 strategies. Analyzing news...</div>';
    magicWlCards.innerHTML = '<div class="empty-state massive">Fetching watchlist technicals...</div>';

    fetchDashboardStocks(); // Refresh WL

    const strategyRadio = document.querySelector('input[name="strategy"]:checked');
    const strategy = strategyRadio ? strategyRadio.value : 's1';
    let custom_rules = null;
    if (strategy.startsWith('cs_') && strategyRadio.getAttribute('data-rules')) {
        custom_rules = atob(strategyRadio.getAttribute('data-rules'));
    }

    ws.send(JSON.stringify({ action: 'trigger_scan', strategy, custom_rules }));

    setTimeout(() => {
        magicScanBtn.disabled = false;
        magicScanBtn.innerHTML = '<span class="icon">✨</span> Magic AI Recommendations';
    }, 60000);
});

// Custom Strategy Logic
const csModal = document.getElementById('custom-strategy-modal');
const csNameInput = document.getElementById('cs-name');
const csRulesInput = document.getElementById('cs-rules');
const csListContainer = document.getElementById('custom-strategies-list');

function openCustomStrategyModal() {
    csModal.style.display = 'flex';
}

function closeCustomStrategyModal() {
    csModal.style.display = 'none';
    csNameInput.value = '';
    csRulesInput.value = '';
}

function saveCustomStrategy() {
    const name = csNameInput.value.trim();
    const rules = csRulesInput.value.trim();
    if (!name || !rules) {
        showToast('Please enter both name and rules', 'error');
        return;
    }

    // Create safe ID
    const id = 'cs_' + Date.now();
    appState.customStrategies.push({ id, name, rules });

    renderCustomStrategies();
    closeCustomStrategyModal();
    showToast('Custom Strategy Saved!', 'success');
}

function renderCustomStrategies() {
    csListContainer.innerHTML = '';
    appState.customStrategies.forEach(cs => {
        const label = document.createElement('label');
        label.className = 'strategy-option custom-algo';
        label.style.display = 'flex';
        label.style.justifyContent = 'space-between';

        label.innerHTML = `
            <div>
                <input type="radio" name="strategy" value="${cs.id}" data-rules="${btoa(cs.rules)}">
                <span style="color:#a78bfa;">${cs.name}</span>
            </div>
            <button class="nav-btn" style="padding:0; min-height:0; color:var(--color-sell);" onclick="removeCustomStrategy('${cs.id}')">&times;</button>
        `;
        csListContainer.appendChild(label);
    });
}

function removeCustomStrategy(id) {
    appState.customStrategies = appState.customStrategies.filter(cs => cs.id !== id);
    renderCustomStrategies();
}

// Rendering Functions
function renderGlobalContext() {
    if (!appState.globalContext || Object.keys(appState.globalContext).length === 0) return;

    // Helper to generate HTML for a dictionary of tickers
    const generateTickerHtml = (tickerDict) => {
        let html = '';
        if (!tickerDict) return html;
        for (const [name, data] of Object.entries(tickerDict)) {
            const dirClass = data.change >= 0 ? 'up' : 'down';
            const arrow = data.change >= 0 ? '▲' : '▼';
            html += `
                <div class="ticker-item">
                    <span>${name}</span>
                    <span class="ticker-val ${dirClass}">${data.value || 0}</span>
                    <span class="ticker-chg ${dirClass}">${arrow} ${data.change_pct}%</span>
                </div>
            `;
        }
        return html;
    };

    // Render VIX
    const vixData = appState.globalContext.vix;
    if (vixData && vixValEl) {
        vixValEl.innerHTML = `${vixData.value || '--'} <small>(${vixData.change_pct > 0 ? '+' : ''}${vixData.change_pct}%)</small>`;
        vixValEl.parentElement.style.color = vixData.value > 18 ? 'var(--vix-fear)' : (vixData.value > 14 ? 'var(--vix-elevated)' : 'var(--vix-calm)');
    }

    // Render Global & India Rows
    const globalRow = document.querySelector('#row-global .ticker-content');
    const indiaRow = document.querySelector('#row-india .ticker-content');

    if (globalRow && appState.globalContext.global) {
        globalRow.innerHTML = generateTickerHtml(appState.globalContext.global);
    }
    if (indiaRow && appState.globalContext.india) {
        indiaRow.innerHTML = generateTickerHtml(appState.globalContext.india);
    }
}

function renderSignals() {
    if (triggerScanBtn.disabled && appState.aiSignals.length > 0) {
        triggerScanBtn.disabled = false;
        triggerScanBtn.innerHTML = '<span class="icon">⚡</span> Run AI Screen';
        lastUpdatedTimeEl.textContent = `Last Scanned: ${new Date().toLocaleTimeString()}`;
    }

    if (!appState.aiSignals || appState.aiSignals.length === 0) {
        signalCardsContainer.innerHTML = '<div class="empty-state massive">Click "Run AI Screen" to discover trades.</div>';
        return;
    }

    const magicAiCards = document.getElementById('magic-ai-cards');
    const magicWlCards = document.getElementById('magic-wl-cards');

    if (appState.lastScanType === 'standard') {
        signalCardsContainer.innerHTML = '';
    } else {
        magicAiCards.innerHTML = '';
        magicWlCards.innerHTML = '';
    }

    let magicAiCount = 0;
    const template = document.getElementById('signal-card-template');

    appState.aiSignals.forEach(signal => {
        const ai = signal.ai_recommendation;
        const ta = signal.ta_data;
        const clone = template.content.cloneNode(true);
        const card = clone.querySelector('.signal-card');

        // Header
        const tickerEl = clone.querySelector('.ticker-name');
        const cleanTicker = signal.ticker.replace('.NS', '');
        tickerEl.textContent = cleanTicker;

        // Add chart click listener
        tickerEl.onclick = () => openChart(cleanTicker);

        const sentBadge = clone.querySelector('.sentiment-badge');
        sentBadge.textContent = signal.sentiment.sentiment + ' NEWS';
        sentBadge.classList.add(signal.sentiment.sentiment);
        sentBadge.title = signal.sentiment.reason;

        // Price & Action
        const priceEl = clone.querySelector('.current-price');
        priceEl.textContent = `₹${ta.close.toFixed(2)}`;
        // Flash animation
        priceEl.classList.add('flash-update');
        setTimeout(() => priceEl.classList.remove('flash-update'), 1000);

        const recBadge = clone.querySelector('.recommendation-badge');
        recBadge.textContent = ai.action;
        recBadge.classList.add(ai.action.replace(' ', ''));

        // Score
        const score = ai.ai_composite_score || 0;
        clone.querySelector('.score-fill').style.width = `${score}%`;
        clone.querySelector('.score-text').textContent = `AI Confidence: ${score}/100`;

        // Explanation
        clone.querySelector('.explanation').textContent = ai.explanation || 'No explanation provided.';

        // Trade Levels
        const entryVal = ai.entry_price ? `₹${ai.entry_price.toFixed(2)}` : '--';
        const slVal = ai.stop_loss ? `₹${ai.stop_loss.toFixed(2)}` : '--';
        const t1Val = ai.target_1 ? `₹${ai.target_1.toFixed(2)}` : '--';
        const t2Val = ai.target_2 ? `₹${ai.target_2.toFixed(2)}` : '--';

        clone.querySelector('.entry-val').textContent = entryVal;
        clone.querySelector('.sl-val').textContent = slVal;
        clone.querySelector('.t1-val').textContent = t1Val;
        clone.querySelector('.t2-val').textContent = t2Val;
        clone.querySelector('.qty-val').textContent = ai.recommended_quantity || '--';

        // Fundamentals
        const funContainer = clone.querySelector('.fundamentals');
        if (signal.fundamentals) {
            const f = signal.fundamentals;
            const mktCap = f.market_cap !== 'N/A' && f.market_cap ? '₹' + (f.market_cap / 10000000).toFixed(2) + 'Cr' : 'N/A';
            const pe = f.pe_ratio !== 'N/A' && f.pe_ratio ? f.pe_ratio.toFixed(1) : 'N/A';
            const div = f.dividend_yield !== 'N/A' && f.dividend_yield ? (f.dividend_yield * 100).toFixed(1) + '%' : 'N/A';

            funContainer.innerHTML = `
                <div><span class="text-muted">Mkt Cap:</span> ${mktCap}</div>
                <div><span class="text-muted">P/E:</span> ${pe}</div>
                <div style="grid-column: span 2;"><span class="text-muted">Sector:</span> ${f.sector || 'N/A'}</div>
            `;
        } else {
            funContainer.innerHTML = `<div style="grid-column: span 2; text-align: center; color: var(--text-muted);">No fundamentals available</div>`;
        }

        // TA Chips
        const chipsContainer = clone.querySelector('.ta-chips');
        const chips = [
            { label: 'RSI', val: ta.rsi_14.toFixed(1), state: ta.rsi_14 > 70 ? 'bearish' : (ta.rsi_14 < 30 ? 'bullish' : '') },
            { label: 'MACD', val: ta.macd.toFixed(2), state: ta.macd > ta.macd_signal ? 'bullish' : 'bearish' },
            { label: 'VWAP', val: ta.vwap.toFixed(2), state: ta.close > ta.vwap ? 'bullish' : 'bearish' },
            { label: 'Vol Surge', val: ta.vol_surge + 'x', state: ta.vol_surge > 1.5 ? 'bullish' : '' },
            { label: 'ADX', val: ta.adx_14.toFixed(1), state: ta.adx_14 > 25 ? 'bullish' : '' } // Strong trend
        ];

        chips.forEach(c => {
            chipsContainer.innerHTML += `<span class="chip ${c.state}">${c.label}: ${c.val}</span>`;
        });

        // Log Trade Button
        const logBtn = clone.querySelector('.log-trade-btn');
        if (ai.action === 'BUY' || ai.action === 'SHORT SELL') {
            logBtn.classList.add('primary');
            logBtn.onclick = () => {
                ws.send(JSON.stringify({
                    action: 'log_trade',
                    ticker: signal.ticker,
                    trade_action: ai.action,
                    qty: ai.recommended_quantity || 1,
                    entry_price: ta.close, // log current market price as entry for reality
                    sl: ai.stop_loss,
                    t1: ai.target_1,
                    t2: ai.target_2
                }));
            };
        } else {
            logBtn.style.display = 'none'; // Only allow 1-click log for real signals
        }

        if (appState.lastScanType === 'standard') {
            signalCardsContainer.appendChild(clone);
        } else {
            // Magic Pane Logic
            if (appState.dashboardStocks.has(cleanTicker)) {
                magicWlCards.appendChild(clone);
            } else if (magicAiCount < 3) {
                magicAiCards.appendChild(clone);
                magicAiCount++;
            }
        }
    });

    if (appState.lastScanType === 'magic') {
        if (magicWlCards.children.length === 0) {
            magicWlCards.innerHTML = '<div class="text-muted" style="text-align:center; padding: 2rem;">No watchlist stocks found in this scan. Add some from the Watchlists page!</div>';
        }
        if (magicAiCards.children.length === 0) {
            magicAiCards.innerHTML = '<div class="text-muted" style="text-align:center; padding: 2rem;">No top recommendations met the criteria.</div>';
        }
    }
}

function renderTrades() {
    openCountEl.textContent = appState.openTrades.length;
    closedCountEl.textContent = appState.closedTrades.length;

    // Calculate total layout PNL
    let totalPnl = 0;

    // Render Open Trades
    openTradesList.innerHTML = '';
    if (appState.openTrades.length === 0) {
        openTradesList.innerHTML = '<div class="empty-state">No open positions</div>';
    } else {
        appState.openTrades.forEach(t => {
            const pnl = t.pnl || 0;
            totalPnl += pnl;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            const sign = pnl >= 0 ? '+' : '';

            openTradesList.innerHTML += `
                <div class="open-trade-card ${t.action.replace(' ', '')}">
                    <div class="ot-header">
                        <span class="ot-ticker">${t.ticker.replace('.NS', '')} (${t.action})</span>
                        <span class="ot-pnl ${pnlClass}">${sign}₹${pnl}</span>
                    </div>
                    <div class="ot-details">
                        <span>Entry: ₹${t.entry_price.toFixed(2)}</span>
                        <span>Current: ₹${(t.current_price || t.entry_price).toFixed(2)}</span>
                        <span>SL: ₹${t.stop_loss.toFixed(2)}</span>
                        <span>Qty: ${t.quantity}</span>
                    </div>
                    <button class="btn close-btn" onclick="closeTrade('${t.id}', ${t.current_price || t.entry_price})">Close Position</button>
                </div>
            `;

            // Intelligent local alerts for SL/Target hits
            checkTradeAlerts(t);
        });
    }

    // Render Closed Trades
    closedTradesBody.innerHTML = '';
    appState.closedTrades.forEach(t => {
        const pnl = t.pnl || 0;
        totalPnl += pnl;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const sign = pnl >= 0 ? '+' : '';

        closedTradesBody.innerHTML += `
            <tr>
                <td>${t.ticker.replace('.NS', '')}</td>
                <td>${t.action}</td>
                <td>₹${t.entry_price.toFixed(2)}</td>
                <td>₹${t.exit_price.toFixed(2)}</td>
                <td>${t.quantity}</td>
                <td class="ot-pnl ${pnlClass}">${sign}₹${pnl.toFixed(2)}</td>
            </tr>
        `;
    });

    // Update Total PNL Header
    totalPnlEl.textContent = `₹${totalPnl.toFixed(2)}`;
    totalPnlEl.className = `pnl-huge ${totalPnl >= 0 ? 'positive' : 'negative'}`;
}

// Ensure alerts only trigger once per trade per level
const handledAlerts = new Set();

function checkTradeAlerts(t) {
    const cp = t.current_price;
    if (!cp) return;

    if (t.action === 'BUY') {
        if (cp <= t.stop_loss && !handledAlerts.has(`${t.id}-sl`)) {
            showToast(`STOP LOSS HIT for ${t.ticker}! Close position.`, 'error');
            handledAlerts.add(`${t.id}-sl`);
        } else if (cp >= t.target_1 && !handledAlerts.has(`${t.id}-t1`)) {
            showToast(`TARGET 1 HIT for ${t.ticker}! Consider trailing SL.`, 'success');
            handledAlerts.add(`${t.id}-t1`);
        } else if (cp >= t.target_2 && !handledAlerts.has(`${t.id}-t2`)) {
            showToast(`TARGET 2 HIT! Full profit target achieved for ${t.ticker}.`, 'success');
            handledAlerts.add(`${t.id}-t2`);
        }
    } else if (t.action === 'SHORT SELL') {
        if (cp >= t.stop_loss && !handledAlerts.has(`${t.id}-sl`)) {
            showToast(`STOP LOSS HIT for ${t.ticker}! Close position.`, 'error');
            handledAlerts.add(`${t.id}-sl`);
        } else if (cp <= t.target_1 && !handledAlerts.has(`${t.id}-t1`)) {
            showToast(`TARGET 1 HIT for ${t.ticker}! Consider trailing SL.`, 'success');
            handledAlerts.add(`${t.id}-t1`);
        } else if (cp <= t.target_2 && !handledAlerts.has(`${t.id}-t2`)) {
            showToast(`TARGET 2 HIT! Full profit target achieved for ${t.ticker}.`, 'success');
            handledAlerts.add(`${t.id}-t2`);
        }
    }
}

// Tab Navigation
const btnDashboard = document.getElementById('nav-dashboard-btn');
const btnBacktest = document.getElementById('nav-backtest-btn');
const secDashboard = document.querySelector('main.main-content:not(#backtest-lab-section)');
const secBacktest = document.getElementById('backtest-lab-section');
const sidebar = document.querySelector('.sidebar');

if (btnDashboard && btnBacktest) {
    btnDashboard.onclick = () => {
        btnDashboard.classList.add('active');
        btnBacktest.classList.remove('active');
        secDashboard.style.display = 'block';
        sidebar.style.display = 'flex';
        secBacktest.style.display = 'none';
    };
    btnBacktest.onclick = () => {
        btnBacktest.classList.add('active');
        btnDashboard.classList.remove('active');
        secDashboard.style.display = 'none';
        sidebar.style.display = 'none'; // Hide trading sidebar for backtest view
        secBacktest.style.display = 'block';
    };
}

// Backtest Logic
const runBtBtn = document.getElementById('run-bt-btn');
const btTickerInput = document.getElementById('bt-ticker-input');
const btLoading = document.getElementById('bt-loading');
const btResults = document.getElementById('bt-results-container');

if (runBtBtn) {
    runBtBtn.onclick = async () => {
        const ticker = btTickerInput.value.toUpperCase().trim();
        if (!ticker) return;

        // UI Loading State
        runBtBtn.disabled = true;
        btResults.style.display = 'none';
        btLoading.style.display = 'block';

        try {
            const resp = await fetch(`/api/backtest/${ticker}`, { method: 'POST' });
            const data = await resp.json();

            if (data.status === 'success') {
                const bm = data.results.best_metrics;
                document.getElementById('bt-trades').textContent = bm.total_trades;
                document.getElementById('bt-winrate').textContent = bm.win_rate + '%';

                const profEl = document.getElementById('bt-profit');
                profEl.textContent = '₹' + bm.net_profit;
                profEl.style.color = bm.net_profit >= 0 ? '#10b981' : '#ef4444';

                document.getElementById('bt-mdd').textContent = bm.max_drawdown_pct + '%';

                document.getElementById('bt-params').textContent = JSON.stringify(data.results.best_parameters, null, 2);

                btResults.style.display = 'block';
                showToast(`AI successfully optimized ${ticker}`, 'success');
            } else {
                showToast(data.error || 'Backtest failed', 'error');
            }
        } catch (e) {
            showToast('Network error during backtest', 'error');
        } finally {
            runBtBtn.disabled = false;
            btLoading.style.display = 'none';
        }
    };
}

// Global scope to be called by onclick string
window.closeTrade = (id, price) => {
    ws.send(JSON.stringify({
        action: 'close_trade',
        trade_id: id,
        exit_price: price
    }));
};

function showToast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    toastContainer.appendChild(el);
    setTimeout(() => {
        if (el.parentNode) el.parentNode.removeChild(el);
    }, 5000);
}

// Chart Logic
closeChartBtn.onclick = () => {
    chartModal.style.display = 'none';
};

async function openChart(ticker) {
    chartTitle.textContent = `${ticker} - 5m Real-time Chart`;
    chartModal.style.display = 'flex';

    // Clear old chart
    if (tvChart) {
        tvChart.remove();
        tvChart = null;
    }

    // Create new chart
    tvChart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { type: 'solid', color: '#181b21' },
            textColor: '#e2e8f0',
        },
        grid: {
            vertLines: { color: '#334155' },
            horzLines: { color: '#334155' },
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            tickMarkFormatter: (time) => {
                const date = new Date(time * 1000);
                return date.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' });
            }
        },
        localization: {
            timeFormatter: (time) => {
                const date = new Date(time * 1000);
                return date.toLocaleString('en-IN', {
                    day: '2-digit', month: 'short',
                    hour: '2-digit', minute: '2-digit',
                    hour12: false, timeZone: 'Asia/Kolkata'
                });
            }
        }
    });

    candlestickSeries = tvChart.addCandlestickSeries({
        upColor: '#10b981', downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981', wickDownColor: '#ef4444',
    });

    try {
        const resp = await fetch(`/api/chart/${ticker}`);
        const result = await resp.json();

        if (result.data) {
            candlestickSeries.setData(result.data);

            // Handle Indicators dynamically
            const indicators = {
                'EMA 9': { data: result.ema9, color: '#3b82f6', series: null },
                'EMA 21': { data: result.ema21, color: '#f59e0b', series: null },
                'VWAP': { data: result.vwap, color: '#ec4899', series: null }
            };

            // Inject toggles into header
            const header = document.querySelector('.modal-header');
            let toggleHtml = '<div id="indicator-toggles" style="display:flex; gap:10px; margin-right:auto; margin-left:20px;">';
            Object.keys(indicators).forEach(key => {
                toggleHtml += `<button class="btn" id="toggle-${key.replace(' ', '')}" style="padding: 4px 8px; font-size:12px; border:1px solid ${indicators[key].color}; color:${indicators[key].color}; background:transparent;">${key}</button>`;
            });
            toggleHtml += '</div>';

            // Remove old toggles if exist
            const oldIdx = document.getElementById('indicator-toggles');
            if (oldIdx) oldIdx.remove();
            chartTitle.insertAdjacentHTML('afterend', toggleHtml);

            Object.keys(indicators).forEach(key => {
                if (indicators[key].data && indicators[key].data.length > 0) {
                    const lineSeries = tvChart.addLineSeries({
                        color: indicators[key].color,
                        lineWidth: 2,
                        crosshairMarkerVisible: false,
                    });
                    lineSeries.setData(indicators[key].data);
                    indicators[key].series = lineSeries;

                    // Setup toggle logic
                    const btn = document.getElementById(`toggle-${key.replace(' ', '')}`);
                    let isVisible = true;
                    btn.onclick = () => {
                        isVisible = !isVisible;
                        lineSeries.applyOptions({ visible: isVisible });
                        btn.style.background = isVisible ? 'transparent' : '#334155';
                    };
                }
            });

            tvChart.timeScale().fitContent();
        } else {
            showToast('Failed to load chart data', 'error');
        }
    } catch (e) {
        showToast('Error communicating with chart endpoint', 'error');
    }
}

// ─── Upstox Auth Helpers ───────────────────────────────────────────────────

async function checkUpstoxStatus() {
    const badge = document.getElementById('upstox-status-badge');
    if (!badge) return;
    try {
        const res = await fetch('/api/upstox/status');
        const data = await res.json();
        if (data.authenticated) {
            badge.textContent = '✅ Connected';
            badge.style.background = '#14532d';
            badge.style.color = '#4ade80';
        } else {
            badge.textContent = '🔴 Not Connected';
            badge.style.background = '#450a0a';
            badge.style.color = '#f87171';
        }
    } catch (e) {
        const b = document.getElementById('upstox-status-badge');
        if (b) b.textContent = '⚠️ Unreachable';
    }
}

async function connectUpstox() {
    try {
        const res = await fetch('/api/upstox/auth-url');
        const data = await res.json();
        if (data.auth_url) {
            window.location.href = data.auth_url;
        } else {
            showToast('Could not get Upstox auth URL', 'error');
        }
    } catch (e) {
        showToast('Error connecting to Upstox', 'error');
    }
}

// Wire up events — DOM is already ready since this script is at bottom of <body>
(function initUpstox() {
    const connectBtn = document.getElementById('upstox-connect-btn');
    if (connectBtn) {
        connectBtn.addEventListener('click', connectUpstox);
    }

    // Re-check status whenever settings panel is opened
    const settingsPanel = document.querySelector('details.settings-panel');
    if (settingsPanel) {
        settingsPanel.addEventListener('toggle', () => {
            if (settingsPanel.open) checkUpstoxStatus();
        });
    }

    // Run an initial status check
    checkUpstoxStatus();
})();
