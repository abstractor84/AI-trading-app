/**
 * V2 AI Trading Companion — Frontend
 * Features: Rich Stock Cards, Chart Viewer, Backtester Lab
 * Dynamic AI Provider→Model mapping, One-click trade, Enriched scan results
 */

// ─── WebSocket ──────────────────────────────────────────────────────
const ws = new WebSocket(`ws://${window.location.host}/ws`);

// ─── AI Provider → Model Mapping ────────────────────────────────────
const MODEL_MAP = {
    google: [
        { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
    ],
    groq: [
        { value: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B' },
        { value: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B Instant' },
        { value: 'mixtral-8x7b-32768', label: 'Mixtral 8x7B' },
    ],
    sambanova: [
        { value: 'Meta-Llama-3.3-70B-Instruct', label: 'Llama 3.3 70B Instruct' },
        { value: 'Meta-Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B Instruct' },
        { value: 'Meta-Llama-3.1-405B-Instruct', label: 'Llama 3.1 405B Instruct' },
    ],
};

function updateModelDropdown() {
    const provider = document.getElementById('ai-provider-input').value;
    const modelSelect = document.getElementById('ai-model-input');
    const models = MODEL_MAP[provider] || [];
    modelSelect.innerHTML = models.map(m =>
        `<option value="${m.value}">${m.label}</option>`
    ).join('');
}

// ─── DOM References ─────────────────────────────────────────────────
const marketPhaseBadge = document.getElementById('market-phase-badge');
const vixValEl = document.getElementById('vix-val');
const istClockEl = document.getElementById('ist-clock');
const wsStatusEl = document.getElementById('ws-status');
const marketPulseEl = document.getElementById('market-pulse');
const aiCallsBadge = document.getElementById('ai-calls-badge');
const phaseGuidance = document.getElementById('phase-guidance');
const aiResultSection = document.getElementById('ai-result-section');
const aiResultType = document.getElementById('ai-result-type');
const aiResultTime = document.getElementById('ai-result-time');
const aiResultBody = document.getElementById('ai-result-body');
const positionsContainer = document.getElementById('positions-container');
const totalPnlEl = document.getElementById('total-pnl');
const timelineContainer = document.getElementById('timeline-container');
const tradesBody = document.getElementById('trades-body');
const toastContainer = document.getElementById('toast-container');

// ─── App State ──────────────────────────────────────────────────────
const appState = { openTrades: [], closedTrades: [], globalContext: {}, marketPhase: {}, aiAdvisor: null, actionTimeline: [], chartInstance: null };

// ─── Init ───────────────────────────────────────────────────────────
document.getElementById('ai-provider-input').addEventListener('change', updateModelDropdown);
updateModelDropdown();

// ─── Clock ──────────────────────────────────────────────────────────
setInterval(() => {
    istClockEl.textContent = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
}, 1000);

// ─── Navigation ─────────────────────────────────────────────────────
const dashboardBtn = document.getElementById('nav-dashboard-btn');
const backtestBtn = document.getElementById('nav-backtest-btn');
const pageDashboard = document.getElementById('page-dashboard');
const pageBacktest = document.getElementById('page-backtest');

dashboardBtn.addEventListener('click', () => {
    dashboardBtn.classList.add('active'); backtestBtn.classList.remove('active');
    pageDashboard.style.display = ''; pageDashboard.classList.add('active');
    pageBacktest.style.display = 'none';
});
backtestBtn.addEventListener('click', () => {
    backtestBtn.classList.add('active'); dashboardBtn.classList.remove('active');
    pageBacktest.style.display = ''; pageDashboard.style.display = 'none';
    pageDashboard.classList.remove('active');
});

// ─── WebSocket Handlers ─────────────────────────────────────────────
ws.onopen = () => { wsStatusEl.className = 'status-indicator connected'; wsStatusEl.title = 'Connected'; };
ws.onclose = () => { wsStatusEl.className = 'status-indicator disconnected'; setTimeout(() => location.reload(), 3000); };

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
        case 'state_update': handleStateUpdate(msg); break;
        case 'trades_update':
            appState.openTrades = msg.open_trades || [];
            appState.closedTrades = msg.closed_trades || [];
            renderPositions(); renderTradeHistory(); break;
        case 'ai_advisor_update': handleAIAdvisorUpdate(msg.data); break;
        case 'scan_results': handleScanResults(msg.data); break;
        case 'backtest_results': renderBacktestResults(msg.data); break;
        case 'chart_data': renderChart(msg.data); break;
        case 'notification': showToast(msg.message, msg.level || 'info'); break;
    }
};

// ─── State Update ───────────────────────────────────────────────────
function handleStateUpdate(msg) {
    if (msg.market_phase) { appState.marketPhase = msg.market_phase; updateMarketPhase(msg.market_phase); }
    if (msg.global_context) { appState.globalContext = msg.global_context; updateMarketPulse(msg.global_context); }
    appState.openTrades = msg.open_trades || [];
    appState.closedTrades = msg.closed_trades || [];
    renderPositions(); renderTradeHistory();
    if (msg.ai_calls_today !== undefined) aiCallsBadge.textContent = `${msg.ai_calls_today}/${msg.ai_calls_limit || 7} calls`;
    if (msg.ai_advisor) handleAIAdvisorUpdate(msg.ai_advisor);
    if (msg.action_timeline) { appState.actionTimeline = msg.action_timeline; renderTimeline(); }
    if (msg.capital) document.getElementById('capital-input').value = msg.capital;
    if (msg.max_loss) document.getElementById('risk-input').value = msg.max_loss;
}

function updateMarketPhase(phase) {
    marketPhaseBadge.textContent = phase.phase_label || phase.phase;
    marketPhaseBadge.className = `phase-badge phase-${(phase.phase || '').toLowerCase()}`;
    phaseGuidance.innerHTML = `<span class="advisor-icon">💡</span><p>${phase.guidance || ''}</p>
        ${phase.mins_to_close > 0 ? `<span class="time-left">${phase.mins_to_close}m to close</span>` : ''}`;
}

function updateMarketPulse(ctx) {
    const india = ctx.india || {};
    const vix = ctx.vix || {};
    vixValEl.textContent = vix.value || '--';
    let html = '';
    for (const name of ['NIFTY 50', 'SENSEX', 'BANK NIFTY', 'NIFTY IT', 'NIFTY AUTO']) {
        const d = india[name];
        if (d) {
            const cls = d.change_pct >= 0 ? 'positive' : 'negative';
            html += `<span class="pulse-item ${cls}">${name}: ₹${d.value?.toLocaleString() || 0} (${d.change_pct >= 0 ? '+' : ''}${d.change_pct?.toFixed(2) || 0}%)</span>`;
        }
    }
    marketPulseEl.innerHTML = html || '<span class="pulse-item">Loading...</span>';
}

// ─── Scan Market ────────────────────────────────────────────────────
const scanBtn = document.getElementById('scan-now-btn');
scanBtn.addEventListener('click', () => {
    scanBtn.disabled = true; scanBtn.innerHTML = '⏳ Scanning...';
    ws.send(JSON.stringify({ action: 'trigger_scan', strategy: document.querySelector('input[name="strategy"]:checked')?.value || 's1' }));
    setTimeout(() => { scanBtn.disabled = false; scanBtn.innerHTML = '⚡ Scan Market Now'; }, 60000);
});

function handleScanResults(data) {
    scanBtn.disabled = false; scanBtn.innerHTML = '⚡ Scan Market Now';
    handleAIAdvisorUpdate({ type: 'SCAN', result: data, timestamp: new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false }) });
}

// ─── Helpers ────────────────────────────────────────────────────────
function rsiColor(rsi) { return rsi < 30 ? '#22c55e' : rsi > 70 ? '#ef4444' : rsi < 45 ? '#34d399' : rsi > 55 ? '#f87171' : '#8b949e'; }
function signalClass(s) { return s.includes('STRONG BUY') ? 'signal-strong-buy' : s.includes('BUY') ? 'signal-buy' : s.includes('STRONG SELL') ? 'signal-strong-sell' : s.includes('SELL') ? 'signal-sell' : 'signal-neutral'; }
function sentimentClass(label) { return label === 'Bullish' ? 'sentiment-bullish' : label === 'Bearish' ? 'sentiment-bearish' : 'sentiment-neutral'; }
function fmt(v) { return v !== undefined && v !== null ? Number(v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'; }

// ─── AI Advisor — Rich Stock Cards ──────────────────────────────────
function handleAIAdvisorUpdate(data) {
    if (!data) return;
    appState.aiAdvisor = data;
    aiResultSection.style.display = 'block';
    aiResultType.textContent = `🤖 ${(data.type || '').replace(/_/g, ' ').toUpperCase()}`;
    aiResultTime.textContent = data.timestamp || '';
    const result = data.result;

    if (Array.isArray(result)) {
        if (result.length === 0) {
            aiResultBody.innerHTML = '<div class="ai-empty">No actionable setups found. Being selective avoids bad trades. ✅</div>';
            return;
        }
        let html = '';
        result.forEach(r => {
            const cls = r.action === 'BUY' ? 'buy' : r.action === 'SHORT SELL' ? 'short' : 'neutral';
            const ticker = (r.ticker || '').replace('.NS', '');
            const fullTicker = (r.ticker || '').endsWith('.NS') ? r.ticker : `${r.ticker}.NS`;
            const ta = r.technicals || {};
            const rl = r.risk_levels || {};
            const fund = r.fundamentals || {};
            const sent = r.sentiment || {};
            const signal = r.signal || 'NEUTRAL';
            const conf = r.confidence || 0;

            html += `<div class="stock-card ${cls}">
                <!-- Header: Ticker + Price + Signal + Confidence -->
                <div class="sc-header">
                    <div class="sc-title">
                        <span class="sc-ticker">${ticker}</span>
                        <span class="badge-${cls}">${r.action || ''}</span>
                        <span class="sc-signal ${signalClass(signal)}">${signal}</span>
                    </div>
                    <div class="sc-price-block">
                        <span class="sc-live-price">₹${fmt(r.live_price)}</span>
                        <div class="sc-confidence" title="AI Confidence">
                            <div class="conf-bar" style="width:${conf}%; background: ${conf > 75 ? '#22c55e' : conf > 50 ? '#fbbf24' : '#ef4444'}"></div>
                            <span>${conf}%</span>
                        </div>
                    </div>
                </div>

                <!-- AI Reasoning -->
                <div class="sc-reasoning">${r.reasoning || r.reason || ''}</div>
                ${r.valid_for_minutes ? `<span class="sc-validity">⏱ Valid for ${r.valid_for_minutes} min</span>` : ''}

                <!-- TA Indicators Row -->
                <div class="sc-ta-row">
                    <div class="ta-chip" title="RSI (14)">
                        <span class="ta-label">RSI</span>
                        <span class="ta-value" style="color:${rsiColor(ta.rsi_14)}">${ta.rsi_14 || '--'}</span>
                    </div>
                    <div class="ta-chip" title="MACD Histogram">
                        <span class="ta-label">MACD</span>
                        <span class="ta-value" style="color:${(ta.macd_hist || 0) >= 0 ? '#22c55e' : '#ef4444'}">${ta.macd_hist >= 0 ? '+' : ''}${ta.macd_hist || '--'}</span>
                    </div>
                    <div class="ta-chip" title="ADX (Trend Strength)">
                        <span class="ta-label">ADX</span>
                        <span class="ta-value" style="color:${(ta.adx_14 || 0) > 25 ? '#6366f1' : '#8b949e'}">${ta.adx_14 || '--'}</span>
                    </div>
                    <div class="ta-chip" title="Volume Surge">
                        <span class="ta-label">VOL</span>
                        <span class="ta-value" style="color:${(ta.vol_surge || 1) > 1.5 ? '#22c55e' : '#8b949e'}">${ta.vol_surge || '--'}x</span>
                    </div>
                    <div class="ta-chip" title="VWAP">
                        <span class="ta-label">VWAP</span>
                        <span class="ta-value">₹${fmt(ta.vwap)}</span>
                    </div>
                    <div class="ta-chip" title="EMA 9/21 Crossover">
                        <span class="ta-label">EMA</span>
                        <span class="ta-value" style="color:${(ta.ema_9 || 0) > (ta.ema_21 || 0) ? '#22c55e' : '#ef4444'}">
                            ${(ta.ema_9 || 0) > (ta.ema_21 || 0) ? '9>21 ▲' : '9<21 ▼'}
                        </span>
                    </div>
                </div>

                <!-- Risk Levels Row (computed by Risk Engine, not AI) -->
                ${Object.keys(rl).length > 0 ? `
                <div class="sc-levels-row">
                    <div class="level-chip entry"><span>Entry</span><span>₹${fmt(r.live_price)}</span></div>
                    <div class="level-chip sl"><span>SL</span><span>₹${fmt(rl.stop_loss)}</span></div>
                    <div class="level-chip t1"><span>T1</span><span>₹${fmt(rl.target_1)}</span></div>
                    <div class="level-chip t2"><span>T2</span><span>₹${fmt(rl.target_2)}</span></div>
                    <div class="level-chip qty"><span>Qty</span><span>${rl.quantity || '--'}</span></div>
                    <div class="level-chip risk"><span>Risk</span><span>₹${rl.max_loss || '--'}</span></div>
                </div>` : ''}

                <!-- Fundamentals + Sentiment Row -->
                <div class="sc-meta-row">
                    ${fund.sector ? `<span class="meta-tag">📊 ${fund.sector}</span>` : ''}
                    ${fund.pe_ratio && fund.pe_ratio !== 'N/A' ? `<span class="meta-tag">PE: ${typeof fund.pe_ratio === 'number' ? fund.pe_ratio.toFixed(1) : fund.pe_ratio}</span>` : ''}
                    ${fund['52_week_high'] && fund['52_week_high'] !== 'N/A' ? `<span class="meta-tag">52H: ₹${fmt(fund['52_week_high'])}</span>` : ''}
                    ${fund['52_week_low'] && fund['52_week_low'] !== 'N/A' ? `<span class="meta-tag">52L: ₹${fmt(fund['52_week_low'])}</span>` : ''}
                    <span class="meta-tag ${sentimentClass(sent.label)}">${sent.label === 'Bullish' ? '📈' : sent.label === 'Bearish' ? '📉' : '➖'} ${sent.label || 'Neutral'} (${sent.headline_count || 0} news)</span>
                </div>

                <!-- Action Buttons -->
                <div class="sc-actions">
                    <button class="btn pick-log-btn" onclick="quickLogFromScan('${fullTicker}', '${r.action || 'BUY'}')">📝 Log Trade</button>
                    <button class="btn pick-chart-btn" onclick="openChart('${fullTicker}')">📈 Chart + Projection</button>
                </div>
            </div>`;
        });
        aiResultBody.innerHTML = html;
    } else if (result && typeof result === 'object' && !result.error) {
        // Exit guidance / position review
        let html = `<div class="ai-exit-guidance">
            <p><strong>${result.should_close_all ? '🔴 Close All' : '🟢 Selective Hold OK'}</strong></p>
            <p>${result.reasoning || ''}</p></div>`;
        if (result.per_position) {
            result.per_position.forEach(p => {
                html += `<div class="ai-exit-item"><span>${(p.ticker || '').replace('.NS', '')}</span>
                    <span class="badge-${p.action === 'CLOSE' ? 'short' : 'buy'}">${p.action}</span>
                    <span>${p.reason || ''}</span></div>`;
            });
        }
        aiResultBody.innerHTML = html;
    } else if (result && result.error) {
        aiResultBody.innerHTML = `<div class="ai-error">⚠️ ${result.error}</div>`;
    }
}

// ─── One-Click Trade ────────────────────────────────────────────────
function quickLogFromScan(ticker, action) {
    showToast(`Logging ${action} on ${ticker.replace('.NS', '')}...`, 'info');
    ws.send(JSON.stringify({ action: 'log_trade', ticker, trade_action: action, entry_price: 0 }));
}

// ─── Positions (with projections, risk advice) ──────────────────────
function renderPositions() {
    let totalPnl = 0;
    if (appState.openTrades.length === 0) {
        positionsContainer.innerHTML = '<div class="empty-state">No open positions.</div>';
        totalPnlEl.textContent = '₹0.00'; totalPnlEl.className = 'pnl-badge'; return;
    }
    let html = '';
    appState.openTrades.forEach(t => {
        const pnl = t.pnl || 0; totalPnl += pnl;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const cssClass = t.action === 'SHORT SELL' ? 'short' : 'buy';
        const advice = t.risk_advice || {};
        const proj = t.projections || {};

        html += `<div class="position-card ${cssClass}">
            <div class="pos-header">
                <div class="pos-title">
                    <span class="pos-ticker">${t.ticker.replace('.NS', '')}</span>
                    <span class="badge-${cssClass}">${t.action}</span>
                    ${advice.advice ? `<span class="advice-badge ${advice.advice.includes('EXIT') ? 'advice-exit' : advice.advice.includes('TRAIL') ? 'advice-trail' : 'advice-hold'}">${advice.advice}</span>` : ''}
                </div>
                <span class="pos-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</span>
            </div>
            <div class="pos-grid">
                <div class="pos-stat"><label>Entry</label><span>₹${t.entry_price.toFixed(2)}</span></div>
                <div class="pos-stat"><label>Current</label><span class="${pnlClass}">₹${(t.current_price || t.entry_price).toFixed(2)}</span></div>
                <div class="pos-stat"><label>SL</label><span class="sl-val">₹${t.stop_loss.toFixed(2)}</span></div>
                <div class="pos-stat"><label>Trail SL</label><span class="trail-val">₹${(t.trailing_sl || t.stop_loss).toFixed(2)}</span></div>
                <div class="pos-stat"><label>T1</label><span>₹${t.target_1.toFixed(2)}</span></div>
                <div class="pos-stat"><label>T2</label><span>₹${t.target_2.toFixed(2)}</span></div>
                <div class="pos-stat"><label>Qty</label><span>${t.quantity}</span></div>
                <div class="pos-stat"><label>ATR</label><span>₹${(t.atr || 0).toFixed(2)}</span></div>
            </div>
            ${Object.keys(proj).length > 0 ? `
            <div class="pos-projections">
                <span class="proj-label">3PM Projections:</span>
                <span class="proj-chip" title="Ensemble Target">🎯 ₹${proj.ensemble_target || '--'}</span>
                <span class="proj-chip" title="VWAP Anchor">VWAP ₹${proj.vwap_anchor || '--'}</span>
                <span class="proj-chip" title="Pivot">Pivot ₹${proj.pivot_point || '--'}</span>
                <span class="proj-chip" title="Momentum">Mom ₹${proj.momentum_vector || '--'}</span>
            </div>` : ''}
            ${advice.reason ? `<div class="pos-advice-reason">${advice.reason}</div>` : ''}
            <div class="pos-actions">
                <button class="btn chart-btn" onclick="openChart('${t.ticker}')">📈 Chart</button>
                <button class="btn close-btn" onclick="closeTrade('${t.id}', ${t.current_price || t.entry_price})">Close Position</button>
            </div>
        </div>`;
    });
    positionsContainer.innerHTML = html;
    appState.closedTrades.forEach(t => totalPnl += (t.pnl || 0));
    totalPnlEl.textContent = `${totalPnl >= 0 ? '+' : ''}₹${totalPnl.toFixed(2)}`;
    totalPnlEl.className = `pnl-badge ${totalPnl >= 0 ? 'positive' : 'negative'}`;
}

// ─── Timeline ───────────────────────────────────────────────────────
function renderTimeline() {
    if (appState.actionTimeline.length === 0) {
        timelineContainer.innerHTML = '<div class="empty-state">Events appear here as the day progresses.</div>';
        return;
    }
    const iconMap = { 'TRADE_OPEN': '🟢', 'TRADE_CLOSE': '🔴', 'AI_SCAN': '🤖', 'AI_POSITION_REVIEW': '📋', 'AI_EXIT_GUIDANCE': '⚡' };
    let html = '';
    [...appState.actionTimeline].reverse().forEach(e => {
        html += `<div class="timeline-event"><span class="tl-time">${e.time}</span>
            <span class="tl-icon">${iconMap[e.type] || '📌'}</span><span class="tl-message">${e.message}</span></div>`;
    });
    timelineContainer.innerHTML = html;
}

// ─── Trade History ──────────────────────────────────────────────────
function renderTradeHistory() {
    let html = '';
    appState.openTrades.forEach(t => {
        const pnl = t.pnl || 0;
        html += `<tr class="row-open"><td>${t.ticker.replace('.NS', '')}</td><td>${t.action}</td>
            <td>₹${t.entry_price.toFixed(2)}</td><td><span class="badge-open">OPEN</span></td>
            <td>${t.quantity}</td><td class="${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</td></tr>`;
    });
    appState.closedTrades.forEach(t => {
        const pnl = t.pnl || 0;
        html += `<tr><td>${t.ticker.replace('.NS', '')}</td><td>${t.action}</td>
            <td>₹${t.entry_price.toFixed(2)}</td><td>₹${(t.exit_price || 0).toFixed(2)}</td>
            <td>${t.quantity}</td><td class="${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</td></tr>`;
    });
    tradesBody.innerHTML = html || '<tr><td colspan="6" class="empty-state">No trades yet.</td></tr>';
}

// ─── Close Trade ────────────────────────────────────────────────────
function closeTrade(tradeId, exitPrice) {
    if (!confirm(`Close trade at ₹${exitPrice.toFixed(2)}?`)) return;
    ws.send(JSON.stringify({ action: 'close_trade', trade_id: tradeId, exit_price: exitPrice }));
}

// ─── Log Trade Form ─────────────────────────────────────────────────
document.getElementById('qt-submit').addEventListener('click', () => {
    const ticker = document.getElementById('qt-ticker').value.trim();
    const action = document.getElementById('qt-action').value;
    const entry = document.getElementById('qt-entry').value;
    if (!ticker || !entry) { showToast('Fill in ticker and entry price', 'error'); return; }
    document.getElementById('qt-feedback').textContent = 'Validating through Risk Engine...';
    ws.send(JSON.stringify({
        action: 'log_trade', ticker: ticker.endsWith('.NS') ? ticker : `${ticker}.NS`,
        trade_action: action, entry_price: parseFloat(entry),
    }));
    document.getElementById('qt-ticker').value = ''; document.getElementById('qt-entry').value = '';
    setTimeout(() => { document.getElementById('qt-feedback').textContent = ''; }, 5000);
});

// ─── Settings ───────────────────────────────────────────────────────
document.getElementById('save-settings-btn').addEventListener('click', () => {
    ws.send(JSON.stringify({
        action: 'update_settings',
        capital: document.getElementById('capital-input').value,
        max_loss: document.getElementById('risk-input').value,
        search_engine: document.getElementById('search-engine-input')?.value || 'ddgs',
        data_provider: document.getElementById('data-provider-input')?.value || 'yfinance',
        ai_provider: document.getElementById('ai-provider-input')?.value || 'google',
        ai_model: document.getElementById('ai-model-input')?.value || 'gemini-2.5-flash',
        auto_refresh: document.getElementById('auto-refresh-input')?.checked ?? true,
    }));
    showToast('Settings saved', 'success');
});

// ─── Chart Viewer ───────────────────────────────────────────────────
let chartRefreshTimer = null;

function openChart(ticker) {
    const modal = document.getElementById('chart-modal');
    modal.style.display = 'flex';
    document.getElementById('chart-title').textContent = `📈 ${ticker.replace('.NS', '')} — Intraday + Projection`;
    document.getElementById('chart-legend').innerHTML = '<div class="empty-state">Loading chart data...</div>';
    document.getElementById('chart-models-used').textContent = '';
    document.getElementById('chart-container').innerHTML = '';
    ws.send(JSON.stringify({ action: 'get_chart_data', ticker }));
    if (chartRefreshTimer) clearInterval(chartRefreshTimer);
    chartRefreshTimer = setInterval(() => { ws.send(JSON.stringify({ action: 'get_chart_data', ticker })); }, 60000);
}

function closeChart() {
    document.getElementById('chart-modal').style.display = 'none';
    if (chartRefreshTimer) { clearInterval(chartRefreshTimer); chartRefreshTimer = null; }
    document.getElementById('chart-container').innerHTML = '';
    appState.chartInstance = null;
}

function renderChart(data) {
    if (data.error) { document.getElementById('chart-legend').innerHTML = `<div class="ai-error">⚠️ ${data.error}</div>`; return; }
    const container = document.getElementById('chart-container');
    container.innerHTML = '';
    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth, height: 500,
        layout: { background: { type: 'solid', color: '#0f1419' }, textColor: '#8b949e', fontSize: 12 },
        grid: { vertLines: { color: 'rgba(139,148,158,0.06)' }, horzLines: { color: 'rgba(139,148,158,0.06)' } },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: 'rgba(139,148,158,0.2)' },
        timeScale: { borderColor: 'rgba(139,148,158,0.2)', timeVisible: true, secondsVisible: false },
    });
    appState.chartInstance = chart;
    new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth })).observe(container);

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
        upColor: '#22c55e', downColor: '#ef4444', borderDownColor: '#ef4444',
        borderUpColor: '#22c55e', wickDownColor: '#ef4444', wickUpColor: '#22c55e',
    });
    const ohlcData = (data.ohlc || []).map(c => ({
        time: Math.floor(new Date(c.time).getTime() / 1000), open: c.open, high: c.high, low: c.low, close: c.close,
    }));
    candleSeries.setData(ohlcData);

    // VWAP line
    if (data.vwap && ohlcData.length > 0) {
        const vwapSeries = chart.addLineSeries({ color: 'rgba(255,193,7,0.5)', lineWidth: 1, lineStyle: 2, title: 'VWAP' });
        vwapSeries.setData([{ time: ohlcData[0].time, value: data.vwap }, { time: ohlcData[ohlcData.length - 1].time, value: data.vwap }]);
    }

    // Projection line + confidence bands
    if (data.projection && data.timestamps && data.projection.length > 0) {
        const projSeries = chart.addLineSeries({ color: '#6366f1', lineWidth: 2, lineStyle: 2, title: 'Projection' });
        const projData = data.timestamps.map((t, i) => ({ time: Math.floor(new Date(t).getTime() / 1000), value: data.projection[i] }));
        if (ohlcData.length > 0) projData.unshift({ time: ohlcData[ohlcData.length - 1].time, value: ohlcData[ohlcData.length - 1].close });
        projSeries.setData(projData);

        if (data.upper_band) {
            const upperSeries = chart.addLineSeries({ color: 'rgba(99,102,241,0.2)', lineWidth: 1, lineStyle: 3 });
            const upperData = data.timestamps.map((t, i) => ({ time: Math.floor(new Date(t).getTime() / 1000), value: data.upper_band[i] }));
            if (ohlcData.length > 0) upperData.unshift({ time: ohlcData[ohlcData.length - 1].time, value: ohlcData[ohlcData.length - 1].close });
            upperSeries.setData(upperData);
        }
        if (data.lower_band) {
            const lowerSeries = chart.addLineSeries({ color: 'rgba(99,102,241,0.2)', lineWidth: 1, lineStyle: 3 });
            const lowerData = data.timestamps.map((t, i) => ({ time: Math.floor(new Date(t).getTime() / 1000), value: data.lower_band[i] }));
            if (ohlcData.length > 0) lowerData.unshift({ time: ohlcData[ohlcData.length - 1].time, value: ohlcData[ohlcData.length - 1].close });
            lowerSeries.setData(lowerData);
        }
    }
    chart.timeScale().fitContent();

    document.getElementById('chart-models-used').textContent = `Models: ${(data.models_used || []).join(' + ')}`;
    document.getElementById('chart-legend').innerHTML = `<div class="legend-items">
        <span class="legend-candle">🟩🟥 Candles</span>
        <span style="color:#6366f1;">━━ Projection</span>
        <span style="color:rgba(99,102,241,0.4);">┄┄ ±1σ Band</span>
        <span style="color:#ffc107;">── VWAP ₹${data.vwap || '--'}</span>
        <span class="legend-price">Current: ₹${data.current_price || '--'}</span>
    </div>`;
}

// ─── Backtester ─────────────────────────────────────────────────────
document.getElementById('run-backtest-btn').addEventListener('click', () => {
    const btn = document.getElementById('run-backtest-btn');
    btn.disabled = true; btn.textContent = '⏳ Running...';
    ws.send(JSON.stringify({
        action: 'run_backtest', ticker: document.getElementById('bt-ticker').value,
        days: parseInt(document.getElementById('bt-days').value),
        capital: parseFloat(document.getElementById('bt-capital').value),
        params: {
            ema_fast: parseInt(document.getElementById('bt-ema-fast').value),
            ema_slow: parseInt(document.getElementById('bt-ema-slow').value),
            rsi_len: parseInt(document.getElementById('bt-rsi-len').value),
            rsi_buy_threshold: parseInt(document.getElementById('bt-rsi-buy').value),
            rsi_short_threshold: parseInt(document.getElementById('bt-rsi-short').value),
            sl_pct: parseFloat(document.getElementById('bt-sl').value) / 100,
            tp_pct: parseFloat(document.getElementById('bt-tp').value) / 100,
        }
    }));
    setTimeout(() => { btn.disabled = false; btn.textContent = '🚀 Run Backtest'; }, 15000);
});

document.getElementById('ai-optimize-btn').addEventListener('click', () => {
    const btn = document.getElementById('ai-optimize-btn');
    btn.disabled = true; btn.textContent = '✨ Optimizing...';
    ws.send(JSON.stringify({
        action: 'ai_optimize', ticker: document.getElementById('bt-ticker').value,
        days: parseInt(document.getElementById('bt-days').value),
        capital: parseFloat(document.getElementById('bt-capital').value),
        params: {
            ema_fast: parseInt(document.getElementById('bt-ema-fast').value),
            ema_slow: parseInt(document.getElementById('bt-ema-slow').value),
            rsi_len: parseInt(document.getElementById('bt-rsi-len').value),
            rsi_buy_threshold: parseInt(document.getElementById('bt-rsi-buy').value),
            rsi_short_threshold: parseInt(document.getElementById('bt-rsi-short').value),
            sl_pct: parseFloat(document.getElementById('bt-sl').value) / 100,
            tp_pct: parseFloat(document.getElementById('bt-tp').value) / 100,
        }
    }));
    setTimeout(() => { btn.disabled = false; btn.textContent = '✨ AI Optimize Parameters'; }, 60000);
});

function renderBacktestResults(data) {
    document.getElementById('run-backtest-btn').disabled = false;
    document.getElementById('run-backtest-btn').textContent = '🚀 Run Backtest';
    document.getElementById('ai-optimize-btn').disabled = false;
    document.getElementById('ai-optimize-btn').textContent = '✨ AI Optimize Parameters';
    const c = document.getElementById('bt-results-container');
    if (data.error) { c.innerHTML = `<div class="ai-error">⚠️ ${data.error}</div>`; return; }
    const cls = data.net_profit >= 0 ? 'positive' : 'negative';
    let html = `<div class="bt-metrics">
        <div class="bt-metric"><label>Trades</label><span>${data.total_trades}</span></div>
        <div class="bt-metric"><label>Win Rate</label><span>${data.win_rate}%</span></div>
        <div class="bt-metric ${cls}"><label>Net Profit</label><span>₹${data.net_profit?.toLocaleString()}</span></div>
        <div class="bt-metric"><label>Drawdown</label><span>${data.max_drawdown_pct}%</span></div>
        <div class="bt-metric"><label>Final Equity</label><span>₹${data.final_equity?.toLocaleString()}</span></div>
    </div>`;
    if (data.trade_log?.length > 0) {
        html += `<h3 style="margin:16px 0 8px;font-size:14px;">Trade Log</h3>
            <table class="trades-table"><thead><tr><th>Type</th><th>Entry</th><th>Exit</th><th>P&L %</th></tr></thead><tbody>`;
        data.trade_log.slice(0, 20).forEach(t => {
            html += `<tr><td>${t.type}</td><td>₹${t.entry_price?.toFixed(2)}</td>
                <td>₹${t.exit_price?.toFixed(2)}</td>
                <td class="${t.pnl_pct > 0 ? 'positive' : 'negative'}">${(t.pnl_pct * 100).toFixed(2)}%</td></tr>`;
        });
        html += '</tbody></table>';
    }
    if (data.optimization_history) {
        html += `<h3 style="margin:16px 0 8px;font-size:14px;">🤖 AI Optimization</h3>`;
        data.optimization_history.forEach((iter, i) => {
            html += `<div class="bt-iter"><strong>Iter ${i + 1}:</strong> Win ${iter.win_rate}% | ₹${iter.net_profit?.toLocaleString()} | DD ${iter.max_drawdown_pct}%</div>`;
        });
    }
    c.innerHTML = html;
}

// ─── Toast ──────────────────────────────────────────────────────────
function showToast(message, level = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${level}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 300); }, 4000);
}
