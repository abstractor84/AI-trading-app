/**
 * History Page Logic
 * Fetches and displays historical data from REST API endpoints.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ─── DOM References ─────────
    const historyBtn = document.getElementById('nav-history-btn');
    const pageHistory = document.getElementById('page-history');

    // Wire up navigation 
    // Need to also un-highlight other buttons, so easiest to hook into existing click navs or just do it here
    const dashboardBtn = document.getElementById('nav-dashboard-btn');
    const backtestBtn = document.getElementById('nav-backtest-btn');
    const pageDashboard = document.getElementById('page-dashboard');
    const pageBacktest = document.getElementById('page-backtest');

    historyBtn.addEventListener('click', () => {
        historyBtn.classList.add('active');
        dashboardBtn.classList.remove('active');
        backtestBtn.classList.remove('active');

        pageHistory.style.display = '';
        pageDashboard.style.display = 'none';
        pageBacktest.style.display = 'none';
        pageDashboard.classList.remove('active');

        // Auto load today on first click
        if (!document.getElementById('history-date-input').value) {
            const today = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD
            document.getElementById('history-date-input').value = today;
            loadHistoryData(today);
        }
    });

    // Override other nav buttons to hide history page
    dashboardBtn.addEventListener('click', () => { pageHistory.style.display = 'none'; });
    backtestBtn.addEventListener('click', () => { pageHistory.style.display = 'none'; });

    // ─── Global Search ──────────
    document.getElementById('global-chart-btn').addEventListener('click', () => {
        const input = document.getElementById('global-chart-input').value.trim().toUpperCase();
        if (input) {
            const ticker = input.endsWith('.NS') ? input : `${input}.NS`;
            if (window.openChart) {
                window.openChart(ticker);
            }
        }
    });

    document.getElementById('global-chart-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') document.getElementById('global-chart-btn').click();
    });

    // ─── History Fetch Logic ────
    document.getElementById('history-fetch-btn').addEventListener('click', () => {
        const dateStr = document.getElementById('history-date-input').value;
        if (dateStr) {
            loadHistoryData(dateStr);
        }
    });
});

async function loadHistoryData(dateStr) {
    try {
        const [tradesRes, scansRes, summaryRes] = await Promise.all([
            fetch(`/api/history/trades?date=${dateStr}`),
            fetch(`/api/history/scans?date=${dateStr}`),
            fetch('/api/history/summary') // Fetching all summaries to find the matching one
        ]);

        const trades = await tradesRes.json();
        const scans = await scansRes.json();
        const summaries = await summaryRes.json();

        const summary = summaries.find(s => s.date === dateStr) || null;

        renderHistoryTrades(trades);
        renderHistoryScans(scans);
        renderHistorySummary(summary);

        if (window.showToast) window.showToast(`Loaded history for ${dateStr}`, 'success');

    } catch (e) {
        console.error("History fetch error", e);
        if (window.showToast) window.showToast('Failed to load history', 'error');
    }
}

function renderHistoryTrades(trades) {
    const tbody = document.getElementById('history-trades-body');
    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No trades recorded.</td></tr>';
        return;
    }

    let html = '';
    trades.forEach(t => {
        const pnl = t.pnl || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        html += `<tr>
            <td style="cursor:pointer; color:#6366f1; text-decoration:underline;" onclick="if(window.openChart) window.openChart('${t.ticker}')">${t.ticker.replace('.NS', '')}</td>
            <td>${t.action}</td>
            <td>₹${t.entry_price?.toFixed(2) || '0.00'}</td>
            <td>${t.status === 'CLOSED' ? `₹${t.exit_price?.toFixed(2)}` : '<span class="badge-open">OPEN</span>'}</td>
            <td>${t.quantity}</td>
            <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</td>
        </tr>`;
    });
    tbody.innerHTML = html;
}

function renderHistorySummary(summary) {
    const container = document.getElementById('history-summary-container');
    if (!summary) {
        container.innerHTML = '<div class="empty-state">No summary generated for this date. (Only generated at midnight)</div>';
        return;
    }

    const pnlClass = summary.total_pnl >= 0 ? 'positive' : 'negative';
    const winRate = summary.total_trades > 0 ? ((summary.wins / summary.total_trades) * 100).toFixed(1) : 0;

    container.innerHTML = `
        <div class="bt-metrics">
            <div class="bt-metric"><label>Total Trades</label><span>${summary.total_trades}</span></div>
            <div class="bt-metric"><label>W/L</label><span>${summary.wins} / ${summary.losses}</span></div>
            <div class="bt-metric"><label>Win Rate</label><span>${winRate}%</span></div>
            <div class="bt-metric ${pnlClass}"><label>Net P&L</label><span>₹${summary.total_pnl.toFixed(2)}</span></div>
        </div>
    `;
}

function renderHistoryScans(scans) {
    const container = document.getElementById('history-scans-container');
    if (!scans || scans.length === 0) {
        container.innerHTML = '<div class="empty-state">No AI scans saved for this date.</div>';
        return;
    }

    let html = '';
    scans.forEach(scan => {
        const time = new Date(scan.timestamp).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' });
        let resultArr = [];
        try {
            resultArr = JSON.parse(scan.output_json) || [];
        } catch (e) { }

        if (!Array.isArray(resultArr)) resultArr = [];

        html += `<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px; color:#8b949e;">
                <span>Scan Time: ${time}</span>
                <span>Model: ${scan.model_used}</span>
            </div>`;

        if (resultArr.length === 0) {
            html += `<span style="font-size:13px;">No actionable setups found.</span></div>`;
            return;
        }

        resultArr.forEach(r => {
            const cls = r.action === 'BUY' ? 'buy' : r.action === 'SHORT SELL' ? 'short' : 'neutral';
            html += `<div style="display:flex; justify-content:space-between; align-items:center; background:#0f1419; padding:8px 12px; margin-bottom:4px; border-radius:6px; font-size:13px;">
                <div>
                    <strong style="color:#e6edf3; cursor:pointer;" onclick="if(window.openChart) window.openChart('${(r.ticker || '').endsWith('.NS') ? r.ticker : r.ticker + '.NS'}')">${(r.ticker || '').replace('.NS', '')}</strong>
                    <span class="badge-${cls}" style="margin-left:8px;">${r.action}</span>
                </div>
                <div>
                    <span style="color:#8b949e;">Conf: ${r.confidence || 0}%</span>
                </div>
            </div>`;
        });
        html += `</div>`;
    });

    container.innerHTML = html;
}
