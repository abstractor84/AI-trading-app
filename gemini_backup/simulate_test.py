"""
Comprehensive Simulation Test Suite
====================================
Tests ALL backend components with simulated market data across different IST times.
Run: cd gemini_nse_trader && venv/bin/python simulate_test.py
"""
import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SimTest")
logger.setLevel(logging.INFO)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = {"pass": 0, "fail": 0, "warn": 0}


def test(name, condition, detail=""):
    global results
    if condition:
        results["pass"] += 1
        logger.info(f"  {PASS} {name}")
    else:
        results["fail"] += 1
        logger.error(f"  {FAIL} {name} — {detail}")


def test_warn(name, detail=""):
    global results
    results["warn"] += 1
    logger.warning(f"  {WARN} {name} — {detail}")


# ─── Generate Simulated OHLCV Data ──────────────────────────────────
def make_ohlcv(base_price=2500, bars=100, interval_min=5, trend="up"):
    """Generate realistic OHLCV DataFrame for testing."""
    dates = pd.date_range(
        start=datetime.now().replace(hour=9, minute=15) - timedelta(minutes=interval_min * bars),
        periods=bars, freq=f"{interval_min}min"
    )
    np.random.seed(42)
    prices = [base_price]
    for i in range(1, bars):
        drift = 0.0002 if trend == "up" else -0.0002 if trend == "down" else 0
        ret = drift + np.random.randn() * 0.003
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    highs = closes * (1 + np.random.uniform(0.001, 0.005, bars))
    lows = closes * (1 - np.random.uniform(0.001, 0.005, bars))
    opens = closes * (1 + np.random.uniform(-0.002, 0.002, bars))
    volumes = np.random.randint(10000, 500000, bars)

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes
    }, index=dates)
    return df


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Market Phase Service — Simulate All IST Times
# ═══════════════════════════════════════════════════════════════════
def test_market_phase():
    logger.info("\n═══ TEST 1: Market Phase Service ═══")
    from services.market_phase import MarketPhaseService, MarketPhase

    svc = MarketPhaseService()

    # Simulate different times of day
    phase_tests = [
        (datetime(2026, 2, 26, 8, 0), MarketPhase.PRE_MARKET, "8:00 AM → Pre-Market"),
        (datetime(2026, 2, 26, 9, 15), MarketPhase.OPENING_15, "9:15 AM → Opening 15 min"),
        (datetime(2026, 2, 26, 9, 29), MarketPhase.OPENING_15, "9:29 AM → Still Opening"),
        (datetime(2026, 2, 26, 10, 0), MarketPhase.MID_SESSION, "10:00 AM → Mid Session"),
        (datetime(2026, 2, 26, 12, 0), MarketPhase.MID_SESSION, "12:00 PM → Mid Session"),
        (datetime(2026, 2, 26, 14, 30), MarketPhase.POWER_HOUR, "2:30 PM → Power Hour"),
        (datetime(2026, 2, 26, 14, 59), MarketPhase.POWER_HOUR, "2:59 PM → Power Hour"),
        (datetime(2026, 2, 26, 15, 0), MarketPhase.POST_MARKET, "3:00 PM → Post Market"),
        (datetime(2026, 2, 26, 15, 30), MarketPhase.POST_MARKET, "3:30 PM → Post Market"),
        (datetime(2026, 2, 26, 18, 0), MarketPhase.CLOSED, "6:00 PM → Market Closed"),
    ]

    for sim_time, expected_phase, desc in phase_tests:
        with patch("services.market_phase.datetime") as mock_dt:
            mock_dt.now.return_value = sim_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            phase = svc.get_current_phase()
            test(desc, phase == expected_phase, f"Got {phase} expected {expected_phase}")

    # Test phase context structure
    ctx = svc.get_phase_context()
    required_keys = ["phase", "phase_label", "mins_to_close", "is_trading_hours",
                     "allow_new_entries", "should_review_positions", "transitioned", "guidance"]
    for k in required_keys:
        test(f"Phase context has '{k}'", k in ctx, f"Missing key: {k}")

    # Test AI schedule
    schedule = svc.get_ai_schedule()
    test("AI schedule has call_interval_mins", "call_interval_mins" in schedule)
    test("AI schedule has prompt_type", "prompt_type" in schedule)


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Risk Engine — All Validation Gates
# ═══════════════════════════════════════════════════════════════════
def test_risk_engine():
    logger.info("\n═══ TEST 2: Risk Engine ═══")
    from services.risk_engine import RiskEngine

    engine = RiskEngine(capital=100000, max_risk_per_trade=500, max_daily_loss=5000)

    # Test ATR computation
    df = make_ohlcv(2500, 100)
    atr = engine.compute_atr(df)
    test("ATR is positive", atr > 0, f"ATR={atr}")
    test("ATR is reasonable (< 5% of price)", atr < 2500 * 0.05, f"ATR={atr}")

    # Test SL/Target computation
    levels = engine.compute_sl_target(2500, "BUY", atr)
    test("SL below entry for BUY", levels["stop_loss"] < 2500, f"SL={levels['stop_loss']}")
    test("T1 above entry for BUY", levels["target_1"] > 2500, f"T1={levels['target_1']}")
    test("T2 above T1 for BUY", levels["target_2"] > levels["target_1"], f"T2={levels['target_2']}")
    test("RR ratio >= 1", levels["rr_ratio"] >= 1.0, f"RR={levels['rr_ratio']}")

    levels_short = engine.compute_sl_target(2500, "SHORT SELL", atr)
    test("SL above entry for SHORT", levels_short["stop_loss"] > 2500)
    test("T1 below entry for SHORT", levels_short["target_1"] < 2500)

    # Test position sizing
    qty = engine.compute_position_size(2500, levels["stop_loss"])
    test("Quantity > 0", qty > 0, f"qty={qty}")
    max_loss = qty * levels["risk_per_share"]
    test(f"Max loss ≤ ₹500 (got ₹{max_loss:.0f})", max_loss <= 510)  # small tolerance

    # Test full validation pipeline
    validation = engine.validate_trade(2500, "BUY", atr, 2500)
    test("Validation passed", validation["passed"], f"Reasons: {validation['reasons']}")
    test("Validation has levels", validation["levels"] is not None)
    test("Validation has quantity", validation["quantity"] > 0)

    # Test ATR=0 blocks trade
    val_bad = engine.validate_trade(2500, "BUY", 0, 2500)
    test("ATR=0 blocks trade", not val_bad["passed"])

    # Test trailing SL
    sl_hold = engine.compute_trailing_sl(2500, 2510, "BUY", atr)
    sl_be = engine.compute_trailing_sl(2500, 2500 + atr, "BUY", atr)
    sl_trail = engine.compute_trailing_sl(2500, 2500 + 2 * atr, "BUY", atr)
    test("Trail SL: small profit → original SL", sl_hold < 2500)
    test("Trail SL: 1×ATR → breakeven", abs(sl_be - 2500.0) < 1.0, f"SL={sl_be}")
    test("Trail SL: 2×ATR → entry+ATR", sl_trail > 2500)

    # Test position action
    trade = {"entry_price": 2500, "action": "BUY", "stop_loss": 2470,
             "target_1": 2530, "target_2": 2560, "quantity": 20}
    advice_hold = engine.get_position_action(trade, 2510, atr, 180)
    test("Position advice: HOLD in profit", advice_hold["advice"] == "HOLD")

    advice_sl = engine.get_position_action(trade, 2465, atr, 180)
    test("Position advice: EXIT on SL hit", "EXIT" in advice_sl["advice"])

    advice_t2 = engine.get_position_action(trade, 2565, atr, 180)
    test("Position advice: EXIT on T2 hit", "EXIT" in advice_t2["advice"])

    advice_eod = engine.get_position_action(trade, 2490, atr, 10)
    test("Position advice: EXIT near day close", "EXIT" in advice_eod["advice"])


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Technical Analysis Service
# ═══════════════════════════════════════════════════════════════════
def test_technical_analysis():
    logger.info("\n═══ TEST 3: Technical Analysis Service ═══")
    from services.technical_analysis import TechnicalAnalysisService

    ta = TechnicalAnalysisService()

    # Test indicator computation on simulated data
    df = make_ohlcv(2500, 100)
    indicators = ta.compute_indicators(df)
    test("Indicators not None", indicators is not None)

    required = ["close", "rsi_14", "ema_9", "ema_21", "macd_hist", "vwap", "adx_14",
                 "bb_upper", "bb_lower", "vol_surge"]
    for k in required:
        test(f"Indicator '{k}' present", k in indicators, f"Missing: {k}")

    # Sanity checks
    test("RSI in [0, 100]", 0 <= indicators.get("rsi_14", -1) <= 100, f"RSI={indicators.get('rsi_14')}")
    test("EMA 9 > 0", indicators.get("ema_9", 0) > 0)
    test("BB Upper > BB Lower", indicators.get("bb_upper", 0) > indicators.get("bb_lower", 0))

    # Test live data fetch (yfinance — may fail if no internet)
    try:
        live_df = ta.fetch_ohlcv("RELIANCE.NS", period="5d", interval="5m")
        test("yfinance fetch returns data", live_df is not None and not live_df.empty,
             f"Got {'None' if live_df is None else f'{len(live_df)} rows'}")
    except Exception as e:
        test_warn("yfinance fetch failed (network?)", str(e)[:80])

    # Test fundamentals
    try:
        fund = ta.fetch_fundamentals("RELIANCE.NS")
        test("Fundamentals has 'sector'", "sector" in fund, f"Got: {list(fund.keys())}")
        test("Fundamentals has 'pe_ratio'", "pe_ratio" in fund)
    except Exception as e:
        test_warn("Fundamentals fetch failed", str(e)[:80])


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Signal Classification
# ═══════════════════════════════════════════════════════════════════
def test_signal_classification():
    logger.info("\n═══ TEST 4: Signal Classification ═══")
    from ws_handler import _classify_signal

    # Strong bullish setup
    bullish = {"rsi_14": 40, "macd_hist": 2.0, "adx_14": 30, "vol_surge": 2.0,
               "ema_9": 2510, "ema_21": 2490, "close": 2520, "vwap": 2500}
    sig = _classify_signal(bullish)
    test(f"Strong bullish → {sig}", sig in ("STRONG BUY", "BUY"))

    # Strong bearish setup
    bearish = {"rsi_14": 70, "macd_hist": -2.0, "adx_14": 30, "vol_surge": 2.0,
               "ema_9": 2490, "ema_21": 2510, "close": 2480, "vwap": 2500}
    sig = _classify_signal(bearish)
    test(f"Strong bearish → {sig}", sig in ("STRONG SELL", "SELL"))

    # Neutral
    neutral = {"rsi_14": 50, "macd_hist": 0.1, "adx_14": 15, "vol_surge": 1.0,
               "ema_9": 2500, "ema_21": 2500, "close": 2500, "vwap": 2500}
    sig = _classify_signal(neutral)
    test(f"Neutral → {sig}", sig == "NEUTRAL")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Price Projector
# ═══════════════════════════════════════════════════════════════════
def test_price_projector():
    logger.info("\n═══ TEST 5: Price Projector ═══")
    from services.price_projector import PriceProjector

    proj = PriceProjector()
    df = make_ohlcv(2500, 200, interval_min=5, trend="up")

    result = proj.generate_projection(df, interval_minutes=5)
    test("Result has 'ohlc'", "ohlc" in result)
    test("Result has 'projection'", "projection" in result)
    test("Result has 'timestamps'", "timestamps" in result)
    test("Result has 'upper_band'", "upper_band" in result)
    test("Result has 'lower_band'", "lower_band" in result)
    test("Result has 'current_price'", "current_price" in result)
    test("Result has 'vwap'", "vwap" in result)
    test("Result has 'models_used'", "models_used" in result)

    # Projection sanity
    if result.get("projection"):
        projections = result["projection"]
        test("Projection has values", len(projections) > 0)
        test("Projection values near current price (within 5%)",
             all(abs(p - result["current_price"]) / result["current_price"] < 0.05 for p in projections),
             f"Current: {result['current_price']}, Proj range: {min(projections):.2f}-{max(projections):.2f}")
        test("Upper band > Lower band", all(
            u > l for u, l in zip(result["upper_band"], result["lower_band"])
        ))

    # OHLC format
    if result.get("ohlc"):
        c0 = result["ohlc"][0]
        test("OHLC has 'time'", "time" in c0)
        test("OHLC has 'open', 'high', 'low', 'close'",
             all(k in c0 for k in ("open", "high", "low", "close")))


# ═══════════════════════════════════════════════════════════════════
# TEST 6: News Sentiment Service
# ═══════════════════════════════════════════════════════════════════
def test_news_sentiment():
    logger.info("\n═══ TEST 6: News Sentiment Service ═══")
    from services.news_sentiment import _keyword_sentiment

    # Test keyword sentiment (no API needed)
    bullish_headlines = [
        "Reliance shares surge on strong Q3 earnings",
        "Reliance hits 52-week high on bullish outlook",
        "Strong GDP growth boosts market rally",
    ]
    result = _keyword_sentiment(bullish_headlines)
    test("Bullish headlines → positive score", result.get("score", 0) > 50,
         f"Score={result.get('score')}")
    test("Bullish label", result.get("label") == "Bullish", f"Label={result.get('label')}")

    bearish_headlines = [
        "Stock crashes after fraud allegations",
        "Market plunges on global recession fears",
        "Company faces massive penalty, shares drop",
    ]
    result = _keyword_sentiment(bearish_headlines)
    test("Bearish headlines → negative score", result.get("score", 100) < 50,
         f"Score={result.get('score')}")

    mixed_headlines = [
        "Market steady as investors wait for earnings",
        "Interest rates unchanged, neutral outlook",
    ]
    result = _keyword_sentiment(mixed_headlines)
    test("Mixed headlines → neutral", 30 <= result.get("score", 0) <= 70,
         f"Score={result.get('score')}")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: App State & Trade Lifecycle
# ═══════════════════════════════════════════════════════════════════
def test_app_state():
    logger.info("\n═══ TEST 7: App State & Trade Lifecycle ═══")
    from services.state import AppState

    state = AppState()

    # Settings
    state.update_settings(200000, 1000, "ddgs", "yfinance", ai_provider="google", ai_model="gemini-2.5-flash")
    test("Capital updated", state.capital == 200000)
    test("Max loss updated", state.max_loss_per_trade == 1000)
    test("AI provider set", state.ai_provider == "google")

    # Log a trade
    trade = state.log_trade(
        ticker="RELIANCE.NS", action="BUY", qty=20,
        entry_price=2500, sl=2470, t1=2530, t2=2560,
        phase="MID_SESSION", atr=20.5, risk_per_share=30.0
    )
    test("Trade logged", len(state.open_trades) > 0)
    test("Trade has ID", trade.get("id") is not None)
    test("Trade has entry_price", trade.get("entry_price") == 2500)

    trade_id = trade["id"]

    # Close the trade
    result = state.close_trade(trade_id, 2540)
    test("Trade closed", any(t["id"] == trade_id for t in state.closed_trades))
    closed = next(t for t in state.closed_trades if t["id"] == trade_id)
    test("Closed trade has exit_price", closed.get("exit_price") == 2540)
    test("PnL correct (profit)", closed.get("pnl", 0) > 0, f"PnL={closed.get('pnl')}")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Projection Mapper (3PM Projections)
# ═══════════════════════════════════════════════════════════════════
def test_projection_mapper():
    logger.info("\n═══ TEST 8: Projection Mapper ═══")
    from services.projection_mapper import ProjectionService

    svc = ProjectionService()
    df = make_ohlcv(2500, 100)

    projections = svc.calculate_projections("TEST.NS", df)
    test("Projections not None", projections is not None)
    if projections:
        test("Has ensemble_target", "ensemble_target" in projections)
        test("Ensemble target > 0", projections.get("ensemble_target", 0) > 0)


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Stock Discovery Service
# ═══════════════════════════════════════════════════════════════════
def test_stock_discovery():
    logger.info("\n═══ TEST 9: Stock Discovery Service ═══")
    from services.stock_discovery import StockDiscoveryService

    svc = StockDiscoveryService()
    test("Universe populated on init", len(svc.universe) > 0, f"Size={len(svc.universe)}")
    test("Universe has Nifty 50 stocks", "RELIANCE.NS" in svc.universe)
    test("Universe has >80 stocks", len(svc.universe) > 80, f"Size={len(svc.universe)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Backtester
# ═══════════════════════════════════════════════════════════════════
def test_backtester():
    logger.info("\n═══ TEST 10: Backtester ═══")
    from services.backtester import VectorizedBacktester

    df = make_ohlcv(2500, 500, interval_min=5, trend="up")
    bt = VectorizedBacktester(df, initial_capital=100000)

    params = {
        "ema_fast": 9, "ema_slow": 21, "rsi_len": 14,
        "rsi_buy_threshold": 45, "rsi_short_threshold": 55,
        "sl_pct": 0.015, "tp_pct": 0.03,
    }

    result = bt.run_strategy(params)
    test("Backtest result not None", result is not None)
    if result:
        test("Has total_trades", "total_trades" in result)
        test("Has win_rate", "win_rate" in result)
        test("Has net_profit", "net_profit" in result)
        test("Has final_equity", "final_equity" in result)
        test("Has max_drawdown_pct", "max_drawdown_pct" in result)
        test("Total trades > 0", result.get("total_trades", 0) > 0,
             f"Trades={result.get('total_trades')}")
        test("Final equity > 0", result.get("final_equity", 0) > 0)


# ═══════════════════════════════════════════════════════════════════
# TEST 11: Quota Service
# ═══════════════════════════════════════════════════════════════════
def test_quota_service():
    logger.info("\n═══ TEST 11: Quota Service ═══")
    from services.quota_service import QuotaService

    qs = QuotaService()
    status = qs.check_quota("gemini-2.5-flash")
    test("check_quota returns dict", isinstance(status, dict))
    test("Quota has 'can_call' key", "can_call" in status)
    test("Initial quota allows calls", status.get("can_call", False))
    test("Has remaining_rpd", status.get("remaining_rpd", 0) > 0)


# ═══════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("  COMPREHENSIVE SIMULATION TEST SUITE")
    logger.info("  Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))
    logger.info("=" * 60)

    tests = [
        test_market_phase,
        test_risk_engine,
        test_technical_analysis,
        test_signal_classification,
        test_price_projector,
        test_news_sentiment,
        test_app_state,
        test_projection_mapper,
        test_stock_discovery,
        test_backtester,
        test_quota_service,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            logger.error(f"\n{FAIL} {t.__name__} CRASHED: {e}")
            traceback.print_exc()
            results["fail"] += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"  RESULTS: {PASS} {results['pass']} passed | {FAIL} {results['fail']} failed | {WARN} {results['warn']} warnings")
    logger.info("=" * 60)

    sys.exit(1 if results["fail"] > 0 else 0)
