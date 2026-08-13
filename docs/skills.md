# SuperNova Trading Project - Agent Skills

This document defines the skills required for developers and agents to work with the SuperNova AI Trading System. The project is a sophisticated NSE (India) intraday trading platform that combines mathematical analysis, AI judgment, and deterministic risk management.

---

## Table of Contents

1. [Trading Domain Skills](#1-trading-domain-skills)
2. [Technical Skills](#2-technical-skills)
3. [Project-Specific Skills](#3-project-specific-skills)

---

## 1. Trading Domain Skills

### 1.1 NSE Trading Knowledge

**Description:** Understanding of the Indian stock market structure, trading instruments, market hours, and regulatory framework.

**Key Concepts:**
- **Market Hours:** 9:15 AM - 3:30 PM IST (regular trading)
- **Instrument Types:** EQ (equity), F&O (futures & options), Currency derivatives
- **Order Types:** Market, Limit, Stop-Loss, Bracket Orders
- **Exchange:** National Stock Exchange (NSE) - India's largest equity exchange
- **Index:** NIFTY 50, Bank Nifty, FinNifty, India VIX

**Key Files/Functions:**
- [`services/upstox_service.py`](services/upstox_service.py:88) - `UpstoxService` class for broker integration
- [`services/market_phase.py`](services/market_phase.py:36) - `MarketPhaseService` for phase detection
- [`NSE_instruments.json`](NSE_instruments.json:1) - Instrument master data (~10MB)
- [`services/stock_discovery.py`](services/stock_discovery.py:11) - `_NIFTY100_SYMBOLS` constant

**Why Important:** The entire system is purpose-built for NSE intraday trading. Understanding market hours, instrument keys, and NSE-specific data formats is essential for any customization or debugging.

---

### 1.2 Technical Analysis

**Description:** Mastery of classical and modern technical indicators used for trade signal generation.

**Key Indicators Implemented:**
| Indicator | Purpose | Key Files |
|-----------|---------|-----------|
| **EMA (9, 21)** | Trend direction | [`services/technical_analysis.py:128`](services/technical_analysis.py:128) |
| **VWAP** | Intraday value anchor | [`services/technical_analysis.py:149`](services/technical_analysis.py:149) |
| **RSI (14)** | Overbought/oversold | [`services/technical_analysis.py:132`](services/technical_analysis.py:132) |
| **MACD** | Momentum convergence | [`services/technical_analysis.py:135`](services/technical_analysis.py:135) |
| **ADX (14)** | Trend strength | [`services/technical_analysis.py:141`](services/technical_analysis.py:141) |
| **Bollinger Bands** | Volatility envelopes | [`services/technical_analysis.py:138`](services/technical_analysis.py:138) |
| **ATR** | Volatility measurement | [`services/risk_engine.py:45`](services/risk_engine.py:45) |

**Advanced ML Indicators:**
- **Lorentzian Classification** - KNN-based pattern recognition ([`services/advanced_indicators.py:14`](services/advanced_indicators.py:14))
- **Adaptive SuperTrend** - K-Means volatility regime detection ([`services/advanced_indicators.py:225`](services/advanced_indicators.py:225))
- **KNN Trend Forecaster** - Multi-feature classification ([`services/advanced_indicators.py:312`](services/advanced_indicators.py:312))

**Key Functions:**
- [`TechnicalAnalysisService.compute_indicators()`](services/technical_analysis.py:112) - Computes all standard indicators
- [`TechnicalAnalysisService.classify_signal()`](services/technical_analysis.py:263) - Multi-indicator trade classification
- [`TechnicalAnalysisService.evaluate_math_probability()`](services/technical_analysis.py:214) - Pure math trade setup scoring

**Why Important:** Technical analysis forms the foundation of the "Math Layer" in the Triple-Layer Consensus Engine. Understanding these indicators is crucial for modifying signal generation logic.

---

### 1.3 Risk Management

**Description:** Deterministic risk controls that protect capital through mathematical position sizing, stop-loss calculation, and R:R ratio enforcement.

**Key Concepts:**
- **Risk:Reward Ratio** - Minimum 1.25:1 enforced by [`risk_engine.py`](services/risk_engine.py:247)
- **Position Sizing** - Fixed-risk model: `Qty = MaxRiskPerTrade / RiskPerShare` ([`services/risk_engine.py:151`](services/risk_engine.py:151))
- **ATR-based SL/TP** - Dynamic levels based on volatility ([`services/risk_engine.py:104`](services/risk_engine.py:104))
- **Daily Drawdown Limit** - Circuit breaker at 2% of capital or fixed amount ([`services/risk_engine.py:207`](services/risk_engine.py:207))
- **Trailing Stop-Loss** - Dynamic protection as price moves in favor ([`services/risk_engine.py:165`](services/risk_engine.py:165))

**Key Functions:**
- [`RiskEngine.compute_sl_target()`](services/risk_engine.py:104) - Calculates SL, T1, T2 using ATR multipliers
- [`RiskEngine.compute_position_size()`](services/risk_engine.py:151) - Determines quantity based on risk
- [`RiskEngine.validate_trade()`](services/risk_engine.py:195) - Multi-gate validation pipeline
- [`RiskEngine.get_position_action()`](services/risk_engine.py:283) - Exit recommendations for open positions

**Why Important:** The Risk Engine is the "law" layer - AI cannot override its decisions. Any changes to trading logic must work within these risk constraints.

---

### 1.4 Market Phase Detection

**Description:** Time-of-day aware trading context that adapts strategy based on NSE market session.

**Phase Definitions** ([`services/market_phase.py:25`](services/market_phase.py:25)):

| Phase | Time (IST) | Trading Rules |
|-------|------------|---------------|
| PRE_MARKET_SETUP | 08:00 - 09:15 | Watch only, prepare |
| OPENING_VOLATILITY | 09:15 - 09:45 | Highest volatility, no entries |
| MID_MORNING_TREND | 09:45 - 11:30 | Prime trading hours |
| LUNCH_CHOP | 11:30 - 13:30 | Low volume, avoid new entries |
| PM_BREAKOUT | 13:30 - 14:30 | Afternoon momentum |
| POWER_HOUR | 14:30 - 15:30 | Exit focus, no new trades |
| POST_MARKET_SETTLEMENT | 15:30 - 16:30 | Review and analyze |

**Key Functions:**
- [`MarketPhaseService.get_current_phase()`](services/market_phase.py:42) - Returns current phase enum
- [`MarketPhaseService.get_phase_context()`](services/market_phase.py:50) - Full phase metadata for AI prompts
- [`MarketPhaseService.get_ai_schedule()`](services/market_phase.py:93) - AI call timing by phase

**Why Important:** The system uses market phase to:
1. Determine when to allow new entries (only MID_MORNING_TREND, PM_BREAKOUT)
2. Schedule AI scan frequency
3. Adjust exit recommendations during POWER_HOUR
4. Display contextual guidance to users

---

### 1.5 Price Projection & Statistical Modeling

**Description:** Mathematical models for forecasting intraday price movements with confidence bands.

**Model Architecture** ([`services/price_projector.py:1`](services/price_projector.py:1)):
1. **Fourier Decomposition (40%)** - Extracts cyclic patterns from price data
2. **ARIMA(2,1,2) (30%)** - Time-series trend extrapolation
3. **VWAP/EMA Gravity (30%)** - Mean-reversion anchoring
4. **Monte Carlo Bands (±1σ)** - 500 simulations for confidence intervals

**Key Functions:**
- [`PriceProjector.generate_projection()`](services/price_projector.py:45) - Main entry point
- [`PriceProjector._fourier_projection()`](services/price_projector.py:172) - Cycle extraction
- [`PriceProjector._arima_projection()`](services/price_projector.py:228) - ARIMA forecasting
- [`PriceProjector._monte_carlo_bands()`](services/price_projector.py:272) - Confidence bands

**Why Important:** Price projections are displayed on the chart as visual targets for traders to validate against actual price action.

---

### 1.6 Stock Discovery & Screening

**Description:** Systems for identifying high-probability trading candidates from the NSE universe.

**Key Files:**
- [`services/stock_discovery.py`](services/stock_discovery.py:27) - `StockDiscoveryService`
- [`services/upstox_service.py`](services/upstox_service.py:70) - Instrument key resolution

**Discovery Methods:**
1. **NIFTY 100 Universe** - 50+ large-cap stocks ([`services/stock_discovery.py:11`](services/stock_discovery.py:11))
2. **Top Gainers Scanner** - Momentum-based filtering using gain % + volume surge
3. **Global Indices Fetch** - NIFTY 50, SENSEX, Bank Nifty, VIX, global markets

**Key Functions:**
- [`StockDiscoveryService.fetch_global_indices()`](services/stock_discovery.py:31) - Index data collection
- [`StockDiscoveryService._get_top_candidates()`](services/stock_discovery.py:143) - Momentum stock scanner
- [`get_instrument_key()`](services/upstox_service.py:70) - Yahoo Finance → Upstox key conversion

**Why Important:** Discovery powers the "SCAN" AI prompt type by providing candidate stocks for analysis.

---

## 2. Technical Skills

### 2.1 Python Development

**Description:** Core backend language using FastAPI, asyncio, and modern Python patterns.

**Key Frameworks/Libraries:**
- **FastAPI** - Web framework ([`main.py`](main.py:1))
- **asyncio** - Async operations for WebSocket handling
- **dataclasses** - Structured data models ([`models.py`](models.py:1))
- **pandas/numpy** - Data manipulation
- **pandas_ta** - Technical indicator library
- **scikit-learn** - ML algorithms (KNN, KMeans)

**Key Files:**
- [`main.py`](main.py:1) - FastAPI application entry point
- [`background_engine.py`](background_engine.py:1) - Async background processing
- [`ws_handler.py`](ws_handler.py:1) - WebSocket message routing

**Patterns Used:**
- Singleton pattern for services (e.g., `risk_engine = RiskEngine()`)
- Dependency injection via FastAPI
- Async/await for I/O-bound operations
- Dataclasses for structured configuration

**Why Important:** The entire backend is Python-based. Customization requires understanding of async patterns and FastAPI routing.

---

### 2.2 WebSocket Programming

**Description:** Real-time bidirectional communication for live price updates and UI synchronization.

**Architecture:**
- Server: Python FastAPI WebSocket ([`ws_handler.py`](ws_handler.py:1))
- Client: Browser JavaScript WebSocket ([`static/js/app.js:112`](static/js/app.js:112))

**Message Types:**
| Type | Direction | Purpose |
|------|-----------|---------|
| `get_chart_data` | Client → Server | Request OHLCV + indicators |
| `chart_data` | Server → Client | Chart payload response |
| `tick` | Server → Client | Live price updates |
| `state_update` | Server → Client | Full application state |
| `trigger_scan` | Client → Server | AI scan request |

**Key Functions:**
- [`ws_handler.py`](ws_handler.py:1) - WebSocket endpoint handler
- [`handleLiveTick()`](static/js/app.js:1455) - Client-side tick processing
- [`renderChart()`](static/js/app.js:490) - Chart rendering engine

**Why Important:** WebSocket is critical for:
1. Real-time P&L updates
2. Live price streaming from Upstox
3. AI scan result delivery
4. Chart data visualization

---

### 2.3 Database (SQLite + Alembic)

**Description:** Persistent storage for trades, settings, AI interactions, and market snapshots.

**Database:** SQLite (`trading_data.db`)

**Models** ([`models.py`](models.py:1)):
| Model | Purpose |
|-------|---------|
| `Trade` | Open/closed positions with full lifecycle |
| `MarketSnapshot` | Periodic market state capture |
| `AIInteraction` | AI API call audit trail |
| `AppSettings` | User configuration |
| `Watchlist` | Stock watchlists |
| `DailySummary` | Aggregated daily metrics |

**Migration Tool:** Alembic
- Config: [`alembic.ini`](alembic.ini:1)
- Environment: [`alembic/env.py`](alembic/env.py:1)
- Versions: [`alembic/versions/`](alembic/versions/)

**Key Functions:**
- [`database.py:18`](database.py:18) - `get_db()` session generator
- [`database.py:25`](database.py:25) - `cleanup_old_records()` - 100-day retention
- [`models.py:15`](models.py:15) - `Trade` model definition

**Why Important:** All trading history, AI interactions, and user settings persist across sessions. Database changes require Alembic migrations.

---

### 2.4 JavaScript/Frontend

**Description:** Lightweight frontend with TradingView Lightweight Charts and glassmorphic UI.

**Key Files:**
- [`static/index.html`](static/index.html:1) - Main dashboard
- [`static/js/app.js`](static/js/app.js:1) - Frontend logic (~1900 lines)
- [`static/css/style.css`](static/css/style.css:1) - Styling
- [`static/css/v3_enhancements.css`](static/css/v3_enhancements.css:1) - V3 features

**Libraries:**
- **Lightweight Charts** (TradingView) - Charting library
- **Vanilla JS** - No framework dependencies

**Key UI Components:**
- Real-time position cards with P&L
- Interactive chart with ML indicator overlays
- AI advisor result cards
- Market pulse ticker
- Timeline event feed
- Settings panel

**Key Functions:**
- [`renderPositions()`](static/js/app.js:825) - Position grid rendering
- [`renderChart()`](static/js/app.js:490) - Lightweight Charts integration
- [`handleStateUpdate()`](static/js/app.js:1344) - State synchronization
- [`openChart()`](static/js/app.js:119) - Chart modal management

**Why Important:** The UI is the user's primary interface. Understanding JavaScript is needed for:
1. Adding new chart indicators
2. Modifying UI components
3. Debugging WebSocket message handling

---

### 2.5 AI/ML Integration

**Description:** Multi-provider AI integration for market analysis, sentiment scoring, and strategy optimization.

**Supported Providers:**
1. **Google Gemini** - Primary provider
2. **Groq** - Fast inference
3. **SambaNova** - Alternative

**AI Services:**
- [`services/ai_scorer.py`](services/ai_scorer.py:33) - Trading signal generation
- [`services/news_sentiment.py`](services/news_sentiment.py:103) - Sentiment analysis
- [`services/strategy_tuner.py`](services/strategy_tuner.py:19) - Parameter optimization

**AI Prompt Types:**
1. **SCAN** - Find trading candidates ([`services/ai_scorer.py:48`](services/ai_scorer.py:48))
2. **POSITION_REVIEW** - Evaluate open trades ([`services/ai_scorer.py:125`](services/ai_scorer.py:125))
3. **EXIT_GUIDANCE** - Power hour exit decisions ([`services/ai_scorer.py:187`](services/ai_scorer.py:187))

**Quota Management:**
- [`services/quota_service.py`](services/quota_service.py:1) - API call tracking
- Max 7 AI calls/day (configurable)
- Automatic failover on rate limits

**Key Functions:**
- [`AIAdvisorService.scan_market()`](services/ai_scorer.py:48) - Candidate scanning
- [`NewsSentimentService.score_sentiment()`](services/news_sentiment.py:160) - News analysis
- [`StrategyTuner.optimize()`](services/strategy_tuner.py:30) - Parameter tuning loop

**Why Important:** AI provides the "Judgment Layer" in the Triple-Layer Consensus. Understanding prompt engineering and API integration is essential for modifying AI behavior.

---

### 2.6 Testing (pytest)

**Description:** Comprehensive test suite covering unit tests, integration tests, and UI tests.

**Test Files:**
- [`tests/test_technical_analysis.py`](tests/test_technical_analysis.py:1)
- [`tests/test_risk_engine.py`](tests/test_risk_engine.py:1)
- [`tests/test_ai_scorer.py`](tests/test_ai_scorer.py:1)
- [`tests/test_backtester.py`](tests/test_backtester.py:1)
- [`tests/test_market_phase.py`](tests/test_market_phase.py:1)
- [`tests/ui/`](tests/ui/) - UI interaction tests

**Testing Patterns:**
- **Fixtures** - [`tests/conftest.py`](tests/conftest.py:1)
- **Mock data** - Simulation mode for offline testing
- **Assertion helpers** - `assert_green()`, `assert_red()`

**Key Functions:**
- Mock OHLCV generation for testing
- Simulation mode: `SIMULATION=true` environment variable
- Backtester validation

**Why Important:** Tests ensure reliability of:
1. Indicator calculations
2. Risk engine validations
3. AI response parsing
4. Backtester accuracy

---

## 3. Project-Specific Skills

### 3.1 Upstox API Integration

**Description:** Integration with Upstox broker for market data and order execution.

**Key Files:**
- [`services/upstox_service.py`](services/upstox_service.py:88) - `UpstoxService`
- [`services/upstox_streamer.py`](services/upstox_streamer.py:1) - WebSocket market feed

**API Versions:**
- **V2** - Profiles, quotes, order placement
- **V3** - Historical candle data (primary)

**Key Endpoints:**
| Endpoint | Purpose |
|----------|---------|
| `/v2/user/profile` | Account verification |
| `/v2/market-quote/quotes` | Real-time quotes |
| `/v3/historical-candle/intraday` | Intraday OHLCV |
| `/v3/historical-candle` | Historical data |

**Key Functions:**
- [`UpstoxService.fetch_ohlcv()`](services/upstox_service.py:255) - OHLCV data fetch
- [`UpstoxService.fetch_market_quote()`](services/upstox_service.py:299) - Live quotes
- [`get_instrument_key()`](services/upstox_service.py:70) - Ticker resolution
- [`validate_token()`](services/upstox_service.py:136) - Auth validation

**Instrument Handling:**
- Uses Upstox instrument master (downloads from `assets.upstox.com`)
- Caches 50+ NSE instruments in memory
- Maps Yahoo Finance tickers (`.NS`) to Upstox keys (`NSE_EQ|...`)

**Why Important:** Upstox is the primary data source. Understanding the API is necessary for:
1. Adding new data endpoints
2. Debugging market data issues
3. Implementing order execution

---

### 3.2 Triple-Layer Consensus Engine

**Description:** The core decision-making architecture combining three independent analyses.

**Architecture Layers:**

```
┌─────────────────────────────────────────────────────┐
│                  USER DECISION                      │
├─────────────────────────────────────────────────────┤
│  LAYER 3: AI JUDGMENT (Google/Groq/SambaNova)      │
│  → "Should I take this trade based on news/sentiment"│
├─────────────────────────────────────────────────────┤
│  LAYER 2: MATH PROBABILITY (Pure Algorithms)         │
│  → "Is this setup mathematically sound?"            │
├─────────────────────────────────────────────────────┤
│  LAYER 1: RISK ENGINE (Deterministic)               │
│  → "Does this pass risk constraints?"              │
└─────────────────────────────────────────────────────┘
```

**Layer 1: Risk Engine** ([`services/risk_engine.py`](services/risk_engine.py:1))
- Cannot be overridden by AI
- Enforces: R:R ≥ 1.25, Max daily loss, Position sizing, Trend alignment

**Layer 2: Math Probability** ([`services/technical_analysis.py:214`](services/technical_analysis.py:214))
- Pure mathematical scoring (0.0 - 1.0)
- Checks: EMA alignment, RSI bounds, MACD direction, ADX strength
- Acts as "pre-filter" before AI

**Layer 3: AI Judgment** ([`services/ai_scorer.py`](services/ai_scorer.py:33))
- Considers: News sentiment, sector rotation, global cues
- Outputs: Action + confidence + reasoning

**Why Important:** This is the architectural foundation. Any changes must respect the consensus model and Layer 1's final authority.

---

### 3.3 Vectorized Backtesting

**Description:** Historical strategy testing using vectorized operations for speed.

**Key Files:**
- [`services/backtester.py`](services/backtester.py:9) - `VectorizedBacktester`

**Features:**
- Historical OHLCV data processing
- Indicator computation (EMA, RSI, VWAP, Lorentzian)
- Vectorized signal generation
- Trade simulation with SL/TP
- Metrics: Net profit, win rate, max drawdown, equity curve

**Key Functions:**
- [`VectorizedBacktester._prepare_indicators()`](services/backtester.py:20) - Indicator computation
- [`VectorizedBacktester.run_strategy()`](services/backtester.py:104) - Strategy execution
- [`VectorizedBacktester._evaluate_metrics()`](services/backtester.py:220) - Performance metrics

**Entry Logic:**
```python
buy_condition = (df['LZ_Signal'] == 1) & \
                (df[ema_fast_col] > df[ema_slow_col]) & \
                (df['Close'] > df[vwap_col])
```

**Why Important:** Backtesting validates strategy effectiveness before live deployment. The vectorized approach handles large datasets efficiently.

---

### 3.4 Strategy Tuning & Optimization

**Description:** AI-driven parameter optimization for strategy improvement.

**Key Files:**
- [`services/strategy_tuner.py`](services/strategy_tuner.py:19) - `StrategyTuner`

**Optimization Process:**
1. Run backtest with initial parameters
2. Analyze metrics (net profit, drawdown, win rate)
3. AI generates parameter mutations
4. Re-run backtest with new parameters
5. Repeat for N iterations

**Key Functions:**
- [`StrategyTuner.optimize()`](services/strategy_tuner.py:30) - Main optimization loop

**Default Parameters:**
```python
{
    'ema_fast': 9,
    'ema_slow': 21,
    'rsi_len': 14,
    'rsi_buy_threshold': 40,
    'rsi_short_threshold': 60,
    'sl_pct': 0.01,
    'tp_pct': 0.02
}
```

**Goals:**
- Maximize net profit
- Keep max drawdown < 10%
- Maintain win rate > 45%

**Why Important:** Strategy tuning allows continuous improvement of trading parameters based on historical performance.

---

### 3.5 News Sentiment Analysis

**Description:** Real-time news aggregation and AI-powered sentiment scoring.

**Key Files:**
- [`services/news_sentiment.py`](services/news_sentiment.py:103) - `NewsSentimentService`

**News Sources:**
1. **Google News RSS** - Default source
2. **DuckDuckGo** - Fallback
3. **Tavily** - AI search (optional)

**Sentiment Methods:**
1. **AI Scoring** - Gemini/Groq/SambaNova analysis
2. **Keyword Fallback** - Local keyword matching when AI unavailable

**Keyword Lists:**
- Positive: surge, rally, profit, growth, bull...
- Negative: crash, decline, loss, fraud, ban...

**Key Functions:**
- [`NewsSentimentService.fetch_news()`](services/news_sentiment.py:122) - News aggregation
- [`NewsSentimentService.score_sentiment()`](services/news_sentiment.py:160) - AI sentiment analysis
- [`_keyword_sentiment()`](services/news_sentiment.py:35) - Fallback scoring

**Output Format:**
```json
{
    "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
    "reason": "Brief explanation"
}
```

**Why Important:** News sentiment provides context for the AI Judgment Layer, helping identify catalysts for trade setups.

---

## Quick Reference: File to Skill Mapping

| Skill | Primary Files |
|-------|---------------|
| NSE Trading | `upstox_service.py`, `market_phase.py`, `stock_discovery.py` |
| Technical Analysis | `technical_analysis.py`, `advanced_indicators.py` |
| Risk Management | `risk_engine.py` |
| Market Phase | `market_phase.py` |
| Price Projection | `price_projector.py` |
| Stock Discovery | `stock_discovery.py` |
| Python/FastAPI | `main.py`, `background_engine.py`, `ws_handler.py` |
| WebSocket | `ws_handler.py`, `app.js` |
| Database | `database.py`, `models.py`, `alembic/` |
| Frontend | `app.js`, `index.html`, `style.css` |
| AI Integration | `ai_scorer.py`, `news_sentiment.py`, `strategy_tuner.py` |
| Testing | `tests/*.py` |
| Upstox API | `upstox_service.py`, `upstox_streamer.py` |
| Consensus Engine | All services - see architecture |
| Backtesting | `backtester.py` |
| Strategy Tuning | `strategy_tuner.py` |
| Sentiment | `news_sentiment.py` |

---

## Prerequisites for Development

1. **Python 3.10+** with pip
2. **API Keys** (`.env`):
   - `UPSTOX_ACCESS_TOKEN` - Broker authentication
   - `GEMINI_API_KEY` - Primary AI
   - `GROQ_API_KEY` - Fast AI fallback
   - `SAMBA_API_KEY` - Alternative AI
   - `TAVILY_API_KEY` - News search
3. **Node.js** (optional) - For frontend development
4. **SQLite** - Built into Python

---

*Last Updated: March 2026*
*Project: SuperNova V3 - AI Trading Companion for NSE*
