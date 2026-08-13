# SuperNova - AI Trading Companion Agent Documentation

> **Agent Purpose**: This document serves as the memory bank for Kilo Code to understand the SuperNova trading project. It provides comprehensive context for development, debugging, and feature implementation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Key Components](#3-key-components)
4. [Services Architecture](#4-services-architecture)
5. [Data Flow](#5-data-flow)
6. [Database Schema](#6-database-schema)
7. [Development Guidelines](#7-development-guidelines)
8. [Testing Standards](#8-testing-standards)
9. [Coding Conventions](#9-coding-conventions)
10. [Known Issues](#10-known-issues)

---

## 1. Project Overview

**SuperNova** is an AI-augmented intraday trading advisor designed specifically for India's **National Stock Exchange (NSE)**. The system combines deterministic technical indicators with advanced Large Language Models (LLMs) to generate high-conviction trade setups with real-time risk management.

### Core Capabilities

- **Real-time Market Analysis**: Technical indicators (EMA, VWAP, RSI, ADX, MACD, Bollinger Bands)
- **AI-Powered Signals**: Multi-provider AI integration (Google Gemini, Groq, SambaNova)
- **Risk Management**: ATR-based stop-loss and target calculation with position sizing
- **Market Phase Tracking**: Indian market session state machine (Pre-market, Opening 15, Mid-Session, Power Hour, Closing)
- **Live Trading**: Upstox API integration for order execution and real-time streaming
- **Glassmorphic Dashboard**: Modern UI with Lightweight Charts for visualization

### Target Users

Intraday traders in India who want AI-augmented trade recommendations with deterministic risk management.

---

## 2. Tech Stack

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.10+ |
| **FastAPI** | REST API & WebSocket server | Latest |
| **Uvicorn** | ASGI server | Latest |
| **SQLAlchemy** | ORM | 2.0+ |
| **SQLite** | Database | Built-in |
| **pandas** | Data analysis | Latest |
| **pandas-ta** | Technical indicators | Latest |
| **python-dotenv** | Environment configuration | Latest |

### External APIs

| Service | Purpose |
|---------|---------|
| **Upstox API** | Market data, order execution, live streaming |
| **Google Gemini** | AI analysis (primary) |
| **Groq (Llama)** | AI analysis (fallback) |
| **SambaNova** | AI analysis (fallback) |
| **Tavily** | News sentiment & research |

### Frontend

| Technology | Purpose |
|------------|---------|
| **Vanilla JavaScript** | Frontend logic |
| **HTML5/CSS3** | Structure & styling |
| **Lightweight Charts** | Financial charting (TradingView) |
| **WebSocket** | Real-time data streaming |

### Infrastructure

- **Database Migrations**: Alembic
- **Testing**: pytest
- **Logging**: Python logging module

---

## 3. Key Components

### Entry Point

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | FastAPI application entry point, OAuth handling, REST API routes, WebSocket endpoint |
| [`requirements.txt`](requirements.txt) | Python dependencies |

### Core Backend

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [`ws_handler.py`](ws_handler.py) | WebSocket message routing, trading command handling | `ConnectionManager`, `handle_websocket()` |
| [`background_engine.py`](background_engine.py) | Phase-aware market monitoring, scheduled AI scanning | `BackgroundEngine` class |
| [`database.py`](database.py) | SQLAlchemy engine, session factory | `SessionLocal`, `Base`, `wipe_all_data()` |
| [`models.py`](models.py) | Database schema definitions | `Trade`, `MarketSnapshot`, `AIInteraction`, `AppSettings` |

### State Management

| File | Purpose | Key Class |
|------|---------|-----------|
| [`services/state.py`](services/state.py) | Global application state, trade management | `AppState` class |

---

## 4. Services Architecture

The services layer contains 20+ modular services, each with single responsibility.

### Core Analysis Services

| Service | File | Purpose | Key Functions |
|---------|------|---------|---------------|
| **Technical Analysis** | [`services/technical_analysis.py`](services/technical_analysis.py) | Calculate indicators, generate LZ mathematical scores | `compute_indicators()`, `evaluate_math_probability()`, `classify_signal()`, `get_chart_payload()` |
| **AI Scorer** | [`services/ai_scorer.py`](services/ai_scorer.py) | Generate prompts, process AI responses, multi-provider fallback | `scan_market()`, `review_positions()`, `exit_guidance()`, `_call_ai()` |
| **Risk Engine** | [`services/risk_engine.py`](services/risk_engine.py) | ATR-based stop-loss, target, position sizing | `calculate_atr_based_levels()`, `validate_trade_setup()` |
| **Market Phase** | [`services/market_phase.py`](services/market_phase.py) | Indian market session state machine | `get_current_phase()`, `is_market_open()`, `minutes_to_close()` |

### Data Services

| Service | File | Purpose | Key Functions |
|---------|------|---------|---------------|
| **Upstox Service** | [`services/upstox_service.py`](services/upstox_service.py) | Historical data, order execution | `get_historical_data()`, `place_order()` |
| **Upstox Streamer** | [`services/upstox_streamer.py`](services/upstox_streamer.py) | Real-time WebSocket streaming | `connect_and_stream()` |
| **Stock Discovery** | [`services/stock_discovery.py`](services/stock_discovery.py) | Multi-factor stock screening | Discovery algorithms |

### Advanced Services

| Service | File | Purpose |
|---------|------|---------|
| **Advanced Indicators** | [`services/advanced_indicators.py`](services/advanced_indicators.py) | Additional technical indicators |
| **Price Projector** | [`services/price_projector.py`](services/price_projector.py) | Future price probability modeling |
| **Projection Mapper** | [`services/projection_mapper.py`](services/projection_mapper.py) | Map price projections to trades |
| **Backtester** | [`services/backtester.py`](services/backtester.py) | Historical strategy testing |
| **News Sentiment** | [`services/news_sentiment.py`](services/news_sentiment.py) | News analysis for sentiment |
| **Strategy Tuner** | [`services/strategy_tuner.py`](services/strategy_tuner.py) | AI-based parameter optimization |
| **Quota Service** | [`services/quota_service.py`](services/quota_service.py) | API rate limiting |
| **Sentinel Service** | [`services/sentinel_service.py`](services/sentinel_service.py) | System health monitoring |
| **Holiday Service** | [`services/holiday_service.py`](services/holiday_service.py) | Market holiday calendar |
| **OpenChart Service** | [`services/openchart_service.py`](services/openchart_service.py) | Chart data preparation |

---

## 5. Data Flow

### Market Data Flow

```
Upstox API/Streamer
        ↓
background_engine.py (Phase-aware loop)
        ↓
services/technical_analysis.py (Indicators)
        ↓
services/ai_scorer.py (AI analysis)
        ↓
services/risk_engine.py (Risk validation)
        ↓
ws_handler.py (WebSocket routing)
        ↓
static/js/app.js (Frontend display)
```

### Trading Command Flow

```
Frontend (static/js/app.js)
        ↓
WebSocket (/ws endpoint)
        ↓
ws_handler.py (Command routing)
        ↓
services/technical_analysis.py (Get indicators)
        ↓
services/ai_scorer.py (AI validation)
        ↓
services/risk_engine.py (Calculate levels)
        ↓
services/upstox_service.py (Execute order)
        ↓
services/state.py (Update state)
        ↓
Database (models.py)
        ↓
WebSocket broadcast (Real-time update)
```

### AI Analysis Flow

```
background_engine.py (_scan_cycle)
        ↓
services/technical_analysis.py (Get top candidates)
        ↓
services/stock_discovery.py (Filter by fundamentals)
        ↓
services/ai_scorer.py (scan_market)
        ↓
AI Provider (Google → Groq → SambaNova fallback)
        ↓
services/risk_engine.py (Validate setup)
        ↓
Database (AIInteraction log)
        ↓
WebSocket broadcast (AI picks to UI)
```

---

## 6. Database Schema

### Core Tables

#### Trade (`trades`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (PK) | UUID trade identifier |
| `ticker` | String | Stock symbol (e.g., RELIANCE.NS) |
| `action` | String | BUY or SHORT SELL |
| `quantity` | Integer | Number of shares |
| `entry_price` | Float | Entry price |
| `stop_loss` | Float | Stop loss price |
| `target_1` | Float | First target (1.5:1 R:R) |
| `target_2` | Float | Second target (3:1 R:R) |
| `exit_price` | Float (nullable) | Exit price |
| `pnl` | Float (nullable) | Profit/Loss |
| `status` | String | OPEN or CLOSED |
| `phase_entered` | String | Market phase when entered |
| `ai_reasoning` | Text | AI recommendation reasoning |
| `valid_until` | DateTime | Recommendation expiry |
| `trailing_sl` | Float | Dynamic trailing stop loss |
| `atr_at_entry` | Float | ATR at entry time |
| `risk_per_share` | Float | Risk per share |
| `max_loss` | Float | Maximum loss for trade |
| `partial_exits_json` | Text | JSON for partial exits |
| `ai_updates_json` | Text | JSON for AI guidance history |

#### MarketSnapshot (`market_snapshots`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment |
| `timestamp` | DateTime | Snapshot time |
| `market_phase` | String | Current phase |
| `nifty_price` | Float | Nifty 50 price |
| `nifty_change_pct` | Float | Nifty change % |
| `banknifty_price` | Float | Bank Nifty price |
| `banknifty_change_pct` | Float | Bank Nifty change % |
| `vix` | Float | India VIX |
| `advances` | Integer | Advances |
| `declines` | Integer | Declines |
| `sector_data_json` | Text | Sector performance JSON |

#### AIInteraction (`ai_interactions`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment |
| `timestamp` | DateTime | Interaction time |
| `prompt_type` | String | SCAN, POSITION_REVIEW, EXIT_GUIDANCE |
| `model_used` | String | Model name (gemini-2.0-flash, etc.) |
| `tokens_used` | Integer | Token count |
| `input_summary` | Text | Abbreviated prompt |
| `output_json` | Text | Full AI response |
| `was_acted_upon` | Boolean | User acted on advice |
| `trade_date` | String | YYYY-MM-DD |

#### AppSettings (`app_settings`)

| Column | Type | Default |
|--------|------|---------|
| `capital` | Float | 100000.0 |
| `max_loss_per_trade` | Float | 1000.0 |
| `max_daily_loss` | Float | 5000.0 |
| `search_engine` | String | tavily |
| `data_provider` | String | upstox |
| `fallback_data` | Boolean | true |
| `fallback_search` | Boolean | true |
| `fallback_ai` | Boolean | true |
| `ai_provider` | String | google |
| `ai_model` | String | gemini-3.1-pro |

### Supporting Tables

- **Watchlist** (`watchlists`): User stock watchlists
- **WatchlistStock** (`watchlist_stocks`): Stocks in watchlists
- **DashboardStock** (`dashboard_stocks`): Dashboard watch stocks
- **ApiUsage** (`api_usage`): API quota tracking
- **DailySummary** (`daily_summaries`): Daily performance metrics

---

## 7. Development Guidelines

### Project Structure

```
gemini_nse_trader/
├── main.py                 # FastAPI entry point
├── ws_handler.py          # WebSocket handler
├── background_engine.py    # Background tasks
├── database.py             # DB connection
├── models.py               # SQLAlchemy models
├── requirements.txt        # Dependencies
├── alembic.ini            # Migration config
├── static/                # Frontend assets
│   ├── index.html
│   ├── watchlists.html
│   ├── css/style.css
│   └── js/app.js
├── services/              # Business logic
│   ├── technical_analysis.py
│   ├── ai_scorer.py
│   ├── risk_engine.py
│   ├── market_phase.py
│   └── ... (20+ services)
├── tests/                 # Test suite
│   ├── test_*.py
│   └── ui/test_*.py
├── docs/                  # Documentation
└── alembic/               # Migrations
    └── versions/
```

### Running the Application

```bash
# Development
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

Create `.env` file with:

```env
# Upstox API
UPSTOX_API_KEY=your_api_key
UPSTOX_API_SECRET=your_api_secret
UPSTOX_ACCESS_TOKEN=your_token

# AI Providers
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
SAMBANOVA_API_KEY=your_sambanova_key

# Configuration
SIMULATION=false
DATA_PROVIDER=upstox
AI_PROVIDER=google
AI_MODEL=gemini-3.1-pro
```

### Adding a New Service

1. Create file in `services/` directory
2. Implement stateless service class or functions
3. Import in relevant handler (`ws_handler.py` or `background_engine.py`)
4. Add tests in `tests/` directory

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 8. Testing Standards

### Test Organization

Tests are located in the `tests/` directory:

| Directory | Purpose |
|-----------|---------|
| `tests/` | Unit tests for services |
| `tests/ui/` | Frontend/UI interaction tests |

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_technical_analysis.py

# With coverage
pytest --cov=. --cov-report=html
```

### Test Files

| Test File | Coverage |
|-----------|----------|
| `test_technical_analysis.py` | TechnicalAnalysisService |
| `test_ai_scorer.py` | AIAdvisorService |
| `test_risk_engine.py` | RiskEngine |
| `test_market_phase.py` | MarketPhaseTracker |
| `test_background_engine.py` | BackgroundEngine |
| `test_upstox_service.py` | UpstoxService |
| `test_upstox_streamer.py` | UpstoxStreamer |
| `test_stock_discovery.py` | StockDiscovery |
| `test_backtester.py` | Backtester |
| `test_price_projector.py` | PriceProjector |
| `test_news_sentiment.py` | NewsSentiment |
| `test_strategy_tuner.py` | StrategyTuner |
| `test_ws_handler.py` | WebSocket handler |
| `test_database.py` | Database operations |
| `test_state.py` | AppState |
| `test_quota_service.py` | Quota management |

### UI Tests

| Test File | Coverage |
|-----------|----------|
| `test_supernova_comprehensive.py` | Full UI workflow |
| `test_v3_chart_interactions.py` | Chart interactions |
| `test_v3_settings_persistence.py` | Settings save/load |
| `test_trading_intent.py` | Trading workflow |

### Test Patterns

- Use pytest fixtures from `tests/conftest.py`
- Mock external APIs (Upstox, AI providers)
- Test both success and failure paths
- Include async tests for WebSocket handlers

---

## 9. Coding Conventions

### Python Style

- **PEP 8** compliance
- Type hints for function parameters and return values
- Docstrings for all public classes and functions
- Use `logging` module for logging (not print)

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `TechnicalAnalysisService` |
| Functions | snake_case | `calculate_indicators()` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Files | snake_case | `technical_analysis.py` |
| Database columns | snake_case | `entry_price` |

### Key Patterns

#### Service Pattern

Services should be stateless:

```python
class TechnicalAnalysisService:
    def __init__(self):
        # No state - stateless service
        pass
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        # Pure function
        pass
```

#### Error Handling

Always handle errors gracefully with fallbacks:

```python
try:
    result = primary_method()
except Exception as e:
    logger.warning(f"Primary failed: {e}")
    result = fallback_method()
```

#### WebSocket Messages

Use structured message format:

```python
{
    "type": "tick" | "scan_result" | "trade_update" | "error",
    "data": {...}
}
```

### Mandatory Rules

See [`SUPERNOVA_MANDATORY_RULES.md`](SUPERNOVA_MANDATORY_RULES.md) for critical development rules including:

- AI fallback requirements
- Data provider fallbacks
- Phase-aware trading rules
- Risk management requirements

---

## 10. Known Issues

### Critical Issues

#### Issue #1: Chart Not Refreshing on Same Ticker

**Location**: [`static/js/app.js`](static/js/app.js:302-320)

**Symptom**: When opening the chart for the same symbol a second time, the chart visual shows data from the previously opened symbol, but OHLC/LTP header values are correctly updated.

**Root Cause**: The `renderChart()` function only clears data when ticker CHANGES (`isNewTicker` check). When reopening the same ticker, old data is merged with new data.

**Fix**: Clear chart state in `openChart()` function before requesting new data:

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
    
    // ... rest of function
};
```

### High Priority Issues

#### Issue #2: No Loading State

**Location**: [`static/js/app.js`](static/js/app.js:72) - `openChart()` function

**Problem**: No visual feedback while waiting for WebSocket response. User sees stale data until new data arrives.

#### Issue #3: Chart Not Properly Cleaned on Close

**Location**: [`static/js/app.js`](static/js/app.js:84-89)

**Problem**: Only sets references to null but doesn't call `.destroy()` on LightweightCharts instances.

**Impact**: Memory leaks and zombie event listeners.

#### Issue #4: Ticker Normalization Inconsistency

**Location**: Multiple places - `openChart()`, `renderChart()`, backend handler

**Problem**: Inconsistent `.NS` suffix handling. Comparison may fail if backend returns "RELIANCE" but `lastChartTicker` is "RELIANCE.NS".

### Medium Priority Issues

#### Issue #5: Global State Usage

**Location**: [`static/js/app.js:30`](static/js/app.js:30)

**Problem**: Uses `window.lastChartTicker` instead of `appState`.

#### Issue #6: Interval Switching May Not Clear Properly

**Location**: [`static/js/app.js:167-185`](static/js/app.js:167-185)

**Problem**: When switching intervals, doesn't set `window.lastChartTicker = ""`.

#### Issue #7: yfinance Cache - No Size Limit

**Location**: [`services/technical_analysis.py:105-125`](services/technical_analysis.py:105)

**Problem**: Cache never expires (only 5-minute window), no size limit.

**Impact**: Memory grows unbounded in long-running apps.

#### Issue #8: No Error Boundary for Chart

**Location**: [`static/js/app.js`](static/js/app.js) - `renderChart()` function

**Problem**: No try-catch, failures are silent.

#### Issue #9: Live Tick Race Condition

**Location**: [`static/js/app.js:432-459`](static/js/app.js:432-459)

**Problem**: If chart data hasn't loaded but live ticks arrive, errors occur.

#### Issue #10: ADX Chart Resize Not Synced

**Location**: [`static/js/app.js:296-299`](static/js/app.js:296-299)

**Problem**: ResizeObserver only watches main container, not adxContainer.

---

## Quick Reference

### Common Tasks

| Task | File to Modify |
|------|----------------|
| Add new technical indicator | `services/technical_analysis.py` |
| Change AI provider | `services/ai_scorer.py` |
| Modify risk calculations | `services/risk_engine.py` |
| Update chart behavior | `static/js/app.js` |
| Add database field | `models.py` + create migration |
| Add new WebSocket command | `ws_handler.py` |

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws` | WebSocket | Real-time data & commands |
| `/` | GET | Main dashboard |
| `/upstox/connect` | GET | OAuth redirect |
| `/api/watchlists` | GET/POST | Watchlist CRUD |
| `/api/dashboard/stocks` | GET/POST | Dashboard stocks |

---

*Document Version: 1.0*  
*Last Updated: 2026-03-20*  
*Project: SuperNova - AI Trading Companion*
