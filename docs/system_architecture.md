# System Architecture Document

## Overview
SuperNova is an AI-augmented intraday trading advisor for the National Stock Exchange (NSE), combining deterministic technical indicators with advanced Large Language Models (LLMs) for high-conviction trade setups. The system provides real-time risk management and portfolio monitoring.

## Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI       │    │   Background    │
│   (HTML/JS/CSS) │◄──►│   Server        │◄──►│   Engine        │
│                 │    │   (WebSocket)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Services      │
                    │   Layer         │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │   Database      │
                    │   (SQLite)      │
                    └─────────────────┘
```

## Components

### 1. Frontend Layer
- **Technology**: Vanilla JavaScript, HTML5, CSS3
- **Purpose**: User interface for dashboard, charts, watchlists
- **Communication**: WebSocket for real-time data, HTTP for REST API
- **Key Files**:
  - `static/index.html`: Main dashboard
  - `static/js/app.js`: WebSocket client and charting logic
  - `static/css/`: Styling (glassmorphic design)

### 2. Backend Layer
- **Technology**: FastAPI (Python), Uvicorn ASGI server
- **Purpose**: API server handling requests, WebSocket connections, OAuth
- **Key Components**:
  - **main.py**: Application entrypoint, FastAPI app setup
  - **ws_handler.py**: WebSocket message routing and command handling
  - **REST API**: Endpoints for watchlists, dashboard stocks
- **Communication**: 
  - REST API for CRUD operations
  - WebSocket for real-time trading commands and market data

### 3. Services Layer
Modular services handling business logic:
- **technical_analysis.py**: Indicator calculations (EMA, VWAP, RSI, ADX)
- **ai_scorer.py**: AI prompt generation and response processing
- **risk_engine.py**: ATR-based stop-loss/target calculation
- **market_phase.py**: Indian market session state machine
- **stock_discovery.py**: Multi-factor stock screening
- **upstox_service.py**: NSE data fetching via Upstox API
- **upstox_streamer.py**: Real-time market data streaming
- **backtester.py**: Historical strategy testing
- **price_projector.py**: Future price probability modeling
- **news_sentiment.py**: News analysis for market sentiment
- **strategy_tuner.py**: AI-based parameter optimization

### 4. Background Engine
- **File**: `background_engine.py`
- **Purpose**: Scheduled market monitoring and AI calls
- **Features**:
  - Phase-aware operation (Opening 15, Mid-Session, Power Hour)
  - Automated AI scoring during market hours
  - Trade position monitoring and exit guidance
  - Market snapshot capturing

### 5. Data Layer
- **Technology**: SQLite with SQLAlchemy ORM
- **Purpose**: Persistent storage for trades, settings, AI interactions
- **Key Tables**:
  - `trades`: Open/closed positions with AI reasoning
  - `market_snapshots`: Periodic market state for context
  - `ai_interactions`: Audit log of all AI API calls
  - `app_settings`: User preferences and limits
  - `watchlists`: User-defined stock groups
  - `daily_summaries`: Performance metrics

### 6. External Integrations
- **Upstox API**: Market data, order execution, real-time streaming
- **AI Providers**: Google Gemini, Groq (Llama), SambaNova
- **Search Engine**: Tavily for news and research
- **Fallback Mechanisms**: Multiple providers for reliability

## Data Flow
1. **Market Data**: Upstox streamer → Background engine → Services → WebSocket → Frontend
2. **Trading Commands**: Frontend → WebSocket → Handler → Services → Upstox API
3. **AI Analysis**: Background engine → AI scorer → LLM API → Risk validation → Database
4. **User Actions**: Frontend → REST API → Database

## Deployment
- **Local Development**: Single process with Uvicorn reload
- **Production**: ASGI server (Uvicorn/Gunicorn) behind reverse proxy
- **Database**: SQLite for simplicity, can scale to PostgreSQL
- **Environment**: Python virtual environment with .env configuration

## Security Considerations
- API keys stored in environment variables
- OAuth flow for Upstox authentication
- Input validation on all endpoints
- Rate limiting for AI API calls (quota_service.py)

## Scalability
- Modular service architecture allows horizontal scaling
- WebSocket connections for real-time features
- Background processing separated from request handling
- Database can be migrated to distributed systems if needed