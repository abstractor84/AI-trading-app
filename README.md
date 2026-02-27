# SuperNova: AI Trading Companion ◈

**SuperNova** is a professional-grade, AI-augmented intraday trading advisor designed specifically for the National Stock Exchange (NSE). It combines deterministic mathematical indicators with advanced Large Language Models (LLMs) to provide high-conviction trade setups, objective risk management, and real-time portfolio monitoring.

---

## 🚀 Key Features

### 1. The Multi-Layered Consensus Engine
SuperNova doesn't just trade on "patterns"; it requires a triple-layer validation before recommending any ticker:
- **Level 1: LZ AI (Math Probability):** A specialized technical scoring engine that assigns a probability (0.0 to 1.0) based on EMA9/21 alignment, VWAP positioning, RSI exhaustion, and ADX trend strength. Only setups > 0.50 are considered.
- **Level 2: AI Validation:** Top mathematical candidates are analyzed by advanced AI models (Gemini, Groq) to provide surgical 2-line technical reasoning.
- **Level 3: Consensus Blocker:** A native logic gate that automatically vetoes AI suggestions if they contradict the underlying mathematical trend (e.g., blocking an AI "BUY" if EMA9 < EMA21).

### 2. Safeguarded Risk Management
- **Volatility-Adjusted Levels:** Stop Losses and Targets are calculated dynamically using **Average True Range (ATR)**, adapting to the stock's specific "noise" level.
- **Strict R:R Gatekeeping:** Every trade must pass a minimum **1.25:1 Risk/Reward ratio**.
- **Market Phase Awareness:** The system acts differently during the *Opening 15*, *Mid-Session*, and *Power Hour*. It forbids new trades as the closing bell approaches to prevent overnight traps.
- **Automated Position Review:** Continuous background monitoring of open trades with advice on trailing SL, booking partial profits (at 1%), or urgency-based exits.

### 3. Advanced Technical Utilities
- **Price Projection Engine:** Generates future price probability bands using statistical modeling and AI forecasting.
- **Vectorized Backtester:** Test strategies on 30-day historical data with 5-minute granularity.
- **AI Strategy Tuner:** Uses LLMs to optimize technical parameters (EMA lengths, RSI thresholds) based on historical performance.

---

## 🛠 Project Structure

```text
├── main.py                # Entrypoint (FastAPI + WebSocket wiring)
├── background_engine.py   # Phase-aware market loop & AI scheduler
├── ws_handler.py          # Real-time command routing (Scan, Trade, Close)
├── services/
│   ├── technical_analysis.py # Indicator calculations (EMA, VWAP, LZ Score)
│   ├── risk_engine.py        # ATR-based SL/Target & Trade Validation
│   ├── ai_scorer.py          # AI Prompting & Output Enrichment
│   ├── market_phase.py       # Indian Market time-state machine
│   ├── stock_discovery.py    # Multi-factor ticker screening
│   └── price_projector.py    # Projection & charting logic
├── static/                
│   ├── index.html         # Real-time Glassmorphic Dashboard
│   └── js/app.js          # WebSocket logic & Charting engine
└── trading_data.db        # SQLite persistence for trades & journals
```

---

## ⚙️ Technical Stack
- **Backend:** Python, FastAPI, SQLAlchemy, Uvicorn
- **Data:** Upstox API V2 / yfinance (Flexible Provider)
- **AI Models:** Google Gemini 2.0 Flash, Groq (Llama 3.1/3.3), SambaNova
- **Frontend:** Vanilla JS, Tailwind-inspired CSS, Lightweight Charts
- **Persistence:** SQLite3

---

## 🚦 Getting Started

1. **Environment Setup:**
   - Clone the repo and create a `.env` file with your `UPSTOX_API_KEY`, `GEMINI_API_KEY`, and `GROQ_API_KEY`.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Authentication:**
   - Run the app: `python main.py`.
   - Complete the **Upstox OAuth** flow via the UI to enable real-time NSE data.

3. **Deploy:**
   - Use the **Scan Market Now** button for manual scouting or let the **Background Engine** cycle through automated setups during market hours.

---

## ⚖️ Disclaimer
*SuperNova is an advisory companion. Trading involves significant risk. The developers are not responsible for financial losses incurred through the use of this software. Always validate signals with your own financial advisor.*

◈ **Built with precision for the Indian Market.**
