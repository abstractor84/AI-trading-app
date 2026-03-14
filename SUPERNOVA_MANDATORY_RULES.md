# SuperNova V3: Consolidated Mandatory Rules & Engineering Policies

This document serves as the absolute source of truth for all development, maintenance, and refactoring tasks within the SuperNova project. These rules are non-negotiable and must be adhered to without exception.

## 1. Core Project Intent & Accuracy Targets
- **Indicator Weighting:** 90% Math/Statistical/ML Indicators | 10% AI/News/Sentiment Assistance.
- **Performance Target:** 80% Accuracy in 180-day historical backtests.
- **Trading Focus:** Intraday operations exclusively.
- **Timezone Integrity:** All operations must respect IST (Asia/Kolkata). Market hours are strictly 09:15 to 15:30 IST. No "Ghost Data" is permitted outside these hours.
- **Retention:** Trade History must support a 90-day detailed view with clickable symbols for chart restoration.

## 2. Mandatory "Skeptic Review" Protocol
After **every** code change, fix, or feature implementation, a comprehensive Skeptic Review must be performed and documented. The review MUST cover:
- **Core Logic:** Architecture, Accuracy, Correctness, Math & Statistical Model calculations.
- **Risk & Stability:** Risk Engine integrity, Dependency check, Regression analysis, Exception handling.
- **Intent Alignment:** Trading intent for intraday, Project intent (90/10 split), AI Quota management.
- **Connectivity:** Connections, Interfaces, Data Flows, WebSocket state sync.
- **UI/UX:** UI Functionality, Chart Rendering (depth & sync), Settings restoration, 100% UI/Backend parity.
- **ML Dynamicness:** Ensure indicators adapt to timeframe switches and dataset lengths (Adaptive Lookback).

## 3. Engineering & Security Standards
- **Contextual Precedence:** `GEMINI.md` mandates take absolute precedence over general defaults.
- **Security First:** Never log, print, or commit secrets, API keys, or `.env` contents. Protect `.git` and configuration folders.
- **Surgical Implementation:** Apply targeted changes. Avoid unrelated refactoring or "cleanup."
- **Idiomatic Quality:** Adhere to existing naming, formatting, and architectural patterns. Ensure type safety and comprehensive documentation.
- **Library Verification:** Never assume a library is available. Check `requirements.txt` or `pip list` before usage.
- **No Destructive Actions:** Do not use `rm` or `git reset` on project files unless explicitly directed.

## 4. Development Lifecycle (Plan-Act-Validate)
1. **Research:** Systematically map the codebase. **Empirical reproduction of issues is mandatory before fixing.**
2. **Strategy:** A user-approved plan is the single source of truth.
3. **Execution:** Resolve sub-tasks via iterative Plan -> Act -> Validate cycles.
4. **Validation:** A task is only complete when behavioral correctness is verified via automated tests AND structural integrity is confirmed within the full project context.

## 5. Restoration & Regression Prevention Policies
- **News Sentinel:** Must be hardcoded to use DuckDuckGo Search (DDGS). All DDGS calls MUST be wrapped in a global `threading.Lock` and executed via `asyncio.to_thread` to prevent segmentation faults.
- **Safe Dictionary Access:** Use a surgically safe extractor (`.get()` with multiple fallbacks for `title`, `url`, `link`, `href`) for all external API results.
- **Defensive Indexing:** All list, Series, and DataFrame index accesses (especially in ML logic and loops) MUST be guarded by range checks or try-except blocks. `IndexError` is a fatal regression and must be proactively prevented by validating bounds before `iloc` or `index[j]` access.
- **Frontend Charting:** 
    - Markers (LZ, ST) MUST be **Sorted by Time** chronologically before calling `setMarkers` (Lightweight Charts requirement).
    - Legends must display real-time values (LTP, ADX, RSI, ST, LZ score).
    - Time parsing must use `Number()` or `+` to prevent string concatenation in epoch calculations.
- **Market Data Ordering:** 
    - **Global Row:** USD/INR, Brent Crude, WTI Crude, Gold, Silver must remain sticky in the leftmost positions (Index 0-4).
    - **India Row:** GIFT Nifty (using `^NSEI` proxy) must be prioritized.
- **State Persistence:** "Close Position" actions must finalize DB commits before broadcasting `trades_update`. State filtering must use strict string comparison for IDs.

## 6. Testing Requirements
- **Automated Tests:** Every change requires a corresponding unit or UI test.
- **Simulation Mode:** Validation must be performed with `SIMULATION=true` to ensure deterministic behavior and proper market hour capping.
- **Log Inspection:** Always check `uvicorn_latest.log` and internal debug logs for "SKEPTIC:" markers and hidden exceptions.

## 7. Application Startup & Log Monitoring
- **Startup Validation:** After any structural change or logic fix, the application MUST be started locally. The engineer must verify that the server reaches "Application startup complete" without tracebacks.
- **Persistence Verification:** After startup, logs MUST be monitored for at least 2 minutes to ensure the process doesn't exit prematurely. Final verification MUST use `lsof -i :8000` (or the configured port) to confirm the listener is active before proceeding to tests.
- **Log Monitoring:** uvicorn_latest.log MUST be monitored during startup and during test execution. Any "ERROR" or "WARNING" (especially 400/401/500 API errors) must be treated as a regression and resolved immediately. No fix is complete if the logs show persistent failures.

---
**STRICT ADHERENCE MANDATORY. NO EXCEPTIONS.**
