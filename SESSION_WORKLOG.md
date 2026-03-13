# SuperNova V3: Architectural Audit & Skeptic Review (Update 9)

## Session Intent
Finalize production-grade stability, fix persistent "Close Position" and "Sentinel" regressions, and achieve 100% UI/Backend parity.

## 1. Skeptic Code Review (Evidence of Failure)
- **RCA - Sentinel Crash (`KeyError: 'link'`):** The traceback shows `KeyError: 'link'` on a line where I used `.get('url', '')`. This indicates a **cache/restart mismatch** or that `primp/DDGS` is throwing an error from *within* its own dictionary access. 
- **RCA - Close Position Failure:** The `str()` comparison in `state.py` was correct, but if the UI is not reflecting it, then `self.open_trades` is likely being **clobbered** by an asynchronous `_load_from_db()` call or a background update that hasn't cleared the cache.
- **RCA - Market Data Imprecision:** 
    - **Brent/WTI:** Added but not appearing because the fetch loop in `stock_discovery.py` was still using the old hardcoded Indian symbols list.
    - **GIFT NIFTY:** Yahoo Finance's `^NSEI` is a good proxy, but I need to ensure it's labeled correctly.
- **UI Logic Flaw:** 
    - Market Data rows are not sticky enough (needs `position: sticky` on the parent).
    - USDINR, Crude, Gold, Silver need to be forced to the **leftmost** position in the ग्लोबल row.

## 2. Active Task List
- [ ] **Surgical fix for Sentinel:** Use a completely safe dictionary extractor.
- [ ] **Fix Close Trade Persistence:** Add debug logs to `close_trade` and ensure DB commit is finalized before broadcast.
- [ ] **Market Data Ordering:** Force USDINR, Crude, GOLD, Silver to index 0-3.
- [ ] **Fix Chart Opening:** Debug `app.js` modal triggers.
- [ ] **Run Full Test Suite:** Simulation mode ON.
- [ ] **Cleanup:** Wipe test data.

## 3. Final Status (Policy Check)
- [ ] Skeptic Review: IN PROGRESS.
- [ ] Architecture Audit: IN PROGRESS.
- [ ] UI Functionality: PENDING.
