"""
Market Phase State Machine
Determines the current NSE market session and provides phase-aware context.
"""
import logging
from datetime import datetime, time as dtime
from enum import Enum

logger = logging.getLogger(__name__)


class MarketPhase(str, Enum):
    PRE_MARKET = "PRE_MARKET"       # 08:00 - 09:15
    OPENING_15 = "OPENING_15"       # 09:15 - 09:30 (Opening Range)
    MID_SESSION = "MID_SESSION"     # 09:30 - 14:00
    POWER_HOUR = "POWER_HOUR"       # 14:00 - 15:00
    POST_MARKET = "POST_MARKET"     # 15:00 - 16:00
    CLOSED = "CLOSED"               # 16:00+ and before 08:00


# Phase boundaries (IST)
_PHASE_SCHEDULE = [
    (dtime(8, 0),   dtime(9, 15),  MarketPhase.PRE_MARKET),
    (dtime(9, 15),  dtime(9, 30),  MarketPhase.OPENING_15),
    (dtime(9, 30),  dtime(14, 0),  MarketPhase.MID_SESSION),
    (dtime(14, 0),  dtime(15, 0),  MarketPhase.POWER_HOUR),
    (dtime(15, 0),  dtime(16, 0),  MarketPhase.POST_MARKET),
]


class MarketPhaseService:
    """Tracks current market phase and provides phase-specific configuration."""

    def __init__(self):
        self._prev_phase = None

    def get_current_phase(self) -> MarketPhase:
        """Determine the current market phase from IST clock."""
        now = datetime.now().time()
        for start, end, phase in _PHASE_SCHEDULE:
            if start <= now < end:
                return phase
        return MarketPhase.CLOSED

    def get_phase_context(self) -> dict:
        """Return phase metadata for AI prompts and UI."""
        phase = self.get_current_phase()
        now = datetime.now()

        # Minutes remaining in current phase
        mins_left_in_phase = 0
        for start, end, p in _PHASE_SCHEDULE:
            if p == phase:
                end_dt = now.replace(hour=end.hour, minute=end.minute, second=0)
                mins_left_in_phase = max(0, int((end_dt - now).total_seconds() / 60))
                break

        # Minutes to market close (15:30)
        close_dt = now.replace(hour=15, minute=30, second=0)
        mins_to_close = max(0, int((close_dt - now).total_seconds() / 60))

        # Phase transition detection
        transitioned = False
        if self._prev_phase and self._prev_phase != phase:
            transitioned = True
            logger.info(f"Market phase transition: {self._prev_phase} → {phase}")
        self._prev_phase = phase

        return {
            "phase": phase.value,
            "phase_label": _PHASE_LABELS[phase],
            "mins_left_in_phase": mins_left_in_phase,
            "mins_to_close": mins_to_close,
            "is_trading_hours": phase in (
                MarketPhase.OPENING_15, MarketPhase.MID_SESSION, MarketPhase.POWER_HOUR
            ),
            "allow_new_entries": phase in (
                MarketPhase.MID_SESSION,  # Allow entries only in mid-session
            ),
            "should_review_positions": phase in (
                MarketPhase.POWER_HOUR, MarketPhase.POST_MARKET
            ),
            "transitioned": transitioned,
            "guidance": _PHASE_GUIDANCE[phase],
        }

    def get_ai_schedule(self) -> dict:
        """Return AI call schedule for the current phase."""
        phase = self.get_current_phase()
        return _AI_SCHEDULE.get(phase, {"call_interval_mins": 0, "prompt_type": None})


# Human-readable labels
_PHASE_LABELS = {
    MarketPhase.PRE_MARKET: "🌅 Pre-Market Analysis",
    MarketPhase.OPENING_15: "🔔 Opening Range (No Trades)",
    MarketPhase.MID_SESSION: "📊 Mid-Session Active",
    MarketPhase.POWER_HOUR: "⚡ Power Hour — Exit Focus",
    MarketPhase.POST_MARKET: "📋 Post-Market Review",
    MarketPhase.CLOSED: "🌙 Market Closed",
}

# Phase-specific user guidance messages
_PHASE_GUIDANCE = {
    MarketPhase.PRE_MARKET: (
        "Market opens at 9:15 AM. Review global cues, gap analysis, "
        "and yesterday's key levels. Do NOT place trades yet."
    ),
    MarketPhase.OPENING_15: (
        "Opening Range is forming (9:15–9:30). Let the first 15 minutes settle. "
        "Observe price action, volume, and opening range high/low. No entries recommended."
    ),
    MarketPhase.MID_SESSION: (
        "Active trading session. AI will analyze opportunities based on confirmed setups. "
        "Only enter trades that pass the Risk Engine validation."
    ),
    MarketPhase.POWER_HOUR: (
        "Last hour before close. Focus on managing open positions. "
        "Trail stop losses, book partial profits, or exit weak positions. "
        "New entries are high-risk at this stage."
    ),
    MarketPhase.POST_MARKET: (
        "Market has closed. Review today's trades, P&L, and lessons learned. "
        "Prepare watchlist and key levels for tomorrow."
    ),
    MarketPhase.CLOSED: (
        "Market is closed. Use this time to study charts, review AI suggestions, "
        "and plan for the next trading day."
    ),
}

# When to call AI and what type of prompt to use
_AI_SCHEDULE = {
    MarketPhase.PRE_MARKET: {
        "call_interval_mins": 0,  # Single call at phase start
        "prompt_type": "SCAN",
        "description": "Gap analysis and global cues review",
    },
    MarketPhase.OPENING_15: {
        "call_interval_mins": 0,  # No AI during opening range
        "prompt_type": None,
        "description": "Observe only — no AI calls",
    },
    MarketPhase.MID_SESSION: {
        "call_interval_mins": 30,  # Every 30 minutes
        "prompt_type": "SCAN",
        "description": "Periodic market scan and position review",
    },
    MarketPhase.POWER_HOUR: {
        "call_interval_mins": 15,  # More frequent during power hour
        "prompt_type": "EXIT_GUIDANCE",
        "description": "Exit guidance for open positions",
    },
    MarketPhase.POST_MARKET: {
        "call_interval_mins": 0,
        "prompt_type": None,
        "description": "Day summary generation",
    },
    MarketPhase.CLOSED: {
        "call_interval_mins": 0,
        "prompt_type": None,
        "description": "No AI calls",
    },
}


# Module-level singleton
market_phase_svc = MarketPhaseService()
