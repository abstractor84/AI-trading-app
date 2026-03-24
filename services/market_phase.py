"""
Market Phase State Machine
Determines the current NSE market session and provides granular phase-aware context.
"""
import logging
from datetime import datetime, time as dtime
from enum import Enum

logger = logging.getLogger(__name__)


class MarketPhase(str, Enum):
    PRE_MARKET_SETUP = "PRE_MARKET_SETUP"             # 08:00 - 09:15
    OPENING_VOLATILITY = "OPENING_VOLATILITY"         # 09:15 - 09:45
    MID_MORNING_TREND = "MID_MORNING_TREND"           # 09:45 - 11:30
    LUNCH_CHOP = "LUNCH_CHOP"                         # 11:30 - 13:30
    MID_SESSION = "MID_MORNING_TREND"                 # Alias for tests
    PM_BREAKOUT = "PM_BREAKOUT"                       # 13:30 - 14:30
    POWER_HOUR = "POWER_HOUR"                         # 14:30 - 15:30
    POST_MARKET_SETTLEMENT = "POST_MARKET_SETTLEMENT" # 15:30 - 16:30
    CLOSED = "CLOSED"                                 # 16:30 - 08:00


# Phase boundaries (IST)
_PHASE_SCHEDULE = [
    (dtime(8, 0),   dtime(9, 15),  MarketPhase.PRE_MARKET_SETUP),
    (dtime(9, 15),  dtime(9, 45),  MarketPhase.OPENING_VOLATILITY),
    (dtime(9, 45),  dtime(11, 30), MarketPhase.MID_MORNING_TREND),
    (dtime(11, 30), dtime(13, 30), MarketPhase.LUNCH_CHOP),
    (dtime(13, 30), dtime(14, 30), MarketPhase.PM_BREAKOUT),
    (dtime(14, 30), dtime(15, 30), MarketPhase.POWER_HOUR),
    (dtime(15, 30), dtime(16, 30), MarketPhase.POST_MARKET_SETTLEMENT),
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
            "phase_label": _PHASE_LABELS.get(phase, "Unknown Phase"),
            "mins_left_in_phase": mins_left_in_phase,
            "mins_to_close": mins_to_close,
            "is_trading_hours": phase in (
                MarketPhase.OPENING_VOLATILITY, MarketPhase.MID_MORNING_TREND, 
                MarketPhase.LUNCH_CHOP, MarketPhase.PM_BREAKOUT, MarketPhase.POWER_HOUR
            ),
            "allow_new_entries": phase in (
                MarketPhase.MID_MORNING_TREND, MarketPhase.PM_BREAKOUT
            ),
            "should_review_positions": phase in (
                MarketPhase.LUNCH_CHOP, MarketPhase.POWER_HOUR, MarketPhase.POST_MARKET_SETTLEMENT
            ),
            "transitioned": transitioned,
            "guidance": _PHASE_GUIDANCE.get(phase, ""),
        }

    def get_ai_schedule(self) -> dict:
        """Return AI call schedule for the current phase."""
        phase = self.get_current_phase()
        return _AI_SCHEDULE.get(phase, {"call_interval_mins": 0, "prompt_type": None})


# Human-readable labels
_PHASE_LABELS = {
    MarketPhase.PRE_MARKET_SETUP: "🌅 Pre-Market Setup",
    MarketPhase.OPENING_VOLATILITY: "🔔 Opening Volatility (Watch Only)",
    MarketPhase.MID_MORNING_TREND: "📈 Mid-Morning Trend",
    MarketPhase.LUNCH_CHOP: "🍔 Lunch Chop (Avoid)",
    MarketPhase.PM_BREAKOUT: "🚀 PM Breakout",
    MarketPhase.POWER_HOUR: "⚡ Power Hour (Exit Focus)",
    MarketPhase.POST_MARKET_SETTLEMENT: "📋 Post-Market Settlement",
    MarketPhase.CLOSED: "🌙 Market Closed",
}

# Phase-specific user guidance messages
_PHASE_GUIDANCE = {
    MarketPhase.PRE_MARKET_SETUP: (
        "Market opens at 9:15 AM. Review global cues, gap analysis, "
        "and yesterday's key levels. Do NOT place trades yet."
    ),
    MarketPhase.OPENING_VOLATILITY: (
        "High volatility in the first 30 mins (9:15–9:45). Let the range settle. "
        "Observe price action and volume. No entries recommended."
    ),
    MarketPhase.MID_MORNING_TREND: (
        "Primary trend formation (9:45-11:30). High probability setups occur here. "
        "AI will scan for solid directional trades."
    ),
    MarketPhase.LUNCH_CHOP: (
        "Low volume chop phase (11:30-13:30). Institutional activity is low. "
        "Avoid new entries. Manage existing setups strictly."
    ),
    MarketPhase.PM_BREAKOUT: (
        "Afternoon momentum returns (13:30-14:30). Look for breakouts or trend continuation. "
    ),
    MarketPhase.POWER_HOUR: (
        "Last hour before close (14:30-15:30). Focus on managing open positions. "
        "Trail stop losses, book partial profits, or exit weak positions. "
        "New entries are high-risk at this stage."
    ),
    MarketPhase.POST_MARKET_SETTLEMENT: (
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
    MarketPhase.PRE_MARKET_SETUP: {
        "call_interval_mins": 30,
        "prompt_type": "SCAN",
        "description": "Gap analysis and global cues review",
    },
    MarketPhase.OPENING_VOLATILITY: {
        "call_interval_mins": 0,  
        "prompt_type": None,
        "description": "Observe only — no AI calls",
    },
    MarketPhase.MID_MORNING_TREND: {
        "call_interval_mins": 15,  
        "prompt_type": "SCAN",
        "description": "Frequent scans during prime trending hours",
    },
    MarketPhase.LUNCH_CHOP: {
        "call_interval_mins": 45,  
        "prompt_type": "POSITION_REVIEW",
        "description": "Infrequent scanning, mostly position management",
    },
    MarketPhase.PM_BREAKOUT: {
        "call_interval_mins": 15,  
        "prompt_type": "SCAN",
        "description": "Frequent scans during afternoon breakout window",
    },
    MarketPhase.POWER_HOUR: {
        "call_interval_mins": 15,  
        "prompt_type": "EXIT_GUIDANCE",
        "description": "Exit guidance for open positions",
    },
    MarketPhase.POST_MARKET_SETTLEMENT: {
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

# Global Singleton
market_phase_svc = MarketPhaseService()
