"""
Deterministic Risk Engine
All SL, Target, Position Sizing, and Risk:Reward math happens here.
No AI involved — pure mathematics.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Enforces disciplined risk management using proven mathematical models.
    Every trade recommendation must pass through this engine before it
    reaches the user.
    """

    def __init__(self, capital: float = 100000.0, max_risk_per_trade: float = 1000.0,
                 max_daily_loss: float = 5000.0):
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0  # Updated externally as trades close

    def update_config(self, capital: float, max_risk_per_trade: float,
                      max_daily_loss: float = None):
        """Update risk parameters from user settings."""
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        if max_daily_loss is not None:
            self.max_daily_loss = max_daily_loss

    def reset_daily(self):
        """Reset daily P&L at market open."""
        self.daily_pnl = 0.0

    def record_closed_pnl(self, pnl: float):
        """Record a closed trade's P&L for daily loss tracking."""
        self.daily_pnl += pnl

    # ─── Core Calculations ──────────────────────────────────────────────

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Compute Average True Range from OHLCV DataFrame.
        ATR measures volatility and is the basis for all SL/Target math.
        """
        if df is None or len(df) < period + 1:
            return 0.0

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Simple Moving Average of True Range
        if len(tr) < period:
            return float(np.mean(tr))
        return float(np.mean(tr[-period:]))

    def compute_pivots(self, high: float, low: float, close: float) -> dict:
        """
        Standard Pivot Points from yesterday's HLC.
        Returns pivot, support (S1-S3), and resistance (R1-R3) levels.
        """
        pivot = (high + low + close) / 3

        return {
            "pivot": round(pivot, 2),
            "r1": round(2 * pivot - low, 2),
            "r2": round(pivot + (high - low), 2),
            "r3": round(high + 2 * (pivot - low), 2),
            "s1": round(2 * pivot - high, 2),
            "s2": round(pivot - (high - low), 2),
            "s3": round(low - 2 * (high - pivot), 2),
        }

    def compute_fibonacci_pivots(self, high: float, low: float, close: float) -> dict:
        """Fibonacci Pivot Points for more nuanced S/R levels."""
        pivot = (high + low + close) / 3
        diff = high - low

        return {
            "pivot": round(pivot, 2),
            "r1": round(pivot + 0.382 * diff, 2),
            "r2": round(pivot + 0.618 * diff, 2),
            "r3": round(pivot + 1.000 * diff, 2),
            "s1": round(pivot - 0.382 * diff, 2),
            "s2": round(pivot - 0.618 * diff, 2),
            "s3": round(pivot - 1.000 * diff, 2),
        }

    # ─── SL / Target / Sizing ───────────────────────────────────────────

    def compute_sl_target(self, entry_price: float, action: str, atr: float,
                          sl_multiplier: float = 1.5, t1_multiplier: float = 2.0,
                          t2_multiplier: float = 3.0) -> dict:
        """
        Compute mathematically validated SL and Targets using ATR.

        - SL = entry ± (sl_multiplier × ATR)
        - T1 = entry ± (t1_multiplier × ATR)  [minimum 1:1 R:R]
        - T2 = entry ± (t2_multiplier × ATR)  [stretch target]
        """
        if atr <= 0:
            logger.warning("ATR is zero or negative; cannot compute SL/Target")
            return None

        # Sanitize action
        action = action.upper().strip()
        if action in ["SELL", "SHORT"]: action = "SHORT SELL"

        atr_sl = atr * sl_multiplier
        atr_t1 = atr * t1_multiplier
        atr_t2 = atr * t2_multiplier

        if action == "BUY":
            sl = entry_price - atr_sl
            t1 = entry_price + atr_t1
            t2 = entry_price + atr_t2
            # Safety Check: SL must be below Entry
            if sl > entry_price: sl, t1 = t1, sl # Flip if somehow inverted
        elif action == "SHORT SELL":
            sl = entry_price + atr_sl
            t1 = entry_price - atr_t1
            t2 = entry_price - atr_t2
            # Safety Check: SL must be above Entry
            if sl < entry_price: sl, t1 = t1, sl
        else:
            logger.warning(f"Unsupported action for RiskEngine: {action}")
            return None

        return {
            "stop_loss": round(sl, 2),
            "target_1": round(t1, 2),
            "target_2": round(t2, 2),
            "risk_per_share": round(atr_sl, 2),
            "reward_t1_per_share": round(atr_t1, 2),
            "rr_ratio": round(atr_t1 / atr_sl, 2) if atr_sl > 0 else 0,
        }

    def compute_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Position sizing based on fixed-risk model.
        Qty = MaxRiskPerTrade / Risk-per-share, capped by available capital.
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        qty_by_risk = int(self.max_risk_per_trade / risk_per_share)
        qty_by_capital = int(self.capital / entry_price)  # Can't buy more than capital allows

        return max(1, min(qty_by_risk, qty_by_capital))

    def compute_trailing_sl(self, entry_price: float, current_price: float,
                            action: str, atr: float, trail_factor: float = 1.0) -> float:
        """
        Dynamic trailing SL based on how far price has moved in favor.

        Rules:
        - If unrealized profit > 1×ATR: trail SL to breakeven
        - If unrealized profit > 2×ATR: trail SL to entry + 1×ATR
        - Always trails, never retreats
        """
        if action == "BUY":
            move = current_price - entry_price
            if move >= 2 * atr:
                return round(entry_price + atr * trail_factor, 2)
            elif move >= atr:
                return round(entry_price, 2)  # Breakeven
            else:
                return round(entry_price - atr * 1.5, 2)  # Original SL
        elif action == "SHORT SELL":
            move = entry_price - current_price
            if move >= 2 * atr:
                return round(entry_price - atr * trail_factor, 2)
            elif move >= atr:
                return round(entry_price, 2)
            else:
                return round(entry_price + atr * 1.5, 2)
        return entry_price

    # ─── Validation Gates ───────────────────────────────────────────────

    def validate_trade(self, entry_price: float, action: str, atr: float,
                       current_price: float = None, ta_data: dict = None,
                       vix_value: float = None) -> dict:
        """
        Hard-coded 'Anti-Loss' validation pipeline.
        Multi-layer safety gates that AI CANNOT override.
        """
        reasons = []
        passed = True

        # Gate 1: Daily loss limit (Circuit Breaker)
        # Block if daily loss > max_daily_loss or > 2% of capital
        drawdown_limit = min(self.max_daily_loss, self.capital * 0.02)
        if self.daily_pnl <= -drawdown_limit:
            reasons.append(f"🔴 CRITICAL: Daily drawdown limit reached (₹{self.daily_pnl:.0f} / ₹{drawdown_limit:.0f}). Trading halted.")
            passed = False

        # Gate 2: Volatility Check (VIX Guard)
        if vix_value and vix_value > 25:
            reasons.append(f"⚠️ HIGH RISK: India VIX is {vix_value:.1f} (> 25). Market too unstable for safe trading.")
            passed = False

        # Gate 3: ATR Validity & Relative Volatility
        if atr <= 0:
            reasons.append("❌ FAIL: ATR is zero — insufficient volatility data.")
            passed = False
        else:
            atr_pct = (atr / entry_price) * 100
            if atr_pct > 5.0:
                reasons.append(f"⚠️ HIGH RISK: Volatility (ATR) is {atr_pct:.1f}% of price. Stock is too 'whippy'.")
                passed = False

        # Gate 4: Trend Alignment (Math > AI)
        if ta_data:
            ema9 = ta_data.get("ema_9")
            ema21 = ta_data.get("ema_21")
            if ema9 and ema21:
                if action == "BUY" and ema9 < ema21:
                    reasons.append("❌ TREND MISMATCH: Cannot BUY when EMA 9 < EMA 21 (Short-term trend is Down).")
                    passed = False
                elif action == "SHORT SELL" and ema9 > ema21:
                    reasons.append("❌ TREND MISMATCH: Cannot SHORT when EMA 9 > EMA 21 (Short-term trend is Up).")
                    passed = False

        # Gate 5: Compute and validate SL/Target
        levels = self.compute_sl_target(entry_price, action, atr)
        if levels is None:
            reasons.append("❌ FAIL: Could not compute SL/Target levels.")
            passed = False
        elif levels["rr_ratio"] < 1.25: # Raised minimum RR from 1.0 to 1.25
            reasons.append(f"⚠️ POOR R:R: Risk:Reward ratio is {levels['rr_ratio']}:1 (minimum 1.25:1 required).")
            passed = False

        # Gate 6: Entry price slippage/sanity
        if current_price is not None:
            drift = abs(current_price - entry_price) / entry_price * 100
            if drift > 0.5:  # Tightened from 1.0% to 0.5%
                reasons.append(f"⚠️ SLIPPAGE: Entry ₹{entry_price:.2f} has drifted {drift:.2f}% from live ₹{current_price:.2f}.")
                if drift > 1.0:
                    passed = False # Hard block on > 1% drift

        # Compute position size if passed
        qty = 0
        if passed and levels:
            qty = self.compute_position_size(entry_price, levels["stop_loss"])
            
        logger.info(f"== RISK ENGINE EVALUATION ==")
        logger.info(f"Targeting: {action} @ ₹{entry_price:.2f} | ATR: {atr:.2f}")
        for res in reasons:
            logger.info(f"  {res}")
        if passed:
            logger.info(f"  ✅ PASSED. Qty: {qty} | Max Risk: ₹{round(qty * levels['risk_per_share'], 2):.2f}")
            logger.info(f"  SL: ₹{levels['stop_loss']:.2f} | T1: ₹{levels['target_1']:.2f} | T2: ₹{levels['target_2']:.2f}")
        else:
            logger.info(f"  ❌ FAILED Validation.")
        logger.info(f"============================")

        return {
            "passed": passed,
            "reasons": reasons,
            "levels": levels,
            "quantity": qty,
            "max_loss_this_trade": round(qty * levels["risk_per_share"], 2) if levels else 0,
        }

    def get_position_action(self, trade: dict, current_price: float,
                            atr: float, mins_to_close: int) -> dict:
        """
        For an open position, determine what the user should do.
        Returns a mathematical recommendation (not AI — pure rules).
        """
        entry = trade["entry_price"]
        action = trade["action"]
        sl = trade.get("stop_loss", entry)
        t1 = trade.get("target_1", entry)
        t2 = trade.get("target_2", entry)

        if action == "BUY":
            pnl_per_share = current_price - entry
        else:
            pnl_per_share = entry - current_price

        pnl_total = pnl_per_share * trade["quantity"]
        trail_sl = self.compute_trailing_sl(entry, current_price, action, atr)

        # Decision rules
        advice = "HOLD"
        reason = ""

        # Rule 1: SL hit
        if action == "BUY" and current_price <= sl:
            advice = "EXIT — SL HIT"
            reason = f"Price ₹{current_price:.2f} has breached SL ₹{sl:.2f}"
        elif action == "SHORT SELL" and current_price >= sl:
            advice = "EXIT — SL HIT"
            reason = f"Price ₹{current_price:.2f} has breached SL ₹{sl:.2f}"

        # Rule 2: T2 hit — book full
        elif (action == "BUY" and current_price >= t2) or \
             (action == "SHORT SELL" and current_price <= t2):
            advice = "EXIT — T2 HIT"
            reason = f"Target 2 (₹{t2:.2f}) reached. Book full profit."

        # Rule 3: T1 hit — trail SL to breakeven
        elif (action == "BUY" and current_price >= t1) or \
             (action == "SHORT SELL" and current_price <= t1):
            advice = "TRAIL SL → Breakeven"
            reason = f"T1 (₹{t1:.2f}) hit. Move SL to ₹{entry:.2f} (breakeven). Hold for T2."

        # Rule 4: Power hour — close if in loss
        elif mins_to_close <= 60 and pnl_per_share < 0:
            advice = "EXIT — Power Hour Loss"
            reason = f"In loss (₹{pnl_total:.0f}) with {mins_to_close}m left. Cut losses."

        # Rule 5: Close if < 15 min left
        elif mins_to_close <= 15:
            advice = "EXIT — Day Close"
            reason = f"Only {mins_to_close}m to close. Exit to avoid overnight risk."

        # Default: Hold with trailing SL
        else:
            advice = "HOLD"
            reason = f"Position is within range. Trail SL to ₹{trail_sl:.2f}"

        return {
            "advice": advice,
            "reason": reason,
            "trailing_sl": trail_sl,
            "pnl_per_share": round(pnl_per_share, 2),
            "pnl_total": round(pnl_total, 2),
            "atr": round(atr, 2),
        }


# Module-level singleton
risk_engine = RiskEngine()
