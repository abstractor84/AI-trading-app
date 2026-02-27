"""
V2 AI Advisor Service
=====================
Surgical, context-aware AI prompts that respect:
- Market phase (time-of-day)
- Open positions (position-aware guidance)
- Time-to-close (urgency)
- Risk Engine math (validated, not hallucinated)

3 Prompt Types:
  1. SCAN         — "Which of these stocks are worth trading right now?"
  2. POSITION     — "I hold X. What should I do with it?"
  3. EXIT         — "Power hour. Should I close before 3 PM?"

Max 7 AI calls per day. All SL/Target math done by RiskEngine, NOT AI.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from google import genai
from dotenv import load_dotenv

from database import SessionLocal
from models import AIInteraction
from services.quota_service import QuotaService

load_dotenv()
logger = logging.getLogger(__name__)
quota_svc = QuotaService()


class AIAdvisorService:
    """
    V2 AI service with 3 distinct prompt types and strict quota management.
    AI provides JUDGMENT and SYNTHESIS only — all math comes from RiskEngine.
    """

    def __init__(self):
        self.google_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if self.google_key:
            self.google_client = genai.Client(api_key=self.google_key)

    # ─── Prompt Type 1: SCAN ────────────────────────────────────────
    def scan_market(self, candidates: list[dict], global_context: dict,
                    phase_ctx: dict, provider: str = "google",
                    model_name: str = "gemini-2.5-flash") -> dict:
        """
        Batch-analyze multiple stocks in ONE AI call.
        Returns top 1-3 actionable picks with reasoning.

        Each candidate = {ticker, ta_data, pivots, atr}
        """
        now = datetime.now()

        # Build compact TA summaries (save tokens)
        stock_summaries = []
        for c in candidates[:8]:  # Cap at 8 stocks per batch
            ta = c.get("ta_data", {})
            m_prob = float(c.get('math_prob', 0.0))
            # Basic sanity guard
            if not ta: continue
            
            summary = (
                f"[{c['ticker']}] - Math Probability: {m_prob:.2f}/1.0\n"
                f"Price: {ta.get('close', 0):.2f} | VWAP: {ta.get('vwap', 0):.2f}\n"
                f"RSI={ta.get('rsi_14', 0):.1f}, "
                f"EMA9={'>' if ta.get('ema_9', 0) > ta.get('ema_21', 0) else '<'}EMA21, "
                f"MACD_Hist={ta.get('macd_hist', 0):.2f}, "
                f"ADX={ta.get('adx_14', 0):.1f}, "
                f"VolSurge={ta.get('vol_surge', 0):.1f}x, "
                f"ATR=₹{c.get('atr', 0):.2f}"
            )
            stock_summaries.append(summary)

        stocks_block = "\n".join(stock_summaries)

        # Extract clean index context
        nifty = global_context.get("india", {}).get("NIFTY 50", {})
        vix = global_context.get("vix", {})

        prompt = f"""You are a DISCIPLINED NSE INTRADAY TRADING ADVISOR.

CURRENT TIME: {now.strftime("%H:%M IST")}
MARKET PHASE: {phase_ctx.get('phase_label', 'Unknown')}
TIME TO CLOSE: {phase_ctx.get('mins_to_close', 0)} minutes

MARKET CONTEXT:
  Nifty 50: ₹{nifty.get('value', 0)} ({nifty.get('change_pct', 0):+.2f}%)
  India VIX: {vix.get('value', 0)} ({vix.get('change_pct', 0):+.2f}%)

STOCKS WITH REAL-TIME TECHNICALS (HIGH MATH PROBABILITY HIGHLIGHTED):
{stocks_block}

TASK: From the highly-mathematically probable stocks above, identify AT MOST 2 stocks that have the ABSOLUTE BEST PROBABILITY intraday setup RIGHT NOW.

RULES (BATTLE-HARDENED):
1. THE MATH MODEL HAS ALREADY VETTED THESE: These stocks arrived to you because a pure mathematical algorithm calculated their Math Probability as > 0.50 based on trend and momentum alignments.
2. ACTION BIAS: Since the math model already approved these setups, if the Math Probability is >= 0.50 and the trend aligns (EMA9 > EMA21 for BUY, EMA9 < EMA21 for SHORT), you MUST recommend a trade unless there is overwhelming, contradictory evidence (like RSI being catastrophically overextended).
3. TREND (MANDATORY): BUY only if EMA9 > EMA21. SHORT SELL only if EMA9 < EMA21.
4. RSI GUARD: No BUY if RSI > 75 (Overbought). No SHORT if RSI < 25 (Oversold).
5. MOMENTUM: ADX > 20 is preferred for trend-following.
6. TIME DECAY: If < 60 min to close (14:30 IST), DO NOT recommend new trades.
7. MATH IS LAW: You simply agree with the direction; the Risk Engine WILL set the precise Stop Loss and Target calculations based on ATR.

OUTPUT: Strictly valid JSON array:
[
  {{
    "ticker": "STOCK.NS",
    "action": "BUY" | "SHORT SELL",
    "confidence": 70-100, 
    "reasoning": "[Math Score: X.XX] - Surgical 2-line TA evidence explaining why this trade works.",
    "valid_for_minutes": 10-15
  }}
]
If NO high-conviction trades exist despite the high math probability, return: []
"""

        return self._call_ai(prompt, "SCAN", provider, model_name,
                             input_summary=f"Scanned {len(candidates)} stocks in {phase_ctx.get('phase', 'UNKNOWN')}")

    # ─── Prompt Type 2: POSITION REVIEW ─────────────────────────────
    def review_positions(self, open_trades: list[dict], global_context: dict,
                         phase_ctx: dict, provider: str = "google",
                         model_name: str = "gemini-2.5-flash") -> dict:
        """
        Review all open positions and advise on each.
        Returns per-position guidance: HOLD, TRAIL, PARTIAL BOOK, EXIT.
        """
        if not open_trades:
            return {"advice": [], "summary": "No open positions."}

        pos_summaries = []
        for t in open_trades:
            risk_advice = t.get("risk_advice", {})
            pos_summaries.append(
                f"• {t['ticker']} {t['action']} ×{t['quantity']} @ ₹{t['entry_price']:.2f}\n"
                f"  Current: ₹{t.get('current_price', 0):.2f} | "
                f"P&L: ₹{t.get('pnl', 0):.2f} | "
                f"SL: ₹{t['stop_loss']:.2f} | Trail SL: ₹{t.get('trailing_sl', 0):.2f}\n"
                f"  Risk Engine says: {risk_advice.get('advice', 'N/A')} — {risk_advice.get('reason', '')}"
            )

        positions_block = "\n".join(pos_summaries)

        nifty = global_context.get("india", {}).get("NIFTY 50", {})
        vix = global_context.get("vix", {})

        prompt = f"""You are an INTRADAY POSITION MANAGER for NSE equities.

CURRENT TIME: {datetime.now().strftime("%H:%M IST")}
MARKET PHASE: {phase_ctx.get('phase_label', 'Unknown')}
TIME TO CLOSE: {phase_ctx.get('mins_to_close', 0)} minutes
NIFTY: ₹{nifty.get('value', 0)} ({nifty.get('change_pct', 0):+.2f}%)
VIX: {vix.get('value', 0)}

OPEN POSITIONS:
{positions_block}

TASK: For EACH position, provide actionable advice.

RULES (CAPITAL PRESERVATION FIRST):
1. RISK ENGINE IS FINAL: If Risk Engine says 'EXIT', YOU MUST AGREE. Do not hope for a reversal.
2. PROFIT TAKING: If unrealized P&L > 1% of trade value, recommend BOOKING 50% immediately, regardless of targets.
3. TIME URGENCY: If < 45 min to close, recommend EXIT for ALL positions unless they are at T2.
4. VIX SPIKE: If VIX > 22 or has jumped > 5% today, increase urgency to 'HIGH' for all exits.
5. NEVER AVERAGE DOWN: Under no circumstances suggest adding to a losing position.

OUTPUT: Strictly valid JSON array:
[
  {{
    "ticker": "STOCK.NS",
    "action": "HOLD" | "TRAIL SL" | "BOOK 50%" | "EXIT",
    "urgency": "LOW" | "MEDIUM" | "HIGH",
    "reasoning": "Brief technical justification"
  }}
]
"""

        return self._call_ai(prompt, "POSITION_REVIEW", provider, model_name,
                             input_summary=f"Reviewed {len(open_trades)} open positions")

    # ─── Prompt Type 3: EXIT GUIDANCE ───────────────────────────────
    def exit_guidance(self, open_trades: list[dict], global_context: dict,
                      phase_ctx: dict, provider: str = "google",
                      model_name: str = "gemini-2.5-flash") -> dict:
        """
        Power Hour / Close-of-day exit advice.
        Specifically focuses on: should user close everything now?
        """
        if not open_trades:
            return {"advice": "No open positions to exit.", "should_close_all": False}

        total_pnl = sum(t.get("pnl", 0) for t in open_trades)

        pos_lines = []
        for t in open_trades:
            pos_lines.append(
                f"• {t['ticker']} {t['action']} P&L: ₹{t.get('pnl', 0):.2f}"
            )

        nifty = global_context.get("india", {}).get("NIFTY 50", {})

        prompt = f"""You are a RISK-FIRST EXIT ADVISOR for NSE intraday trading.

TIME: {datetime.now().strftime("%H:%M IST")}
PHASE: {phase_ctx.get('phase_label', '')}
MINUTES TO CLOSE: {phase_ctx.get('mins_to_close', 0)}

TOTAL UNREALIZED P&L: ₹{total_pnl:.2f}
NIFTY TREND: {nifty.get('change_pct', 0):+.2f}%

OPEN POSITIONS:
{chr(10).join(pos_lines)}

QUESTION: Should the user close ALL positions before market close?

RULES:
1. If total P&L is positive → recommend booking profits (don't be greedy before close).
2. If total P&L is negative and < 30 min left → recommend cutting losses.
3. If positions are trending strongly in the user's favor → can hold with tight trail.
4. NEVER recommend holding losing positions into close.

OUTPUT: Strictly valid JSON:
{{
  "should_close_all": true | false,
  "reasoning": "2-3 lines",
  "per_position": [
    {{"ticker": "X.NS", "action": "CLOSE" | "HOLD WITH TRAIL", "reason": "..."}}
  ]
}}
"""

        return self._call_ai(prompt, "EXIT_GUIDANCE", provider, model_name,
                             input_summary=f"Exit guidance for {len(open_trades)} positions, total P&L ₹{total_pnl:.0f}")

    # ─── Core AI Call (shared) ──────────────────────────────────────
    def _call_ai(self, prompt: str, prompt_type: str, provider: str,
                 model_name: str, input_summary: str = "") -> dict:
        """
        Execute AI call with quota check and audit logging.
        """
        # Quota gate
        quota = quota_svc.check_quota(model_name)
        if not quota["can_call"]:
            logger.warning(f"Quota exceeded for {model_name}")
            return {"error": f"AI quota exceeded for {model_name}. Try again later."}

        try:
            if provider == "google":
                result = self._call_google(model_name, prompt)
            elif provider == "groq":
                result = self._call_groq(model_name, prompt)
            elif provider == "sambanova":
                result = self._call_sambanova(model_name, prompt)
            else:
                return {"error": f"Unknown provider: {provider}"}

            # Audit log
            self._log_interaction(prompt_type, model_name, input_summary, result)
            
            # Update local quota tracker
            quota_svc.log_usage(model_name)

            return result

        except Exception as e:
            if "429" in str(e) and model_name != "gemini-2.5-flash":
                logger.warning(f"AI Model {model_name} rate limited (429). Falling back to gemini-2.5-flash...")
                try:
                    if provider == "google":
                        result = self._call_google("gemini-2.5-flash", prompt)
                    elif provider == "sambanova":
                        result = self._call_sambanova("Meta-Llama-3.1-8B-Instruct", prompt) # Known default for sambanova
                    else:
                        result = self._call_groq(model_name, prompt) # Keep groq for now
                    quota_svc.log_usage("gemini-2.5-flash")
                    return result
                except Exception as ef:
                    logger.error(f"AI Fallback also failed: {ef}")
                    return {"error": str(ef)}
            
            logger.error(f"AI call failed ({provider}/{model_name}): {e}")
            return {"error": str(e)}

    def _call_google(self, model_name: str, prompt: str) -> dict:
        """Call Google Gemini API."""
        if not self.google_key:
            return {"error": "GEMINI_API_KEY not set"}

        response = self.google_client.models.generate_content(
            model=model_name, contents=prompt
        )
        return self._parse_json_response(response.text)

    def _call_groq(self, model_name: str, prompt: str) -> dict:
        """Call Groq API."""
        import requests
        if not self.groq_key:
            return {"error": "GROQ_API_KEY not set"}

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.groq_key}",
                     "Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            return self._parse_json_response(text)
        return {"error": f"Groq returned {resp.status_code}"}

    def _call_sambanova(self, model_name: str, prompt: str) -> dict:
        """Call SambaNova API."""
        import requests
        sambanova_key = os.getenv("SAMBA_API_KEY")
        if not sambanova_key:
            return {"error": "SAMBA_API_KEY not set"}

        headers = {
            "Authorization": f"Bearer {sambanova_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        res = requests.post("https://api.sambanova.ai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            result = self._parse_json_response(text)
            logger.info(f"Parsed AI result: {result}")
            return result
        logger.error(f"SambaNova returned {res.status_code}: {res.text}")
        return {"error": f"SambaNova returned {res.status_code}: {res.text}"}

    def _parse_json_response(self, text: str) -> dict:
        """Clean and parse AI JSON response."""
        logger.info(f"--- RAW AI OUTPUT START ---\n{text}\n--- RAW AI OUTPUT END ---")
        
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            else:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
        else:
            # Fallback for conversational wrappers
            if "[" in cleaned and "]" in cleaned:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
                try:
                    js = json.loads(cleaned[start:end])
                    if isinstance(js, list): return js
                except Exception:
                    pass
                    
            if "{" in cleaned and "}" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                try:
                    js = json.loads(cleaned[start:end])
                    return js if isinstance(js, (dict, list)) else {"data": js}
                except Exception:
                    pass

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, (dict, list)) else {"data": parsed}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse AI response as JSON: {cleaned[:200]}")
            return {"raw_response": cleaned, "parse_error": True}

    def _log_interaction(self, prompt_type: str, model_used: str,
                         input_summary: str, output: dict):
        """Audit log to ai_interactions table."""
        try:
            with SessionLocal() as db:
                interaction = AIInteraction(
                    prompt_type=prompt_type,
                    model_used=model_used,
                    input_summary=input_summary,
                    output_json=json.dumps(output, default=str)[:5000],
                    trade_date=datetime.now().strftime("%Y-%m-%d"),
                )
                db.add(interaction)
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to log AI interaction: {e}")


# Module-level singleton
ai_advisor = AIAdvisorService()
