import os
import json
import logging
from google import genai

logger = logging.getLogger(__name__)

class AIScorerService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.error("GEMINI_API_KEY is required for the AIScorerService!")

    def generate_recommendation(self, ticker: str, ta_data: dict, sentiment: dict, global_context: dict, capital: float, max_loss_per_trade: float, strategy: str = "s1", custom_rules: str = None) -> dict:
        """Generate trade recommendation using Gemini based on combined signals."""
        if not self.client:
            return self._fallback_response(ticker)

        strategy_context = f"Selected Strategy Profile: {strategy}"
        if custom_rules:
            strategy_context += f"\nCRITICAL CUSTOM RULES PROVIDED BY USER:\n{custom_rules}\n(You MUST evaluate these exact constraints against the TA data. If they are not met, score confidence lower or declare AVOID, explaining why the custom rules failed)."

        # Build prompt
        prompt = f"""
        You are an expert intraday equities trader on the Indian National Stock Exchange (NSE). 
        You are analyzing "{ticker}" for a day trade.
        
        {strategy_context}
        
        Here are the real-time 5-minute technical indicators:
        {json.dumps(ta_data, indent=2)}
        
        News Sentiment: {sentiment}
        
        Global/Macro Context (including India VIX):
        {json.dumps(global_context, indent=2)}
        
        My trading capital is ₹{capital} and my maximum accepted risk per trade is ₹{max_loss_per_trade}.
        
        Based on this, decide if I should BUY, SHORT SELL, WATCH, or AVOID right now.
        Consider volume surges, RSI divergence, MACD momentum, VWAP crossover, and Bollinger Band position.
        WARNING: If VIX is very high, reduce position scales. If VIX implies calm, proceed normally.
        Provide a numeric confidence score (0-100).
        If BUY or SHORT SELL, calculate strictly:
        - Exact Entry Price
        - Stop-Loss (tight for intraday, around 1-2% depending on volatility)
        - Target 1 (Risk:Reward 1:1.5 or 1:2 Minimum)
        - Target 2 (Risk:Reward 1:3)
        - Recommended Quantity (Derived safely so that if stop loss hits, I only lose ~ ₹{max_loss_per_trade}. Qty = Max_Loss / abs(Entry - SL))
        
        Provide the output EXCLUSIVELY as a strictly valid JSON object. No markdown formatting, no comments.
        
        Schema:
        {{
            "action": "BUY" | "SHORT SELL" | "WATCH" | "AVOID",
            "confidence_score": 0-100,
            "entry_price": float or null,
            "stop_loss": float or null,
            "target_1": float or null,
            "target_2": float or null,
            "recommended_quantity": int or null,
            "explanation": "Short summary of why this action was chosen based on indicators agreeing/disagreeing."
        }}
        """

        # Try models in order — stop on first success, skip on 404/429
        for model_name in ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash']:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                data = json.loads(response.text)
                data["ai_composite_score"] = data.get("confidence_score", 0)
                return data
            except Exception as e:
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    logger.warning(f"Gemini rate limit on {model_name}, trying next.")
                elif '404' in err_str or 'NOT_FOUND' in err_str:
                    logger.warning(f"Gemini model {model_name} not found, trying next.")
                else:
                    logger.error(f"Gemini {model_name} error for {ticker}: {e}")
                    break  # Non-retryable error

        logger.error(f"All Gemini models failed for {ticker}. Returning fallback.")
        return self._fallback_response(ticker)

    def _fallback_response(self, ticker):
        return {
            "action": "AVOID",
            "confidence_score": 0,
            "entry_price": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "recommended_quantity": 0,
            "explanation": "Failed to connect to AI engine.",
            "ai_composite_score": 0
        }
