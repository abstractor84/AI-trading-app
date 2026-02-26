import os
import json
import logging
from google import genai
from services.backtester import VectorizedBacktester
import pandas as pd

from services.quota_service import QuotaService
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Singleton QuotaService
quota_svc = QuotaService()


class StrategyTuner:
    def __init__(self):
        self.google_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.samba_key = os.getenv("SAMBA_API_KEY")
        
        if self.google_key:
            self.google_client = genai.Client(api_key=self.google_key)
        else:
            self.google_client = None
            
    def optimize(self, ticker: str, df: pd.DataFrame, initial_params: dict, iterations: int = 3, provider: str = "google", model_name: str = "gemini-3.1-pro") -> dict:
        """
        Runs the backtester, feeds results to AI, and asks for parameter mutations 
        to maximize Sharpe/Win Rate in a self-healing loop.
        """
        current_params = initial_params
        best_result = None
        best_params = None
        
        history_log = []

        for i in range(iterations):
            logger.info(f"AI Optimizer Iteration {i+1} for {ticker} using {current_params} on {provider}/{model_name}")
            
            # 1. Run physical calculation
            bt = VectorizedBacktester(df)
            result = bt.run_strategy(current_params)
            
            # Stash reference SL for metrics internal use
            bt.params_reference_sl_pct = current_params.get('sl_pct', 0.01)
            # Re-run for absolute accuracy in equity loop
            result = bt.run_strategy(current_params)
            
            metrics = {
                k: v for k, v in result.items() if k not in ['trade_log', 'equity_curve']
            }
            
            history_log.append({
                "iteration": i + 1,
                "params": current_params,
                "metrics": metrics
            })
            
            # Keep track of best performer by net profit and MDD
            if best_result is None or (metrics['net_profit'] > best_result['net_profit'] and metrics['max_drawdown_pct'] < 15):
                best_result = metrics
                best_params = current_params
                
            # If this is the last iteration, break (we don't need to ask AI for more)
            if i == iterations - 1:
                break
                
            # NEW: Skip AI API call if 0 trades found (prevents waste of quota on data-less symbols)
            if metrics.get('total_trades', 0) == 0:
                logger.warning(f"0 trades found for {ticker} in iteration {i+1}. Skipping AI mutation to save quota.")
                # We can either stop or try one "blind" mutation
                current_params['sl_pct'] = round(current_params['sl_pct'] * 1.2, 4) # Widen SL as a blind guess
                continue
                
            # 2. Ask AI to evaluate and mutate parameters
            prompt = f"""
You are a Quantitative Trader optimizing an Intraday Momentum script for NSE Equity: {ticker}.
This is Iteration {i+1}. 
The current tested parameters are:
{json.dumps(current_params, indent=2)}

The engine backtested 3 months of 5-min intervals. The mathematical results:
{json.dumps(metrics, indent=2)}

Your Goal is to maximize Net Profit while keeping Max Drawdown under 10%.
If Win Rate is low (< 45%), consider slowing down EMA fast/slow, or adjusting RSI entry thresholds.
If Max Drawdown is high, consider tightening the SL_pct (e.g. from 0.01 to 0.005) or taking profit earlier.

Analyze why the current metrics are what they are. 
Then, provide a newly mutated set of parameters to test for Iteration {i+2}.

Expected JSON Output format strictly:
{{
    "analysis": "Brief 2 sentence thought on why the current params performed this way",
    "new_params": {{
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_len": 14,
        "rsi_buy_threshold": 40,
        "rsi_short_threshold": 60,
        "sl_pct": 0.008,
        "tp_pct": 0.015
    }}
}}
"""

            quota = quota_svc.check_quota(model_name)
            if not quota["can_call"]:
                logger.warning(f"Quota exceeded for {model_name}. Falling back to genetic mutation.")
                # Fallback to random genetic mutation if quota is out
                current_params['ema_fast'] = current_params['ema_fast'] + 1
                current_params['sl_pct'] = current_params['sl_pct'] * 0.9
                continue

            try:
                ai_feedback = None
                
                if provider == "google":
                    if not self.google_client:
                        raise ValueError("GEMINI_API_KEY missing")
                    response = self.google_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
                    )
                    tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                    quota_svc.log_usage(model_name, tokens=tokens)
                    ai_feedback = json.loads(response.text)
                    
                elif provider == "groq":
                    if not self.groq_key:
                        raise ValueError("GROQ_API_KEY missing")
                    headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": model_name, "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3, "response_format": {"type": "json_object"}
                    }
                    import requests
                    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
                    res.raise_for_status()
                    jdoc = res.json()
                    tokens = jdoc.get("usage", {}).get("total_tokens", 0)
                    quota_svc.log_usage(model_name, tokens=tokens)
                    content = jdoc["choices"][0]["message"]["content"]
                    if content.startswith("```json"): content = content.replace("```json\n", "").replace("```", "")
                    ai_feedback = json.loads(content)
                    
                elif provider == "sambanova":
                    if not self.samba_key:
                        raise ValueError("SAMBA_API_KEY missing")
                    headers = {"Authorization": f"Bearer {self.samba_key}", "Content-Type": "application/json"}
                    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
                    import requests
                    res = requests.post("https://api.sambanova.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
                    res.raise_for_status()
                    jdoc = res.json()
                    tokens = jdoc.get("usage", {}).get("total_tokens", 0)
                    quota_svc.log_usage(model_name, tokens=tokens)
                    content = jdoc["choices"][0]["message"]["content"]
                    if content.startswith("```json"): content = content.replace("```json\n", "").replace("```", "")
                    ai_feedback = json.loads(content)

                if ai_feedback:
                    logger.info(f"AI Tuning Analysis: {ai_feedback.get('analysis')}")
                    current_params = ai_feedback.get('new_params', current_params)
            except Exception as e:
                logger.error(f"AI Tuner failed to generate new params on {provider}: {e}")
                # Fallback to random genetic mutation if AI fails
                current_params['ema_fast'] = current_params['ema_fast'] + 1
                current_params['sl_pct'] = current_params['sl_pct'] * 0.9
                
        return {
            "best_metrics": best_result,
            "best_parameters": best_params,
            "history": history_log
        }
