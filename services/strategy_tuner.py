import os
import json
import logging
from google import genai
from services.backtester import VectorizedBacktester
import pandas as pd

logger = logging.getLogger(__name__)

class StrategyTuner:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
    def optimize(self, ticker: str, df: pd.DataFrame, initial_params: dict, iterations: int = 3) -> dict:
        """
        Runs the backtester, feeds results to AI, and asks for parameter mutations 
        to maximize Sharpe/Win Rate in a self-healing loop.
        """
        current_params = initial_params
        best_result = None
        best_params = None
        
        history_log = []

        for i in range(iterations):
            logger.info(f"AI Optimizer Iteration {i+1} for {ticker} using {current_params}")
            
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
            try:
                # Use Gemini 3.0 Pro for complex reasoning
                response = self.client.models.generate_content(
                    model='gemini-3.0-pro',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3
                    )
                )
                ai_feedback = json.loads(response.text)
                logger.info(f"AI Tuning Analysis: {ai_feedback.get('analysis')}")
                current_params = ai_feedback.get('new_params', current_params)
            except Exception as e:
                logger.error(f"AI Tuner failed to generate new params: {e}")
                # Fallback to random genetic mutation if AI fails
                current_params['ema_fast'] = current_params['ema_fast'] + 1
                current_params['sl_pct'] = current_params['sl_pct'] * 0.9
                
        return {
            "best_metrics": best_result,
            "best_parameters": best_params,
            "history": history_log
        }
