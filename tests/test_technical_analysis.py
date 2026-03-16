import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from services.technical_analysis import TechnicalAnalysisService

@pytest.fixture
def svc():
    return TechnicalAnalysisService()

@pytest.fixture
def sample_df():
    size = 100
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, size),
        "High": np.linspace(101, 111, size),
        "Low": np.linspace(99, 109, size),
        "Close": np.linspace(100.5, 110.5, size),
        "Volume": np.linspace(1000, 2000, size)
    }, index=pd.date_range("2024-01-01", periods=size, freq="5min"))
    return df

def test_fetch_ohlcv_all(svc):
    with patch("services.technical_analysis._upstox_svc") as mock_upstox, \
         patch("services.technical_analysis.yf.Ticker") as mock_ticker, \
         patch("os.getenv", return_value="false"):
        
        # 1. Upstox success
        mock_upstox.is_authenticated = True
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        df_1m = pd.DataFrame({"Open": [100]*60, "High": [105]*60, "Low": [95]*60, "Close": [101]*60, "Volume": [100]*60}, index=idx)
        mock_upstox.fetch_ohlcv.return_value = df_1m
        res = svc.fetch_ohlcv("RELIANCE.NS", interval="5m")
        assert len(res) == 12
        
        # 2. yfinance fallback
        mock_upstox.fetch_ohlcv.return_value = None
        mock_df = pd.DataFrame({"Close": [100]}, index=[pd.Timestamp.now()])
        mock_ticker.return_value.history.return_value = mock_df
        res = svc.fetch_ohlcv("RELIANCE.NS")
        assert not res.empty

def test_compute_indicators_full(svc, sample_df):
    res = svc.compute_indicators(sample_df)
    assert res is not None
    assert "ema_9" in res
    assert "lorentzian" in res

def test_evaluate_math_probability_comprehensive(svc):
    # Bullish Strong (1.0)
    ta_bull = {"close": 110, "vwap": 100, "ema_9": 105, "ema_21": 100, "adx_14": 30, "rsi_14": 50, "macd_hist": 1}
    assert svc.evaluate_math_probability(ta_bull) == 1.0
    
    # Bearish Strong (1.0)
    ta_bear = {"close": 90, "vwap": 100, "ema_9": 95, "ema_21": 100, "adx_14": 30, "rsi_14": 40, "macd_hist": -1}
    assert svc.evaluate_math_probability(ta_bear) == 1.0

    # Weak Bullish Trend
    ta_weak = {"close": 95, "vwap": 100, "ema_9": 105, "ema_21": 100, "rsi_14": 80}
    assert svc.evaluate_math_probability(ta_weak) == 0.1

    # Neutral/Moderate RSI
    ta_neu = {"close": 110, "vwap": 100, "ema_9": 105, "ema_21": 100, "rsi_14": 50, "adx_14": 10, "macd_hist": 0}
    # 0.4 (Trend) + 0.2 (RSI) = 0.6
    assert svc.evaluate_math_probability(ta_neu) == 0.6

def test_classify_signal_comprehensive(svc):
    # Strong Buy
    ta_sb = {"close": 110, "vwap": 100, "ema_9": 105, "ema_21": 100, "macd_hist": 1, "adx_14": 30, "rsi_14": 25, "vol_surge": 3}
    assert svc.classify_signal(ta_sb) == "STRONG BUY"
    
    # Sell -> SHORT SELL
    ta_sell = {"close": 95, "vwap": 100, "ema_9": 95, "ema_21": 100}
    assert svc.classify_signal(ta_sell) == "SHORT SELL"
    
    # Short Sell -> STRONG SHORT SELL
    ta_ss = {"close": 90, "vwap": 100, "ema_9": 95, "ema_21": 100, "macd_hist": -1, "adx_14": 30, "vol_surge": 3, "rsi_14": 80}
    assert svc.classify_signal(ta_ss) == "STRONG SHORT SELL"

def test_get_status_and_fundamentals(svc):
    with patch("services.technical_analysis._upstox_svc") as mock_upstox, \
         patch("services.quota_service.quota_svc.get_total_daily_usage", return_value=2), \
         patch("services.technical_analysis.yf.Ticker") as mock_ticker:
        
        mock_upstox.is_authenticated = True
        mock_upstox.fetch_profile.return_value = {"user_name": "User"}
        res = svc.get_connection_status()
        assert res["upstox"]["user"] == "User"
        assert res["ai"]["remaining"] == 18
        
        mock_ticker.return_value.info = {"sector": "Energy"}
        f = svc.fetch_fundamentals("T")
        assert f["sector"] == "Energy"

def test_compute_indicators_exceptions(svc, sample_df):
    # Typical price exception
    df = sample_df.copy()
    df["High"] = "Invalid"
    # To avoid crashing pandas_ta but still run through the code,
    # we'll mock the ta methods and add expected columns
    with patch.object(pd.DataFrame, "ta") as mock_ta, \
         patch("services.technical_analysis.classifier.classify_series", return_value=[{"signal": 0, "score": 0}]):
        # Add required columns so extraction logic doesn't fail

        df["MACD_12_26_9"] = [0.0]*len(df)
        df["MACDh_12_26_9"] = [0.0]*len(df)
        df["MACDs_12_26_9"] = [0.0]*len(df)
        df["BBM_20_2.0"] = [0.0]*len(df)
        df["BBU_20_2.0"] = [0.0]*len(df)
        df["BBL_20_2.0"] = [0.0]*len(df)
        df["ADX_14"] = [0.0]*len(df)
        df["EMA_9"] = [0.0]*len(df)
        df["EMA_21"] = [0.0]*len(df)
        df["RSI_14"] = [0.0]*len(df)
        
        res = svc.compute_indicators(df)
        assert res is not None
        # Should have fallen back in Typical Price AND VWAP (due to "Invalid" High)
        assert res["vwap"] == 110.5

    # VWAP explicit exception test
    df2 = sample_df.copy()
    with patch("pandas.Series.cumsum", side_effect=Exception("VWAP Error")):
        res = svc.compute_indicators(df2)
        assert res["vwap"] == 110.5 # Fallback to close

def test_analyze_stock_full_flow(svc, sample_df):
    with patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv", return_value=sample_df):
        res = svc.analyze_stock("RELIANCE.NS")
        assert res is not None
        assert "lorentzian" in res
    
    with patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv", side_effect=Exception("Error")):
        assert svc.analyze_stock("RELIANCE.NS") is None

def test_compute_indicators_edge_branches(svc):
    # Small DF
    assert svc.compute_indicators(pd.DataFrame({"Close": [100]*10})) is None

def test_evaluate_math_probability_more_branches(svc):
    # ADX 20 (Moderate momentum -> 0.1), RSI 50 -> 0.2, Trend -> 0.4 = 0.7
    ta_mod = {"close": 110, "vwap": 100, "ema_9": 105, "ema_21": 100, "adx_14": 20, "rsi_14": 50, "macd_hist": 0}
    assert svc.evaluate_math_probability(ta_mod) == 0.7
    
    # Overextended RSI 80 (Overbought on a bullish trend blocks the trade -> 0)
    ta_ext = {"close": 110, "vwap": 100, "ema_9": 105, "ema_21": 100, "rsi_14": 80}
    assert svc.evaluate_math_probability(ta_ext) == 0.0



@patch("services.technical_analysis.yf.download")
@patch("services.technical_analysis.yf.Ticker")
@patch("os.getenv", return_value="false")
def test_fetch_ohlcv_yfinance_download_fallback(mock_os, mock_ticker, mock_download, svc):
    from services.technical_analysis import _upstox_svc
    _upstox_svc.is_authenticated = False
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    mock_df = pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp.now()])
    mock_download.return_value = mock_df

    res = svc.fetch_ohlcv("TICKER")
    assert not res.empty
    assert mock_download.called

def test_fetch_ohlcv_exceptions(svc):
    with patch("services.technical_analysis.yf.Ticker", side_effect=Exception("Fetch Fail")), \
         patch("os.getenv", return_value="false"):
        res = svc.fetch_ohlcv("TICKER")
        assert res.empty


