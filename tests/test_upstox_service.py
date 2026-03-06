import pytest
import responses
import json
import gzip
from io import BytesIO
from unittest.mock import patch, MagicMock
from services.upstox_service import UpstoxService, get_instrument_key, _instrument_cache

@pytest.fixture(autouse=True)
def clear_cache():
    _instrument_cache.clear()

@responses.activate
def test_load_instrument_cache_success():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    data = [{"trading_symbol": "RELIANCE", "instrument_key": "NSE_EQ|INE002A01018"}]
    buf = BytesIO()
    with gzip.open(buf, 'wt', encoding='utf-8') as f:
        json.dump(data, f)
    responses.add(responses.GET, url, body=buf.getvalue(), status=200)
    
    key = get_instrument_key("RELIANCE.NS")
    assert key == "NSE_EQ|INE002A01018"

def test_get_instrument_key_missing():
    with patch("services.upstox_service._load_instrument_cache"):
        assert get_instrument_key("NONEXISTENT") is None

def test_upstox_service_init_no_token():
    with patch("os.getenv", return_value=None):
        svc = UpstoxService()
        assert svc.is_authenticated is False

@responses.activate
def test_validate_token_success():
    with patch("os.getenv", return_value="fake_token"):
        svc = UpstoxService()
        responses.add(responses.GET, f"{svc.BASE_URL}/user/profile", status=200)
        assert svc.validate_token() is True
        assert svc.is_authenticated is True

@responses.activate
def test_fetch_profile_success():
    with patch("os.getenv", return_value="fake_token"):
        svc = UpstoxService()
        responses.add(responses.GET, f"{svc.BASE_URL}/user/profile", 
                      json={"data": {"name": "Test User"}}, status=200)
    res = svc.fetch_profile()
    assert res["name"] == "Test User"

def test_reload_token():
    svc = UpstoxService()
    with patch("os.getenv", return_value="reloaded_token"), \
         patch.object(svc, "validate_token") as mock_val:
        svc.reload_token()
        assert svc.access_token == "reloaded_token"
        assert mock_val.called

        @responses.activate
        def test_fetch_intraday_candles_success():
            with patch("os.getenv", return_value="fake_token"):
                svc = UpstoxService()
                # [Timestamp, Open, High, Low, Close, Volume, OI]
                candles = [["2024-01-01T09:15:00+05:30", 100, 101, 99, 100.5, 1000, 0]]
                responses.add(responses.GET, "https://api.upstox.com/v3/historical-candle/intraday/KEY/minutes/5",
                              json={"data": {"candles": candles}}, status=200)
        
                df = svc.fetch_intraday_candles("KEY", interval="5minute")
                assert not df.empty
                assert df.iloc[0]["Close"] == 100.5            
            @responses.activate
            def test_fetch_historical_candles_success():
                with patch("os.getenv", return_value="fake_token"):
                    svc = UpstoxService()
                    candles = [["2024-01-01T09:15:00+05:30", 100, 101, 99, 100.5, 1000, 0]]
                    import re
                    # Match URL with any date suffix
                    url_re = re.compile(r"https://api\.upstox\.com/v3/historical-candle/KEY/minutes/5/.*")
                    responses.add(responses.GET, url_re, json={"data": {"candles": candles}}, status=200)
                    
                    df = svc.fetch_historical_candles("KEY", interval="5", from_date="2024-01-01", to_date="2024-01-05")
                    assert df is not None
                    assert not df.empty
            
            @responses.activate
            def test_fetch_intraday_candles_unauthorized():
                with patch("os.getenv", return_value="expired_token"):
                    svc = UpstoxService()
                    responses.add(responses.GET, "https://api.upstox.com/v3/historical-candle/intraday/KEY/minutes/5", status=401)
                    res = svc.fetch_intraday_candles("KEY", interval="5")
                    assert res is None
@responses.activate
def test_fetch_market_quote_success():
    with patch("os.getenv", return_value="token"):
        svc = UpstoxService()
        responses.add(responses.GET, f"{svc.BASE_URL}/market-quote/quotes", 
                      json={"data": {"NSE_EQ|RELIANCE": {"last_price": 2500}}}, status=200)
        res = svc.fetch_market_quote("NSE_EQ|RELIANCE")
        assert res["data"]["NSE_EQ|RELIANCE"]["last_price"] == 2500

def test_fetch_ohlcv_unauthenticated():
    svc = UpstoxService()
    svc.is_authenticated = False
    assert svc.fetch_ohlcv("T") is None


@patch.object(UpstoxService, "fetch_intraday_candles")
@patch.object(UpstoxService, "fetch_historical_candles")
@patch("services.upstox_service.get_instrument_key", return_value="KEY")
def test_fetch_ohlcv_merge(mock_key, mock_hist, mock_intra):
    with patch("os.getenv", return_value="token"):
        svc = UpstoxService()
        import pandas as pd
        idx1 = pd.to_datetime(["2024-01-01 09:15:00"]).tz_localize("Asia/Kolkata")
        idx2 = pd.to_datetime(["2024-01-01 09:20:00"]).tz_localize("Asia/Kolkata")
        
        df1 = pd.DataFrame({"Close": [100.0]}, index=idx1)
        df2 = pd.DataFrame({"Close": [101.0]}, index=idx2)
        
        mock_intra.return_value = df2
        mock_hist.return_value = df1
        
        res = svc.fetch_ohlcv("TICKER", days=2)
        assert len(res) == 2
        assert res.iloc[0]["Close"] == 100.0
        assert res.iloc[1]["Close"] == 101.0

@responses.activate
def test_upstox_errors_and_exceptions():
    with patch("os.getenv", return_value="token"):
        svc = UpstoxService()
        
        # 1. Rate limit (429)
        responses.add(responses.GET, f"{svc.BASE_URL}/historical-candle/intraday/K/5minute", status=429)
        assert svc.fetch_intraday_candles("K") is None
        
        # 2. Validation exception
        responses.add(responses.GET, f"{svc.BASE_URL}/user/profile", body=Exception("Conn error"))
        assert svc.validate_token() is False
        
        # 3. Candles to DF empty
        assert svc._candles_to_df([]).empty
        
        # 4. Market quote exception
        with patch("services.upstox_service.requests.get", side_effect=Exception("Quote Fail")):
            assert svc.fetch_market_quote("K") is None

def test_candles_to_df_tz_localize():
    svc = UpstoxService()
    # Test with naive timestamps
    import pandas as pd
    candles = [["2024-01-01 09:15:00", 100, 101, 99, 100, 1000, 0]]
    df = svc._candles_to_df(candles)
    assert df.index.tz is not None
    assert str(df.index.tz) == "Asia/Kolkata"

@responses.activate
def test_upstox_http_errors():
    with patch("os.getenv", return_value="token"):
        svc = UpstoxService()
        
        # Intraday 429 and 500
        responses.add(responses.GET, f"{svc.BASE_URL}/historical-candle/intraday/K/5minute", status=429)
        svc.fetch_intraday_candles("K")
        
        responses.add(responses.GET, f"{svc.BASE_URL}/historical-candle/intraday/K/5minute", status=500)
        svc.fetch_intraday_candles("K")

        # Historical 401, 429, 500
        import re
        url_re = re.compile(rf"{svc.BASE_URL}/historical-candle/K/5minute/.*")
        responses.add(responses.GET, url_re, status=401)
        svc.fetch_historical_candles("K")
        
        responses.add(responses.GET, url_re, status=429)
        svc.fetch_historical_candles("K")
        
        responses.add(responses.GET, url_re, status=500)
        svc.fetch_historical_candles("K")

@responses.activate
def test_fetch_profile_exception():
    with patch("os.getenv", return_value="token"):
        svc = UpstoxService()
        with patch("services.upstox_service.requests.get", side_effect=Exception("Hard error")):
            assert svc.fetch_profile() is None

def test_fetch_market_quote_exception():
    svc = UpstoxService()
    svc.is_authenticated = True
    with patch("requests.get", side_effect=Exception("Quote Fail")):
        assert svc.fetch_market_quote("K") is None


def test_validate_token_no_token():
    svc = UpstoxService()
    svc.access_token = None
    assert svc.validate_token() is False

def test_reload_token_empty():
    svc = UpstoxService()
    with patch("os.getenv", return_value=None):
        svc.reload_token()
        assert svc.is_authenticated is False


