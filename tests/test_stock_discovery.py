import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from services.stock_discovery import StockDiscoveryService

@pytest.fixture
def svc():
    return StockDiscoveryService()

@patch("services.stock_discovery.yf.Ticker")
def test_fetch_global_indices_success(mock_ticker, svc):
    # Mock history with 2 rows for change calculation
    mock_hist = pd.DataFrame({"Close": [100.0, 105.0]})
    mock_ticker.return_value.history.return_value = mock_hist
    
    res = svc.fetch_global_indices()
    assert res["global"]["S&P 500"]["change_pct"] == 5.0
    assert res["vix"]["change_pct"] == 5.0

@patch("services.stock_discovery.yf.Ticker")
def test_fetch_global_indices_partial_data(mock_ticker, svc):

    # Mock history with only 1 row (insufficient for change calculation)
    mock_hist = pd.DataFrame({"Close": [100.0]})
    mock_ticker.return_value.history.return_value = mock_hist
    
    res = svc.fetch_global_indices()
    assert res["global"]["S&P 500"]["value"] == 100.0
    assert res["global"]["S&P 500"]["change"] == 0.0
    mock_ticker.side_effect = Exception("VIX Error")
    res = svc.fetch_global_indices()
    assert res["vix"]["value"] == 0

@patch("services.stock_discovery.pd.read_csv")
def test_discover_nse_universe_bad_csv(mock_read, svc):
    # CSV missing 'Symbol' column
    mock_df = pd.DataFrame({"Wrong": ["A", "B"]})
    mock_read.return_value = mock_df
    res = svc.discover_nse_universe()
    # Should fallback
    assert len(res) >= 50

def test_get_top_candidates_lazy_init(svc):
    svc.universe = []
    with patch.object(svc, "discover_nse_universe") as mock_disc, \
         patch("services.stock_discovery.yf.download") as mock_down:
        mock_down.return_value = pd.DataFrame()
        svc._get_top_candidates()
        assert mock_disc.called

@patch("services.stock_discovery.yf.download")
def test_get_top_candidates_nan_handling(mock_download, svc):
    # Create multi-index with NaN
    svc.universe = ["NAN.NS"]
    iterables = [["NAN.NS"], ["Close", "Volume"]]
    index = pd.MultiIndex.from_product(iterables, names=["ticker", "metric"])
    df = pd.DataFrame([[np.nan, 1000], [np.nan, 5000]], columns=index)
    mock_download.return_value = df
    
    res = svc._get_top_candidates(limit=1)
    assert res == [] # NAN ticker was skipped


@patch("services.stock_discovery.pd.read_csv")
def test_discover_nse_universe_success(mock_read, svc):
    mock_df = pd.DataFrame({"Symbol": ["RELIANCE", "TCS", "NAN"]})
    mock_read.return_value = mock_df
    
    res = svc.discover_nse_universe()
    assert "RELIANCE.NS" in res
    assert "TCS.NS" in res
    assert "NAN.NS" not in res

@patch("services.stock_discovery.pd.read_csv", side_effect=Exception("Network error"))
def test_discover_nse_universe_fallback(mock_read, svc):
    res = svc.discover_nse_universe()
    # Should fallback to internal cache (Nifty 100)
    assert "RELIANCE.NS" in res
    assert len(res) >= 50

@patch("services.stock_discovery.yf.download")
def test_get_top_candidates_success(mock_download, svc):
    # Mock yf.download return format (Ticker, Metric) multi-index
    svc.universe = ["AAPL.NS", "MSFT.NS"]
    
    # Create a multi-index dataframe as returned by yf.download(group_by="ticker")
    iterables = [["AAPL.NS", "MSFT.NS"], ["Close", "Volume"]]
    index = pd.MultiIndex.from_product(iterables, names=["ticker", "metric"])
    
    # Simple data: 2 rows (days)
    data = [
        [100, 1000, 200, 1000], # Day 1
        [110, 5000, 220, 5000]  # Day 2
    ]
    df = pd.DataFrame(data, columns=index)
    mock_download.return_value = df
    
    res = svc._get_top_candidates(limit=2)
    assert len(res) == 2
    # AAPL: 10% gain, 5x vol surge (average of 1000 and 5000 is 3000? No, mean of all rows)
    # MSFT: 5% loss, 0.6x vol surge
    assert res[0] == "AAPL.NS"

@patch("services.stock_discovery.yf.download", side_effect=Exception("Download failed"))
def test_get_top_candidates_error(mock_download, svc):
    svc.universe = ["K1.NS", "K2.NS"]
    res = svc._get_top_candidates(limit=1)
    # Should fallback to returning start of universe
    assert res == ["K1.NS"]
