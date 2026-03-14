import pytest
from services.stock_discovery import StockDiscoveryService

def test_fetch_global_indices_structure():
    svc = StockDiscoveryService()
    # Mocking yfinance might be complex, but let's test the dictionary structure first
    # Or we can do a real (but limited) fetch if network allowed
    data = svc.fetch_global_indices()
    
    assert "india" in data
    assert "global" in data
    assert "vix" in data
    
    india = data["india"]
    assert "GIFT Nifty" in india
    assert "NIFTY 50" in india
    assert "BANK NIFTY" in india
    assert "NIFTY PHARMA" in india
    assert "NIFTY MIDCAP 100" in india
    
    glob = data["global"]
    assert "USD/INR" in glob
    assert "WTI Crude" in glob
    assert "Brent Crude" in glob
    assert "Gold" in glob
    assert "Silver" in glob
