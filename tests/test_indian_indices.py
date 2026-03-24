#!/usr/bin/env python3
"""
Unit tests for Indian Indices instrument key validation.

Tests verify:
1. Correct Upstox instrument key formats
2. Case sensitivity (SENSEX vs Sensex)
3. Exchange mapping for different index types
4. Response parsing for various key formats
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/sakthi/Trading/gemini_nse_trader')


class TestUpstoxIndices(unittest.TestCase):
    """Test Upstox instrument key definitions."""
    
    def test_upstox_indices_keys_defined(self):
        """Test that all required indices are defined."""
        from services.stock_discovery import StockDiscoveryService
        
        # Expected indices as per UI requirements
        expected_indices = [
            "GIFT Nifty", "Nifty 50", "Sensex", "Bank Nifty", 
            "FinNifty", "Nifty Midcap 100", "Nifty Smallcap 100"
        ]
        
        for idx in expected_indices:
            self.assertIn(idx, StockDiscoveryService._UPSTOX_INDICES, 
                         f"Missing index: {idx}")
    
    def test_sensex_key_case_sensitivity(self):
        """Test that Sensex uses correct case (SENSEX not Sensex)."""
        from services.stock_discovery import StockDiscoveryService
        
        sensex_key = StockDiscoveryService._UPSTOX_INDICES.get("Sensex", "")
        
        # Must be BSE_INDEX|SENSEX (ALL CAPS)
        self.assertEqual(sensex_key, "BSE_INDEX|SENSEX", 
                        f"Sensex key should be BSE_INDEX|SENSEX, got: {sensex_key}")
    
    def test_nifty_midcap_100_key_format(self):
        """Test that Nifty Midcap 100 uses correct key format."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_INDICES.get("Nifty Midcap 100", "")
        
        # Must be NIFTY MIDCAP 100 (uppercase)
        self.assertEqual(key, "NSE_INDEX|NIFTY MIDCAP 100", 
                        f"Midcap 100 key should be NSE_INDEX|NIFTY MIDCAP 100, got: {key}")
    
    def test_nifty_smallcap_100_key_format(self):
        """Test that Nifty Smallcap 100 uses correct key format."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_INDICES.get("Nifty Smallcap 100", "")
        
        # Must be NIFTY SMLCAP 100 (uppercase, SMLCAP not SMALLCAP)
        self.assertEqual(key, "NSE_INDEX|NIFTY SMLCAP 100", 
                        f"Smallcap 100 key should be NSE_INDEX|NIFTY SMLCAP 100, got: {key}")
    
    def test_finnifty_key_format(self):
        """Test that FinNifty uses correct key format."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_INDICES.get("FinNifty", "")
        
        # Must be Nifty Fin Service
        self.assertEqual(key, "NSE_INDEX|Nifty Fin Service", 
                        f"FinNifty key should be NSE_INDEX|Nifty Fin Service, got: {key}")
    
    def test_nifty_50_key_format(self):
        """Test that Nifty 50 uses correct key format."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_INDICES.get("Nifty 50", "")
        
        self.assertEqual(key, "NSE_INDEX|Nifty 50", 
                        f"Nifty 50 key should be NSE_INDEX|Nifty 50, got: {key}")
    
    def test_bank_nifty_key_format(self):
        """Test that Bank Nifty uses correct key format."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_INDICES.get("Bank Nifty", "")
        
        self.assertEqual(key, "NSE_INDEX|Nifty Bank", 
                        f"Bank Nifty key should be NSE_INDEX|Nifty Bank, got: {key}")


class TestIndiaVIXKey(unittest.TestCase):
    """Test India VIX key definition."""
    
    def test_india_vix_key(self):
        """Test India VIX uses correct key."""
        from services.stock_discovery import StockDiscoveryService
        
        key = StockDiscoveryService._UPSTOX_VIX_KEY
        
        self.assertEqual(key, "NSE_INDEX|India VIX", 
                        f"India VIX key should be NSE_INDEX|India VIX, got: {key}")


class TestYFinanceUnavailableIndices(unittest.TestCase):
    """Test yfinance fallback indices."""
    
    def test_yfinance_unavailable_list(self):
        """Test yfinance fallback list is properly configured."""
        from services.stock_discovery import StockDiscoveryService
        
        # GIFT Nifty should be in yfinance unavailable (not available in Upstox REST API)
        unavailable = StockDiscoveryService._UPSTOX_UNAVAILABLE_INDICES
        
        self.assertIn("GIFT Nifty", unavailable, 
                     "GIFT Nifty should be in yfinance fallback list")
    
    def test_midcap_smallcap_not_in_yfinance_fallback(self):
        """Test Midcap/Smallcap are NOT in yfinance fallback (now available in Upstox)."""
        from services.stock_discovery import StockDiscoveryService
        
        unavailable = StockDiscoveryService._UPSTOX_UNAVAILABLE_INDICES
        
        # These should now be available via Upstox
        self.assertNotIn("Nifty Midcap 100", unavailable, 
                        "Nifty Midcap 100 should be fetched via Upstox")
        self.assertNotIn("Nifty Smallcap 100", unavailable, 
                        "Nifty Smallcap 100 should be fetched via Upstox")


class TestResponseParsing(unittest.TestCase):
    """Test response parsing logic."""
    
    def test_formats_to_try_includes_case_insensitive(self):
        """Test that parsing includes case-insensitive matching."""
        # Simulate response data with uppercase key
        response_data = {
            "BSE_INDEX|SENSEX": {"last_price": 74000, "net_change": 500},
            "NSE_INDEX|NIFTY MIDCAP 100": {"last_price": 54000, "net_change": 300},
        }
        
        # Test case-insensitive matching logic
        test_key = "BSE_INDEX|SENSEX"  # Original key
        
        found_key = None
        for k in response_data:
            if k.upper() == test_key.upper():
                found_key = k
                break
        
        self.assertEqual(found_key, "BSE_INDEX|SENSEX", 
                        "Case-insensitive matching should find the key")


class TestFullIntegration(unittest.TestCase):
    """Full integration test - fetch actual data."""
    
    @unittest.skipIf(
        os.environ.get("SKIP_INTEGRATION_TESTS") == "1",
        "Integration tests disabled"
    )
    def test_fetch_global_indices_all_present(self):
        """Test that fetch_global_indices returns all 7 Indian indices."""
        from services.stock_discovery import StockDiscoveryService
        
        svc = StockDiscoveryService()
        result = svc.fetch_global_indices()
        
        india = result.get("india", {})
        
        # Check all 7 indices are present
        expected = [
            "GIFT Nifty", "Nifty 50", "Sensex", "Bank Nifty", 
            "FinNifty", "Nifty Midcap 100", "Nifty Smallcap 100"
        ]
        
        missing = [idx for idx in expected if idx not in india or india[idx].get("value") is None]
        
        if missing:
            # Print what's missing for debugging
            print(f"\nMissing indices: {missing}")
            print(f"Present indices: {list(india.keys())}")
        
        self.assertEqual(len(missing), 0, 
                        f"Missing indices: {missing}. Present: {list(india.keys())}")
    
    @unittest.skipIf(
        os.environ.get("SKIP_INTEGRATION_TESTS") == "1",
        "Integration tests disabled"
    )
    def test_india_vix_present(self):
        """Test that India VIX is returned."""
        from services.stock_discovery import StockDiscoveryService
        
        svc = StockDiscoveryService()
        result = svc.fetch_global_indices()
        
        vix = result.get("vix", {})
        
        self.assertIsNotNone(vix.get("value"), 
                            "India VIX should have a value")
        self.assertNotEqual(vix.get("value"), 0, 
                           "India VIX should not be 0")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUpstoxIndices))
    suite.addTests(loader.loadTestsFromTestCase(TestIndiaVIXKey))
    suite.addTests(loader.loadTestsFromTestCase(TestYFinanceUnavailableIndices))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestFullIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
