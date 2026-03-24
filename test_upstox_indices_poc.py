#!/usr/bin/env python3
"""
POC Script: Verify Upstox Instrument Key Formats for Indian Indices

This script tests different instrument key formats to find the correct ones
that work with Upstox API for fetching Indian market indices.

Key findings from research:
1. Upstox uses format: {exchange}_{segment}|{identifier}
2. Exchange: NSE, BSE, MCX, NSE_IFSC (for GIFT City)
3. Segment: NSE_INDEX, BSE_INDEX for indices
4. Case matters: SENSEX (not Sensex), Nifty 50 (not NIFTY 50)

Expected instrument keys:
- Nifty 50: NSE_INDEX|Nifty 50
- Nifty Bank: NSE_INDEX|Nifty Bank
- Sensex: BSE_INDEX|SENSEX (CAPS!)
- India VIX: NSE_INDEX|India VIX
- GIFT Nifty: NSE_IFSC|GIFT Nifty (International Exchange)
- FinNifty: NSE_INDEX|Nifty Fin Service
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv()


def get_upstox_client():
    """Get authenticated Upstox client."""
    try:
        from services.upstox_service import upstox_client
        if upstox_client.is_authenticated:
            return upstox_client
        else:
            logger.error("Upstox client not authenticated")
            return None
    except Exception as e:
        logger.error(f"Failed to import upstox_client: {e}")
        return None


def test_market_quote_api(client, instrument_keys):
    """Test market quote API with different key formats."""
    results = {}
    
    if not client:
        logger.error("No authenticated client")
        return results
    
    for name, key in instrument_keys.items():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {name} -> {key}")
            logger.info(f"{'='*60}")
            
            # Test with the key as-is
            q = client.fetch_market_quote(key)
            
            if q and 'data' in q:
                raw_data = q['data']
                logger.info(f"Response keys: {list(raw_data.keys())}")
                
                # Check what keys are available
                if raw_data:
                    # Try to find our key in the response
                    found_key = None
                    data = None
                    
                    # Try exact match
                    if key in raw_data:
                        found_key = key
                        data = raw_data[key]
                    # Try with pipe replaced by colon
                    elif key.replace("|", ":") in raw_data:
                        found_key = key.replace("|", ":")
                        data = raw_data[found_key]
                    # Try just the name part
                    elif key.split("|")[-1] in raw_data:
                        found_key = key.split("|")[-1]
                        data = raw_data[found_key]
                    # Try uppercase for BSE indices
                    elif key.upper() in [k.upper() for k in raw_data]:
                        for k in raw_data:
                            if k.upper() == key.upper():
                                found_key = k
                                data = raw_data[k]
                                break
                    
                    if data:
                        current = data.get('last_price', 0)
                        net_change = data.get('net_change', 0)
                        prev_close = data.get('ohlc', {}).get('close', current)
                        change_pct = (net_change / prev_close * 100) if prev_close != 0 else 0
                        results[name] = {
                            "instrument_key_sent": key,
                            "instrument_key_found": found_key,
                            "value": current,
                            "change": net_change,
                            "change_pct": round(change_pct, 2),
                            "success": True
                        }
                        logger.info(f"✅ SUCCESS: {name} = {current} ({change_pct:+.2f}%)")
                    else:
                        results[name] = {
                            "instrument_key_sent": key,
                            "available_keys": list(raw_data.keys()),
                            "success": False,
                            "error": "Key not found in response"
                        }
                        logger.warning(f"❌ Key not found in response. Available: {list(raw_data.keys())}")
                else:
                    results[name] = {"success": False, "error": "Empty response data"}
                    logger.warning(f"❌ Empty response for {name}")
            else:
                results[name] = {"success": False, "error": q.get('error', 'No data returned')}
                logger.warning(f"❌ No data returned for {name}: {q}")
                
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            logger.error(f"❌ Error for {name}: {e}")
    
    return results


def test_yfinance_tickers():
    """Test different yfinance ticker formats for indices."""
    import yfinance as yf
    
    # Different ticker formats to test
    test_tickers = [
        ("Nifty Midcap 100", "^NIFTYMIDCAP100"),
        ("Nifty Midcap 100", "NIFTYMIDCAP.NS"),
        ("Nifty Midcap 100", "^NIFTYMIDCAP"),
        ("Nifty Smallcap 100", "^NIFTYSMALLCAP100"),
        ("Nifty Smallcap 100", "NIFTYSMALLCAP.NS"),
        ("Nifty Smallcap 100", "^NIFTYSMALLCAP"),
    ]
    
    results = {}
    
    for name, ticker in test_tickers:
        try:
            logger.info(f"\nTesting yfinance: {name} -> {ticker}")
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            
            if not data.empty:
                current = float(data['Close'].iloc[-1])
                prev = float(data['Close'].iloc[-2]) if len(data) > 1 else current
                change = current - prev
                change_pct = (change / prev * 100) if prev != 0 else 0
                results[ticker] = {
                    "name": name,
                    "success": True,
                    "value": current,
                    "change_pct": round(change_pct, 2)
                }
                logger.info(f"✅ SUCCESS: {ticker} = {current} ({change_pct:+.2f}%)")
            else:
                results[ticker] = {"name": name, "success": False, "error": "Empty data"}
                logger.warning(f"❌ Empty data for {ticker}")
        except Exception as e:
            results[ticker] = {"name": name, "success": False, "error": str(e)}
            logger.error(f"❌ Error for {ticker}: {e}")
    
    return results


def test_instrument_search():
    """Test Upstox Instrument Search API for finding correct index keys."""
    try:
        from services.upstox_service import upstox_client
        if not upstox_client.is_authenticated:
            logger.error("Upstox not authenticated")
            return {}
        
        # Test search for different indices
        search_queries = [
            "Nifty 50",
            "Nifty Bank", 
            "Sensex",
            "India VIX",
            "GIFT Nifty",
            "FinNifty",
            "Nifty Midcap",
            "Nifty Smallcap"
        ]
        
        results = {}
        
        for query in search_queries:
            try:
                logger.info(f"\nSearching for: {query}")
                # Use search API
                url = "https://api.upstox.com/v2/instruments/search"
                params = {
                    "query": query,
                    "exchanges": "NSE" if "Nifty" in query else "BSE",
                    "segments": "INDEX",
                    "records": 5
                }
                headers = {
                    "Authorization": f"Bearer {upstox_client.access_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('data'):
                        for item in data['data']:
                            logger.info(f"  Found: {item.get('instrument_key')} - {item.get('name')} ({item.get('trading_symbol')})")
                            results[query] = {
                                "instrument_key": item.get('instrument_key'),
                                "name": item.get('name'),
                                "trading_symbol": item.get('trading_symbol'),
                                "segment": item.get('segment'),
                                "exchange": item.get('exchange')
                            }
                    else:
                        logger.warning(f"  No results for {query}")
                        results[query] = {"error": "No results"}
                else:
                    logger.warning(f"  Search failed: {resp.status_code} - {resp.text[:200]}")
                    results[query] = {"error": f"HTTP {resp.status_code}"}
                    
            except Exception as e:
                logger.error(f"  Search error for {query}: {e}")
                results[query] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        logger.error(f"Instrument search failed: {e}")
        return {}


def main():
    """Main POC execution."""
    logger.info("="*80)
    logger.info("POC: Upstox Indian Indices Instrument Key Verification")
    logger.info("="*80)
    
    # Step 1: Get authenticated client
    client = get_upstox_client()
    if not client:
        logger.error("Cannot proceed without authenticated Upstox client")
        return
    
    logger.info(f"Upstox client authenticated: {client.is_authenticated}")
    logger.info(f"Access token: {client.access_token[:20]}...")
    
    # Step 2: Define all index keys to test (comprehensive list)
    # Based on research: correct exchange/segment mapping
    test_keys = {
        # NSE Indices
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Nifty Bank": "NSE_INDEX|Nifty Bank", 
        "India VIX": "NSE_INDEX|India VIX",
        
        # Testing different case variations
        "Sensex (SENSEX caps)": "BSE_INDEX|SENSEX",
        "Sensex (Sensex mixed)": "BSE_INDEX|Sensex",
        
        # GIFT Nifty (International Exchange)
        "GIFT Nifty": "NSE_IFSC|GIFT Nifty",
        
        # FinNifty
        "FinNifty": "NSE_INDEX|Nifty Fin Service",
        
        # Midcap/Smallcap
        "Nifty Midcap 100": "NSE_INDEX|Nifty Midcap 100",
        "Nifty Smallcap 100": "NSE_INDEX|Nifty Smallcap 100",
    }
    
    # Step 3: Test market quote API
    logger.info("\n\n" + "="*80)
    logger.info("STEP 1: Testing Market Quote API with different keys")
    logger.info("="*80)
    
    results = test_market_quote_api(client, test_keys)
    
    # Step 4: Test instrument search API
    logger.info("\n\n" + "="*80)
    logger.info("STEP 2: Testing Instrument Search API")
    logger.info("="*80)
    
    search_results = test_instrument_search()
    
    # Step 5: Test yfinance tickers
    logger.info("\n\n" + "="*80)
    logger.info("STEP 3: Testing yfinance ticker symbols")
    logger.info("="*80)
    
    yf_results = test_yfinance_tickers()
    
    # Summary
    logger.info("\n\n" + "="*80)
    logger.info("SUMMARY OF FINDINGS")
    logger.info("="*80)
    
    print("\n✅ WORKING Upstox Keys:")
    for name, r in results.items():
        if r.get("success"):
            print(f"  {name}: {r.get('instrument_key_found')} -> {r.get('value')}")
    
    print("\n❌ FAILED Upstox Keys:")
    for name, r in results.items():
        if not r.get("success"):
            print(f"  {name}: {r.get('instrument_key_sent')} - {r.get('error')}")
    
    print("\n📋 Instrument Search Results:")
    for query, r in search_results.items():
        if "instrument_key" in r:
            print(f"  {query}: {r.get('instrument_key')} ({r.get('exchange')}:{r.get('segment')})")
        else:
            print(f"  {query}: FAILED - {r.get('error')}")
    
    print("\n✅ WORKING yfinance Tickers:")
    for ticker, r in yf_results.items():
        if r.get("success"):
            print(f"  {ticker}: {r.get('value')} ({r.get('change_pct'):+.2f}%)")
    
    # Save results to file
    output = {
        "market_quote_results": results,
        "instrument_search_results": search_results,
        "yfinance_results": yf_results
    }
    
    output_file = Path("/home/sakthi/Trading/gemini_nse_trader/poc_results.json")
    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
