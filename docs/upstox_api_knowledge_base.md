# Upstox API Knowledge Base

> **Source**: Official Upstox Developer API Documentation
> **Last Updated**: 2026-03-21
> **API Version**: V2 and V3

---

## Table of Contents

1. [Overview](#overview)
2. [Instrument Search API](#instrument-search-api)
3. [Instruments API](#instruments-api)
4. [Historical Data API](#historical-data-api)
5. [Market Quote API](#market-quote-api)
6. [Market Information API](#market-information-api)
7. [WebSocket API](#websocket-api)
8. [Streamer Functions](#streamer-functions)
9. [Common Headers & Authentication](#common-headers--authentication)
10. [Error Codes](#error-codes)

---

## Overview

Upstox provides a suite of RESTful APIs for building investment and trading platforms. The API base URL is:
- **Live**: `https://api.upstox.com/v2`
- **Sandbox**: `https://api.upstox.com/v2`

### Key Concepts

- **instrument_key**: Unique identifier used across Upstox APIs. Format: `{exchange}_{segment}|{identifier}`
  - Examples: `NSE_EQ|INE002A01018`, `NSE_FO|52023`, `NSE_INDEX|Nifty 50`
  - Preferred over `exchange_token` as it remains unique even after contract expiry

- **exchange_token**: Exchange-specific token that may be reused after expiry

---

## Instrument Search API

> **Documentation**: https://upstox.com/developer/api-documentation/instrument-search

### Purpose
Search instruments by name, symbol, or contract details without downloading full JSON files. Use instead of Instrument JSON files when you don't need the complete list.

### Endpoint
```
GET https://api.upstox.com/v2/instruments/search
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Free text search, max 50 characters (e.g., `RELIANCE`, `RELIANCE CE 1500`) |
| `exchanges` | string | No | Comma-separated: `ALL`, `NSE`, `BSE`, `MCX`. Default: `ALL` |
| `segments` | string | No | Comma-separated: `ALL`, `EQ`, `FO`, `CURR`, `COMM`, `INDEX`, `OPT`, `FUT`. Default: `ALL` |
| `instrument_types` | string | No | Comma-separated: `CE`, `PE` (options), `A`, `X` (series) |
| `expiry` | string | No | Expiry keywords or `yyyy-MM-dd` format |
| `atm_offset` | integer | No | Distance from ATM strike (0=ATM, positive=above, negative=below) |
| `page_number` | integer | No | Page number starting from 1. Default: `1` |
| `records` | integer | No | Records per page. Default: `10`, Max: `30` |

### Expiry Keywords

**Weekly:**
- `current_week`, `this_week`, `near_week`, `weekly`, `next_week`, `far_week`

**Monthly:**
- `current_month`, `this_month`, `near_month`, `monthly`, `next_month`, `far_month`

**Specific Date:**
- `yyyy-MM-dd` format (e.g., `2025-10-30`)

### ATM Search Example

| `atm_offset` | Strike Returned | Meaning |
|--------------|-----------------|---------|
| `0` | 24,500 | At the money |
| `2` | 24,600 | Two strikes above ATM |
| `-2` | 24,400 | Two strikes below ATM |

### Sample Request
```bash
curl --location 'https://api.upstox.com/v2/instruments/search?query=RELIANCE&exchanges=NSE&segments=FO&instrument_types=CE,PE&expiry=current_month&atm_offset=0&page_number=1&records=20' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'Authorization: Bearer {your_access_token}'
```

### Response Structure (EQ)

```json
{
  "status": "success",
  "data": [
    {
      "name": "RELIANCE INDUSTRIES LTD",
      "segment": "NSE_EQ",
      "exchange": "NSE",
      "isin": "INE002A01018",
      "instrument_key": "NSE_EQ|INE002A01018",
      "exchange_token": "2885",
      "trading_symbol": "RELIANCE",
      "short_name": "Reliance",
      "tick_size": 10.0,
      "lot_size": 1,
      "instrument_type": "EQ",
      "freeze_quantity": 100000.0,
      "qty_multiplier": 1,
      "security_type": "NORMAL"
    }
  ],
  "meta_data": {
    "page": {
      "page_number": 1,
      "total_pages": 1,
      "records": 20,
      "total_records": 2
    }
  }
}
```

### Response Fields (EQ)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Issuer/security name |
| `segment` | string | Segment code (e.g., `NSE_EQ`, `BSE_EQ`) |
| `exchange` | string | Exchange: `NSE`, `BSE`, `MCX` |
| `isin` | string | International Securities Identification Number |
| `instrument_key` | string | Unique identifier for Upstox APIs |
| `exchange_token` | string | Exchange-specific token |
| `trading_symbol` | string | Trading symbol |
| `short_name` | string | Short display name |
| `tick_size` | number | Minimum price movement |
| `lot_size` | number | Size of one lot |
| `instrument_type` | string | Series/type (`EQ`, `A`, etc.) |
| `freeze_quantity` | number | Maximum quantity that can be frozen |
| `qty_multiplier` | number | Quantity multiplier |
| `security_type` | string | Security classification |

### Response Structure (Futures)

```json
{
  "name": "RELIANCE INDUSTRIES LTD",
  "segment": "NSE_FO",
  "exchange": "NSE",
  "expiry": "2026-03-30",
  "weekly": false,
  "instrument_key": "NSE_FO|52023",
  "exchange_token": "52023",
  "trading_symbol": "RELIANCE FUT 30 MAR 26",
  "tick_size": 10.0,
  "lot_size": 500,
  "instrument_type": "FUT",
  "freeze_quantity": 15000.0,
  "underlying_key": "NSE_EQ|INE002A01018",
  "underlying_type": "EQUITY",
  "underlying_symbol": "RELIANCE",
  "strike_price": 0.0,
  "qty_multiplier": 1,
  "minimum_lot": 500
}
```

### Response Structure (Options)

```json
{
  "name": "RELIANCE INDUSTRIES LTD",
  "segment": "NSE_FO",
  "exchange": "NSE",
  "expiry": "2026-03-30",
  "weekly": false,
  "instrument_key": "NSE_FO|157318",
  "exchange_token": "157318",
  "trading_symbol": "RELIANCE 1380 CE 30 MAR 26",
  "tick_size": 5.0,
  "lot_size": 500,
  "instrument_type": "CE",
  "freeze_quantity": 15000.0,
  "underlying_key": "NSE_EQ|INE002A01018",
  "underlying_type": "EQUITY",
  "underlying_symbol": "RELIANCE",
  "strike_price": 1380.0,
  "qty_multiplier": 1,
  "minimum_lot": 500
}
```

### Response Structure (Index)

```json
{
  "name": "Nifty 50",
  "segment": "NSE_INDEX",
  "exchange": "NSE",
  "instrument_key": "NSE_INDEX|Nifty 50",
  "exchange_token": "26000",
  "trading_symbol": "NIFTY",
  "instrument_type": "INDEX"
}
```

### Error Codes

| Error Code | Description |
|------------|-------------|
| `UDAPI1169` | Query parameter cannot be empty |
| `UDAPI1170` | Query exceeds 50-character limit |
| `UDAPI1171` | Invalid exchange value |
| `UDAPI1172` | Invalid segment value |
| `UDAPI1173` | Records per page exceeds maximum of 30 |
| `UDAPI1174` | Page number must be 1 or greater |
| `UDAPI1175` | Invalid expiry value |

### Search Tips

- **Reliable pattern**: Short symbol + filters (`expiry`, `instrument_types`, `atm_offset`)
- Do NOT include spaces between comma-separated values (e.g., `NSE,BSE` not `NSE, BSE`)
- Avoid single characters or digits (too broad)
- Strike price only (e.g., `24000`) matches many contracts
- ISIN or exchange_token not searchable via `query`

---

## Instruments API

> **Documentation**: https://upstox.com/developer/api-documentation/instruments

### Purpose
Download complete list of BOD (Beginning of Day) contracts for trading.

### JSON File URLs

| File | URL |
|------|-----|
| Complete | `https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz` |
| NSE | `https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz` |
| BSE | `https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz` |
| MCX | `https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz` |

### Other Instrument Lists

| Type | URL |
|------|-----|
| Suspended | `https://assets.upstox.com/market-quote/instruments/exchange/suspended-instrument.json.gz` |
| MTF | `https://assets.upstox.com/market-quote/instruments/exchange/MTF.json.gz` |
| NSE MIS | `https://assets.upstox.com/market-quote/instruments/exchange/NSE_MIS.json.gz` |
| BSE MIS | `https://assets.upstox.com/market-quote/instruments/exchange/BSE_MIS.json.gz` |

### Segments

- `NSE_EQ` - NSE Equity
- `NSE_INDEX` - NSE Index
- `NSE_FO` - NSE F&O
- `NCD_FO` - NCD F&O
- `BSE_EQ` - BSE Equity
- `BSE_INDEX` - BSE Index
- `BSE_FO` - BSE F&O
- `BCD_FO` - BCD F&O
- `MCX_FO` - MCX F&O
- `NSE_COM` - NSE Commodity

### Underlying Types

- `COM` - Commodity
- `INDEX` - Index
- `EQUITY` - Equity
- `CUR` - Currency
- `IRD` - Interest Rate Derivative

### Important Notes

- Files are refreshed daily at ~6 AM
- CSV format is **deprecated** - use JSON format
- BOD instrument for next day won't include delisted stocks or expired contracts

---

## Historical Data API

> **Documentation**: https://upstox.com/developer/api-documentation/historical-data

### Sub-APIs

#### V3 APIs (Recommended)
1. **Historical Candle Data V3**
   - Endpoint: `GET https://api.upstox.com/v3/historical-candle`
   - Supports multiple timeframes for technical analysis

2. **Intraday Candle Data V3**
   - Endpoint: `GET https://api.upstox.com/v3/intraday-candle`
   - Current trading day price data

#### V2 APIs (Legacy)
1. **Historical Candle Data**
   - Endpoint: `GET https://api.upstox.com/historical-candle`
   - Intervals: 1-minute, 30-minute, daily, weekly, monthly

2. **Intraday Candle Data**
   - Endpoint: `GET https://api.upstox.com/intraday-candle`
   - Current trading day data

---

## Market Quote API

> **Documentation**: https://upstox.com/developer/api-documentation/market-quote

### Sub-APIs

#### V3 APIs (Recommended)
1. **Full Market Quotes V3**
   - Get complete snapshots for up to 500 instruments
   - Includes OHLC, volume, bid-ask depth

2. **OHLC Quotes V3**
   - Open, high, low, close prices
   - Up to 500 instruments

3. **LTP Quotes V3**
   - Last traded price
   - Up to 500 instruments

#### V2 APIs (Legacy)
1. **Full Market Quotes**
2. **OHLC Quotes**
3. **LTP Quotes**
4. **Option Greeks** - Delta, gamma, theta, vega, implied volatility

---

## Market Information API

> **Documentation**: https://upstox.com/developer/api-documentation/market-information

### Sub-APIs

1. **Market Holidays**
   - Get market holiday list for current year
   - Covers NSE, BSE, MCX exchanges

2. **Market Timings**
   - Get session open/close times for each exchange
   - Specific date parameter

3. **Exchange Status**
   - Check if markets are open, closed, or pre-open
   - NSE, BSE, MCX status

---

## WebSocket API

> **Documentation**: https://upstox.com/developer/api-documentation/websocket

### Purpose
Real-time streaming of market data and order updates over persistent connections.

### Technical Advantages
- **Efficiency**: Data pushed as available, no polling
- **Real-time**: Instant communication
- **Reduced overhead**: Single persistent connection

### Use Cases
- Real-time updates required
- High frequency data updates
- Network overhead reduction priority

### Sub-APIs

#### Market Data Feed V3
- Connect to live market data WebSocket
- Stream real-time price updates
- Subscribe to instrument-specific data

#### Get Market Data Feed Authorized URL V3
- Get authorized WebSocket endpoint
- Required before connecting

#### Portfolio Stream Feed
- Real-time order updates
- Position updates
- Holding updates
- GTT order updates

#### Get Portfolio Stream Feed Authorized URL
- Get authorized portfolio WebSocket endpoint

---

## Streamer Functions

> **Documentation**: https://upstox.com/developer/api-documentation/streamer-function

### Prerequisites
SDK must be installed for the programming language being used.

---

### MarketDataStreamerV3

#### Modes

| Mode | Description |
|------|-------------|
| `ltpc` | Last trade price, time, quantity, previous close |
| `full` | Full mode: ltpc + D5 depth + 1min/30min/daily candle + additional details |
| `option_greeks` | Only option greeks |
| `full_d30` | Full mode + 30 market level quotes |

#### Functions

1. **constructor MarketDataStreamerV3(apiClient, instrumentKeys, mode)**
   - Initialize streamer with optional instrument keys and mode

2. **connect()**
   - Establish WebSocket connection

3. **subscribe(instrumentKeys, mode)**
   - Subscribe to updates for given instrument keys
   - Both parameters mandatory

4. **unsubscribe(instrumentKeys)**
   - Stop updates for specified instruments

5. **changeMode(instrumentKeys, mode)**
   - Switch mode for subscribed instruments

6. **disconnect()**
   - End WebSocket connection

7. **auto_reconnect(enable, interval, retryCount)**
   - Customize auto-reconnect behavior
   - Parameters: enable flag, interval (seconds), max retries

#### Events

| Event | Description |
|-------|-------------|
| `open` | Connection established |
| `close` | Connection closed |
| `message` | Market updates received |
| `error` | Error occurred |
| `reconnecting` | Reconnect attempt initiated |
| `autoReconnectStopped` | Auto-reconnect exhausted |

#### Python Example

```python
import upstox_client

def on_message(message):
    print(message)

def main():
    configuration = upstox_client.Configuration()
    access_token = <ACCESS_TOKEN>
    configuration.access_token = access_token
    
    streamer = upstox_client.MarketDataStreamerV3(
        upstox_client.ApiClient(configuration),
        ["NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank"],
        "full"
    )
    
    streamer.on("message", on_message)
    streamer.connect()

if __name__ == "__main__":
    main()
```

---

### PortfolioDataStreamer

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_client` | object | Required | API client instance |
| `order_update` | boolean | True | Receive order updates |
| `position_update` | boolean | False | Receive position updates |
| `holding_update` | boolean | False | Receive holding updates |
| `gtt_update` | boolean | False | Receive GTT order updates |

#### Functions

1. **constructor PortfolioDataStreamer()**
2. **connect()** - Establish connection
3. **disconnect()** - End connection
4. **auto_reconnect(enable, interval, retryCount)** - Auto-reconnect settings

#### Events

Same as MarketDataStreamerV3

#### Python Example (All Updates)

```python
import upstox_client

def on_message(message):
    print(message)

def on_open():
    print("connection opened")

def main():
    configuration = upstox_client.Configuration()
    configuration.access_token = <ACCESS_TOKEN>
    
    streamer = upstox_client.PortfolioDataStreamer(
        upstox_client.ApiClient(configuration),
        order_update=True,
        position_update=True,
        holding_update=True,
        gtt_update=True
    )
    
    streamer.on("message", on_message)
    streamer.on("open", on_open)
    streamer.connect()

if __name__ == "__main__":
    main()
```

---

## Common Headers & Authentication

### Required Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `Accept` | `application/json` |
| `Authorization` | `Bearer {your_access_token}` |

### Authentication Flow
1. Use Login API to get authorization code
2. Exchange code for access token
3. Use access token in all API requests

---

## Error Codes

### Instrument Search Errors

| Code | Description |
|------|-------------|
| `UDAPI1169` | Query parameter cannot be empty |
| `UDAPI1170` | Query exceeds 50-character limit |
| `UDAPI1171` | Invalid exchange value |
| `UDAPI1172` | Invalid segment value |
| `UDAPI1173` | Records per page exceeds 30 |
| `UDAPI1174` | Page number must be ≥1 |
| `UDAPI1175` | Invalid expiry value |

---

## Rate Limits

> **Note**: Refer to https://upstox.com/developer/api-documentation/rate-limiting for latest limits

---

## SDKs & Installation

Upstox provides official SDKs for:
- Python: `pip install upstox-python`
- Node.js: `npm install upstox-js-sdk`
- Java
- PHP

---

## MCP Integration

Upstox provides MCP (Model Context Protocol) integration for AI applications:
- Reference: https://upstox.com/developer/api-documentation/mcp-integration

---

## Related Documentation

- [Authentication](https://upstox.com/developer/api-documentation/authentication)
- [API Structure](https://upstox.com/developer/api-documentation/request-response)
- [Rate Limits](https://upstox.com/developer/api-documentation/rate-limiting)
- [SDK Documentation](https://upstox.com/developer/api-documentation/sdk)
- [Field Pattern Appendix](https://upstox.com/developer/api-documentation/appendix/field-pattern)
- [Security Type Appendix](https://upstox.com/developer/api-documentation/appendix/equity-security-type)

---

*This knowledge base is auto-generated from official Upstox API documentation.*
