import datetime
import logging
from openchart import NSEData

logging.basicConfig(level=logging.INFO)

nse = NSEData()
symbol = "RELIANCE"

# test 5m data
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=30)
start_dt = datetime.datetime.combine(start_date, datetime.time(9, 15))
end_dt = datetime.datetime.combine(end_date, datetime.time(15, 30))

print("Trying openchart 5m data...")
try:
    df = nse.historical(
        symbol=symbol,
        segment="EQ",
        start=start_dt,
        end=end_dt,
        interval="5m"
    )
    print(df.head() if df is not None else "No data returned")
except Exception as e:
    print(f"Exception: {e}")

# Try finding symbol first
try:
    search_results = nse.search(symbol, 'EQ')
    print("Search results:")
    print(search_results.head())
except Exception as e:
    print(f"Search exception: {e}")

