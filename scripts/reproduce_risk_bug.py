from services.risk_engine import RiskEngine

re = RiskEngine()

def test_levels(price, action, atr):
    print(f"\n--- Testing: {action} at ₹{price} (ATR: {atr}) ---")
    levels = re.compute_sl_target(price, action, atr)
    if levels:
        sl = levels['stop_loss']
        t1 = levels['target_1']
        t2 = levels['target_2']
        print(f"Result: SL: {sl}, T1: {t1}, T2: {t2}")
        
        if action.upper() == "BUY":
            if sl < price and t1 > price:
                print("✅ PASS: SL below Entry, Target above Entry")
            else:
                print("❌ FAIL: Inverted levels for BUY")
        elif action.upper() in ["SHORT SELL", "SELL"]:
            if sl > price and t1 < price:
                print("✅ PASS: SL above Entry, Target below Entry")
            else:
                print("❌ FAIL: Inverted levels for SHORT SELL")
    else:
        print("❌ FAIL: No levels returned")

# The case reported by user: INDUSINDBK Buy at 965.5, ATR approx 2.95
test_levels(965.5, "BUY", 2.95)
test_levels(965.5, "buy", 2.95) # Case insensitivity test

# Testing Short Sell for parity
test_levels(965.5, "SHORT SELL", 2.95)
test_levels(965.5, "SELL", 2.95) # Alias test
