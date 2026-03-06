import math

def scrub_nans(obj):
    if isinstance(obj, dict):
        return {k: scrub_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [scrub_nans(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return 0.0
    return obj

try:
    print(scrub_nans({"a": None, "b": "string", "c": 1.5, "d": float('nan')}))
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
