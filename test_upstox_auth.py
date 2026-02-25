"""
Upstox OAuth Token Exchange Test Script
========================================
Step 1: Print the Authorize URL. Open it in your browser and log in.
Step 2: After login, Upstox redirects to http://localhost:8000/?code=XYZ
Step 3: Paste the code here. The script exchanges it for an access token.
Step 4: Saves the token to .env as UPSTOX_ACCESS_TOKEN.
"""
import os
import sys
import requests
from dotenv import load_dotenv, set_key

load_dotenv()

CLIENT_ID     = os.getenv("UPSTOX_API_KEY", "")
CLIENT_SECRET = os.getenv("UPSTOX_API_SECRET", "")
REDIRECT_URI  = "http://localhost:8000"
DOT_ENV_PATH  = ".env"

# -------------------------------------------------------------------------
# Step 1 – Print the authorize URL
# -------------------------------------------------------------------------
auth_url = (
    f"https://api.upstox.com/v2/login/authorization/dialog"
    f"?response_type=code"
    f"&client_id={CLIENT_ID}"
    f"&redirect_uri={REDIRECT_URI}"
)

print("=" * 60)
print("STEP 1: Open the following URL in your browser:")
print("=" * 60)
print(auth_url)
print()

# -------------------------------------------------------------------------
# Step 2 – Wait for the user to paste the code
# -------------------------------------------------------------------------
code = input("STEP 2: Paste the 'code' value from the redirect URL here: ").strip()

if not code:
    print("No code provided. Exiting.")
    sys.exit(1)

# -------------------------------------------------------------------------
# Step 3 – Exchange for token
# -------------------------------------------------------------------------
token_url = "https://api.upstox.com/v2/login/authorization/token"
payload = {
    "code":          code,
    "client_id":     CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri":  REDIRECT_URI,
    "grant_type":    "authorization_code",
}

print("\nSTEP 3: Exchanging code for access token...")
resp = requests.post(
    token_url,
    data=payload,
    headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
    timeout=15,
)

print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    access_token = data.get("access_token", "")
    token_type   = data.get("token_type", "")
    print(f"\n✅ Token received! Type: {token_type}")
    print(f"Token (first 20 chars): {access_token[:20]}...")

    # -------------------------------------------------------------------------
    # Step 4 – Save to .env
    # -------------------------------------------------------------------------
    set_key(DOT_ENV_PATH, "UPSTOX_ACCESS_TOKEN", access_token)
    print(f"\n✅ UPSTOX_ACCESS_TOKEN saved to {DOT_ENV_PATH}")
    print("Restart the app server to pick up the new token.")
else:
    print(f"\n❌ Token exchange failed:")
    print(resp.text)
