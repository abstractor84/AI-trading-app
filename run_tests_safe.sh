#!/bin/bash
cd /home/sakthi/Trading/gemini_nse_trader
source venv/bin/activate
python -m pytest "$@"
