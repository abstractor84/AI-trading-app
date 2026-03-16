#!/bin/bash
export SIMULATION=true
export ENABLE_SIMULATION=true
export UPSTOX_ACCESS_TOKEN=mocked_token
./venv/bin/python3 -m pytest tests/ui/test_supernova_comprehensive.py -v
