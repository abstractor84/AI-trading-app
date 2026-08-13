# Virtual Environment Guidelines

## CRITICAL RULE
- ALWAYS use the virtual environment from project root: `/home/sakthi/Trading/gemini_nse_trader/venv`
- NEVER use system-wide python (`python` or `python3` without venv)
- NEVER install pip packages in system-wide python

## How to Use

### Running Python scripts:
```bash
cd /home/sakthi/Trading/gemini_nse_trader
source venv/bin/activate
python your_script.py
```

### Running Tests:
```bash
cd /home/sakthi/Trading/gemini_nse_trader
source venv/bin/activate
python -m pytest tests/...
```

### Installing Packages:
```bash
cd /home/sakthi/Trading/gemini_nse_trader
source venv/bin/activate
pip install package_name
```

## Why This Matters
- System-wide Python may have different package versions
- The project was developed and tested with specific venv packages
- Using system Python can cause compatibility issues and test failures
