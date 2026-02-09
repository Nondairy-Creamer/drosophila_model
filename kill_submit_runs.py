#!/usr/bin/env python3
"""Kill all submit_runs.py and run_model.py processes."""
import os
os.system('kill $(pgrep -f "submit_runs.py") 2>/dev/null; kill $(pgrep -f "run_model.py") 2>/dev/null')
