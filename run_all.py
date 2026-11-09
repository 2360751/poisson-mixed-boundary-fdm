"""Shim so both entry points work:

    python run_all.py        (from the repository root)
    python code/run_all.py   (the canonical entry, per the code-review checklist)

Delegates to code/run_all.py, which regenerates every report figure and table.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "code", "run_all.py")]))
