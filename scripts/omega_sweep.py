"""
Sweep the drive frequency omega and print a table of Chern numbers and gaps.

Run from the project root with:
    python scripts/omega_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.runners import run_omega_sweep
from floquet_graphene.constants import POLARIZATION


if __name__ == "__main__":
    results = run_omega_sweep(omega_min=0.15, omega_max=0.80, num_points=50, pol=POLARIZATION)
    # `results` is a list of dicts (one per omega) if you want to save it,
    # e.g.: import json; json.dump(results, open("results/omega_sweep.json", "w"))
