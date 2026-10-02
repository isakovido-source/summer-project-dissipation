"""
Generate the interactive 3D band figure and the 2D Dirac-cut figure for a
given omega, saved under results/.

Run from the project root with:
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from floquet_graphene.runners import run_generate_graphs
from floquet_graphene.constants import POLARIZATION


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "results"
    run_generate_graphs(omega_val=0.36, results_dir=str(results_dir), pol=POLARIZATION)
