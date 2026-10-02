"""
Scan direct gaps and the valence-band Chern number vs omega, then plot.

This script is the worked example of "compute here, plot there": it calls
run_scan_with_all_gaps() to get plain numbers back, and only then calls
plot_gap_and_chern_scan() to draw them.

Run from the project root with:
    python scripts/gap_and_chern_scan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.runners import run_scan_with_all_gaps
from floquet_graphene.plotting import plot_gap_and_chern_scan
from floquet_graphene.constants import POLARIZATION


if __name__ == "__main__":
    scan_results = run_scan_with_all_gaps(omega_min=0.15, omega_max=0.80, num_points=40, pol=POLARIZATION)
    plot_gap_and_chern_scan(scan_results, pol=POLARIZATION, save_plot=True, output_filename="results/gap_chern_scan.png")
