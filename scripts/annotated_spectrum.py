"""
Plot the full quasienergy spectrum along the K-K' cut, annotated with Chern
and winding numbers for every band/gap.

Run from the project root with:
    python scripts/annotated_spectrum.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.plotting import plot_floquet_spectrum_with_invariants
from floquet_graphene.constants import POLARIZATION


if __name__ == "__main__":
    plot_floquet_spectrum_with_invariants(
        omega=0.36,
        pol=POLARIZATION,
        save_plot=True,
        output_filename="results/floquet_spectrum_omega_0.36.png",
    )
