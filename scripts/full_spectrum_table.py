"""
Print the full Chern/winding spectrum table for a given M and omega.

Run from the project root with:
    python scripts/full_spectrum_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.reporting import print_spectral_comparison_table
from floquet_graphene.constants import AMPLITUDE, HOPPING, POLARIZATION, LATTICE_CONSTANT


if __name__ == "__main__":
    print_spectral_comparison_table(
        M=3, omega=0.36, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT
    )
