"""
Compute and print the Chern numbers of the central valence/conduction bands
at a single drive frequency omega.

Run from the project root with:
    python scripts/chern_at_omega.py
"""

import sys
from pathlib import Path

# Make the floquet_graphene package importable when this script is run
# directly (i.e. without `pip install -e .`). This walks up from this
# file's folder (scripts/) to the project root, and adds that to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.runners import run_calculate_chern
from floquet_graphene.constants import POLARIZATION


if __name__ == "__main__":
    run_calculate_chern(target_omega=0.36, pol=POLARIZATION)
