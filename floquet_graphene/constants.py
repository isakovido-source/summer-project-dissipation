"""
Physical constants and default parameters for the driven-graphene Floquet model.

This is the ONLY place default values should be defined. Every other module
imports from here instead of hard-coding a number, so changing a default
means changing it in exactly one place.
"""

import numpy as np

# ----------------------------------------------------------------------
# Units: we choose hbar = 1 and e = 1, so energies/frequencies/times are
# all expressed in the same (dimensionless) units.
# ----------------------------------------------------------------------
H_BAR = 1.0
ELECTRON_CHARGE = 1.0

# ----------------------------------------------------------------------
# Default physical parameters of the driven graphene model.
# These are just DEFAULTS: every function below accepts them as keyword
# arguments so a script can override them without touching this file.
# ----------------------------------------------------------------------
POLARIZATION = -1          # +1 = left circular, -1 = right circular
AMPLITUDE = 1.2            # drive amplitude A_0
HOPPING = 0.2              # nearest-neighbor hopping t
LATTICE_CONSTANT = 1.0     # nearest-neighbor distance a

# Derived quantities (kept here so they always stay consistent with the
# defaults above; a script using different A_0/t/a should recompute these
# itself rather than reuse the module-level values).
FERMI_VELOCITY = 3 * HOPPING * LATTICE_CONSTANT / (2 * H_BAR)   # v_F = 3ta/(2*hbar)

# ----------------------------------------------------------------------
# Numerical thresholds used by the adaptive Chern-number solver.
# ----------------------------------------------------------------------
FLUX_THRESHOLD = 0.90 * np.pi   # FHS admissibility: max plaquette flux must stay below this
MIN_RESOLVABLE_GAP = 1e-4       # gaps smaller than this are treated as "closed"
