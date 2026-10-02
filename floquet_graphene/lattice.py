"""
Graphene lattice geometry and Brillouin-zone (BZ) grid construction.

Kept separate from the Hamiltonians so that any function needing the lattice
vectors or a k-grid imports it from one place instead of re-deriving it.
"""

import numpy as np

from .constants import LATTICE_CONSTANT


def get_lattice_vectors(a=LATTICE_CONSTANT):
    """
    Returns real-space and reciprocal-space lattice vectors for graphene.
    """
    # Real-space lattice vectors
    a1 = (np.sqrt(3) * a / 2.0) * np.array([np.sqrt(3), 1.0])
    a2 = (np.sqrt(3) * a / 2.0) * np.array([np.sqrt(3), -1.0])

    # Reciprocal lattice vectors (a_i . b_j = 2*pi*delta_ij)
    b1 = (2.0 * np.pi / (3.0 * a)) * np.array([1.0, np.sqrt(3)])
    b2 = (2.0 * np.pi / (3.0 * a)) * np.array([1.0, -np.sqrt(3)])

    return a1, a2, b1, b2


def build_bz_grid(n_grid, a=LATTICE_CONSTANT):
    """
    Builds a uniform N_grid x N_grid mesh of the Brillouin zone in terms of
    the reciprocal lattice vectors b1, b2 (fractional coordinates i/N, j/N).

    This is the grid construction that used to be copy-pasted (with the same
    three lines) inside evaluate_fhs_on_grid, compute_direct_gap and
    compute_band_gaps_spectrum in the original single-file script.

    Returns
    -------
    KX, KY : ndarray of shape (n_grid, n_grid)
    """
    _, _, b1, b2 = get_lattice_vectors(a=a)

    i_vals = np.arange(n_grid)
    j_vals = np.arange(n_grid)
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")

    kx = (i_grid / n_grid) * b1[0] + (j_grid / n_grid) * b2[0]
    ky = (i_grid / n_grid) * b1[1] + (j_grid / n_grid) * b2[1]
    return kx, ky


def dirac_cut_grid(a=LATTICE_CONSTANT, num_pts=500):
    """
    Builds a 1D cut of k-space through the K and K' Dirac points at
    kx = 2*pi / (3a), spanning ky from -4pi/(3*sqrt(3)*a) to +4pi/(3*sqrt(3)*a).

    Returns KX, KY as column vectors of shape (num_pts, 1), matching the
    shape expected by the Hamiltonian builders (which are vectorized over
    the leading grid axes).
    """
    kx_fixed = 2.0 * np.pi / (3.0 * a)
    ky_min = -4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    ky_max = +4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)

    ky_vals = np.linspace(ky_min, ky_max, num_pts)
    kx_vals = np.full_like(ky_vals, kx_fixed)

    kx = kx_vals[:, np.newaxis]
    ky = ky_vals[:, np.newaxis]
    return kx, ky, ky_min, ky_max
