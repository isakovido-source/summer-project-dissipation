"""
Direct band-gap calculations across the Brillouin zone, and locating the
drive frequency at which a gap closes (a topological transition).
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import AMPLITUDE, HOPPING, POLARIZATION, LATTICE_CONSTANT
from .lattice import build_bz_grid
from .hamiltonians import build_floquet_hamiltonian


def compute_direct_gap(
    omega_val,
    N,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    max_order=1,
    N_k=120,
    hamiltonian_builder = build_floquet_hamiltonian
):
    """
    Computes the minimum direct gap across the entire BZ between bands 2N and 2N+1.
    """
    KX, KY = build_bz_grid(N_k, a=a)

    H_F = hamiltonian_builder(
        KX, KY, N=N, omega=omega_val, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order
    )
    evals = np.linalg.eigvalsh(H_F)

    band_val = 2 * N
    band_cond = 2 * N + 1
    gaps = evals[..., band_cond] - evals[..., band_val]

    min_idx = np.unravel_index(np.argmin(gaps), gaps.shape)
    return float(gaps[min_idx]), (float(KX[min_idx]), float(KY[min_idx]))


def find_topological_critical_frequency(
    omega_bracket,
    N,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    max_order=1,
    tol=1e-5,
    N_k_base=100,
    hamiltonian_builder = build_floquet_hamiltonian
):
    """
    Finds the exact transition frequency omega_c where the direct gap closes.
    """
    omega_a, omega_b = omega_bracket
    print(f"[*] Starting search for gap closing in [{omega_a:.6f}, {omega_b:.6f}] (N={N})...")

    def objective(w, n_grid):
        gap, _ = compute_direct_gap(
            w, N=N, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order, N_k=n_grid, hamiltonian_builder = hamiltonian_builder
        )
        return gap

    res = minimize_scalar(
        lambda w: objective(w, N_k_base),
        bracket=(omega_a, (omega_a + omega_b) / 2.0, omega_b),
        bounds=omega_bracket,
        method="bounded",
        options={"xatol": tol},
    )

    omega_candidate = float(res.x)
    N_k_fine = int(N_k_base * 1.8)
    final_gap, k_closing = compute_direct_gap(
        omega_candidate, N=N, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order, N_k=N_k_fine
    )

    print("[\u2713] Transition identified:")
    print(f"    Critical frequency omega_c : {omega_candidate:.6f}")
    print(f"    Minimum gap at crossing   : {final_gap:.2e}")
    print(f"    Closing momentum point k*  : ({k_closing[0]:.4f}, {k_closing[1]:.4f})")

    return omega_candidate, final_gap, k_closing


def compute_band_gaps_spectrum(
    omega,
    M,
    target_band,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    max_order=1,
    N_k=100,
    hamiltonian_builder = build_floquet_hamiltonian
):
    """
    Computes direct band gaps across the entire BZ immediately below and above 'target_band'.

    Returns
    -------
    gap_below    : float or None (gap to target_band - 1, None if target_band == 0)
    gap_above    : float or None (gap to target_band + 1, None if target_band == total_bands - 1)
    k_min_below  : tuple (kx, ky) of the minimum gap location below
    k_min_above  : tuple (kx, ky) of the minimum gap location above
    """
    KX, KY = build_bz_grid(N_k, a=a)

    H_F = hamiltonian_builder(
        KX, KY, N=M, omega=omega, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order
    )
    evals = np.linalg.eigvalsh(H_F)
    total_bands = evals.shape[-1]

    # 1. Gap below (target_band - (target_band - 1))
    if target_band > 0:
        diff_below = evals[..., target_band] - evals[..., target_band - 1]
        min_idx_b = np.unravel_index(np.argmin(diff_below), diff_below.shape)
        gap_below = float(diff_below[min_idx_b])
        k_min_below = (float(KX[min_idx_b]), float(KY[min_idx_b]))
    else:
        gap_below, k_min_below = None, None

    # 2. Gap above ((target_band + 1) - target_band)
    if target_band + 1 < total_bands:
        diff_above = evals[..., target_band + 1] - evals[..., target_band]
        min_idx_a = np.unravel_index(np.argmin(diff_above), diff_above.shape)
        gap_above = float(diff_above[min_idx_a])
        k_min_above = (float(KX[min_idx_a]), float(KY[min_idx_a]))
    else:
        gap_above, k_min_above = None, None

    return gap_below, gap_above, k_min_below, k_min_above


def compute_band_isolation_gap_from_evals(evals, band_idx):
    """
    Extracts the minimum direct gap that isolates 'band_idx' from its neighbors
    using pre-diagonalized eigenvalues across the Brillouin zone.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalue tensor of shape (..., total_bands), sorted ascendingly.
    band_idx : int
        Index of the target band.

    Returns
    -------
    isolation_gap : float
        min(gap_below, gap_above).
    gap_below : float or None
        Minimum direct gap to band_idx - 1 across all k-points.
    gap_above : float or None
        Minimum direct gap to band_idx + 1 across all k-points.
    """
    total_bands = evals.shape[-1]
    if band_idx < 0 or band_idx >= total_bands:
        raise IndexError(f"band_idx {band_idx} is out of bounds for total_bands={total_bands}")

    gap_below = None
    if band_idx > 0:
        gap_below = float(np.min(evals[..., band_idx] - evals[..., band_idx - 1]))

    gap_above = None
    if band_idx + 1 < total_bands:
        gap_above = float(np.min(evals[..., band_idx + 1] - evals[..., band_idx]))

    if gap_below is not None and gap_above is not None:
        isolation_gap = min(gap_below, gap_above)
    elif gap_below is not None:
        isolation_gap = gap_below
    elif gap_above is not None:
        isolation_gap = gap_above
    else:
        isolation_gap = 0.0

    return isolation_gap, gap_below, gap_above