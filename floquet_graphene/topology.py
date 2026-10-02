"""
Topological invariants: Berry-flux / Chern number via the Fukui-Hatsugai-Suzuki
(FHS) lattice method, and the winding number of a quasienergy gap.
"""

import math

import numpy as np

from .constants import (
    AMPLITUDE,
    HOPPING,
    POLARIZATION,
    LATTICE_CONSTANT,
    FLUX_THRESHOLD,
    MIN_RESOLVABLE_GAP,
)
from .lattice import build_bz_grid
from .hamiltonians import build_floquet_hamiltonian
from .gaps import compute_band_gaps_spectrum


def evaluate_fhs_on_grid(
    w,
    N_rep,
    band_idx,
    N_grid,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
):
    """Evaluates Berry flux and Chern number for 'band_idx' on a fixed N_grid x N_grid mesh."""
    KX, KY = build_bz_grid(N_grid, a=a)

    H_F_all = build_floquet_hamiltonian(KX, KY, N=N_rep, omega=w, A_0=A_0, t=t, pol=pol, a=a, max_order=1)
    evals_all, evecs_all = np.linalg.eigh(H_F_all)

    total_bands = evals_all.shape[-1]

    # Gap check: Edge bands (0 and total_bands-1) touch due to truncation.
    # For internal bands, we check the gap to the adjacent band.
    if band_idx + 1 < total_bands:
        direct_bandgap = np.min(evals_all[..., band_idx + 1] - evals_all[..., band_idx])
    else:
        direct_bandgap = np.min(evals_all[..., band_idx] - evals_all[..., band_idx - 1])

    # Link variables and Berry flux
    u_grid = evecs_all[..., band_idx]
    u_next_x = np.roll(u_grid, shift=-1, axis=0)
    u_next_y = np.roll(u_grid, shift=-1, axis=1)

    overlap_x = np.sum(np.conj(u_grid) * u_next_x, axis=-1)
    overlap_y = np.sum(np.conj(u_grid) * u_next_y, axis=-1)

    Ux = overlap_x / (np.abs(overlap_x) + 1e-12)
    Uy = overlap_y / (np.abs(overlap_y) + 1e-12)

    Ux_next_y = np.roll(Ux, shift=-1, axis=1)
    Uy_next_x = np.roll(Uy, shift=-1, axis=0)
    W_plaq = Ux * Uy_next_x * np.conj(Ux_next_y) * np.conj(Uy)

    F12 = np.angle(W_plaq)
    max_flux = np.max(np.abs(F12))

    chern_sum = np.sum(F12) / (2.0 * np.pi)
    chern_number = int(np.round(chern_sum))

    return chern_number, chern_sum, max_flux, direct_bandgap


def compute_floquet_chern_adaptive(
    N,
    band_idx,
    omega,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    flux_threshold=FLUX_THRESHOLD,
    gap_threshold=MIN_RESOLVABLE_GAP,
    max_grid=360,
):
    """
    Computes the Chern number of 'band_idx' adaptively.
    Initial grid is analytically evaluated from FHS admissibility condition
    |F12| < pi using full physical constants.
    Aborts immediately if direct_gap < gap_threshold.
    """
    from .constants import ELECTRON_CHARGE, H_BAR  # local import avoids a top-level cycle risk

    if N == 0:
        return None, 0.0, 0.0, 0.0, 60

    # Dimensionless driving intensity z
    z_param = (ELECTRON_CHARGE * A_0 * a) / H_BAR

    # Condition 1: Direct momentum sampling of avoided crossings
    n_crossings = (2.0 * np.pi) / max(z_param, 1e-4)

    # Condition 2: FHS admissibility on Berry curvature peak
    factor = np.sqrt((8 * np.pi) / (27 * np.sqrt(3)))  # factor ~= 0.733
    n_admissibility = factor * (H_BAR * omega / max(t, 1e-4) * max(z_param**2, 1e-4))

    # Initial grid determination from the most restrictive analytical condition
    estimated_n = int(np.ceil(max(n_crossings, n_admissibility)))
    N_grid = int(np.clip(estimated_n, 60, max_grid))
    step_increase = 40

    while N_grid <= max_grid:
        c_num, c_sum, max_flux, direct_gap = evaluate_fhs_on_grid(
            w=omega, N_rep=N, band_idx=band_idx, N_grid=N_grid, A_0=A_0, t=t, pol=pol, a=a
        )

        # Abort immediately if the band touches an adjacent band
        if direct_gap < gap_threshold:
            return None, 0.0, direct_gap, max_flux, N_grid

        # Check FHS admissibility condition (|F12| < flux_threshold < pi)
        if max_flux < flux_threshold:
            return c_num, c_sum, direct_gap, max_flux, N_grid

        print(
            f"    [FHS Refinement] band={band_idx}: max_flux={max_flux:.2f} rad >= "
            f"{flux_threshold:.2f} rad on {N_grid}x{N_grid}. Increasing grid..."
        )
        N_grid += step_increase

    print(f"    [Warning] Reached max_grid={max_grid} for band {band_idx} with max_flux={max_flux:.2f} rad.")
    return c_num, c_sum, direct_gap, max_flux, N_grid


def compute_winding_number_for_band(
    target_band,
    omega,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    M=None,
    max_grid=300,
    max_order=1,
):
    """
    Computes the winding number W above band 'target_band':
      W = sum_{beta = 0}^{target_band} C_{beta}
    Verifies that the target gap is open using compute_band_gaps_spectrum.
    """
    if M is None:
        M = max(2, int(math.ceil(6.0 * t / omega)))

    total_bands = 2 * (2 * M + 1)
    if target_band < 0 or target_band >= total_bands:
        raise ValueError(
            f"target_band {target_band} is invalid for M={M} (total bands available: 0 to {total_bands - 1})"
        )

    # 1. Verify that the target gap is open using compute_band_gaps_spectrum
    _, target_gap, _, _ = compute_band_gaps_spectrum(
        omega=omega, M=M, target_band=target_band, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order, N_k=80
    )

    if target_gap is None or target_gap < MIN_RESOLVABLE_GAP:
        gap_str = f"{target_gap:.2e}" if target_gap is not None else "None"
        print(f"    [!] Target gap above band {target_band} is closed ({gap_str}). Winding number undefined.")
        return None

    # 2. Sum Chern numbers of all bands below the open gap
    W_sum = 0
    for beta in range(target_band + 1):
        c_beta, _, _, _, _ = compute_floquet_chern_adaptive(
            N=M,
            band_idx=beta,
            omega=omega,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            gap_threshold=0.0,
            max_grid=max_grid,
        )

        if c_beta is None:
            c_beta = 0

        W_sum += c_beta

    return W_sum
