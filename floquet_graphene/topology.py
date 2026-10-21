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
    ELECTRON_CHARGE,
    H_BAR,
)
from .lattice import build_bz_grid
from .hamiltonians import build_floquet_hamiltonian
from .gaps import compute_band_gaps_spectrum, compute_band_isolation_gap_from_evals


def evaluate_fhs_on_grid(
    w,
    N_rep,
    band_idx,
    N_grid,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    max_order=1,
    hamiltonian_builder = build_floquet_hamiltonian 
):
    """
    Evaluates Berry flux and Chern number for 'band_idx' on a fixed N_grid x N_grid mesh.

    Parameters
    ----------
    w : float
        Drive frequency omega.
    N_rep : int
        Floquet sector truncation order.
    band_idx : int
        Index of the target Floquet band.
    N_grid : int
        Discretization size per BZ reciprocal vector direction.
    A_0 : float
        Vector potential amplitude.
    t : float
        Nearest-neighbor hopping parameter.
    pol : str
        Polarization type ("circular", etc.).
    a : float
        Carbon-carbon distance.
    max_order : int
        Maximum photon coupling order retained in the Floquet Hamiltonian.

    Returns
    -------
    chern_number : int
        Quantized Chern number.
    chern_sum : float
        Raw unrounded Chern sum (Berry flux / 2*pi).
    max_flux : float
        Maximum Berry flux per plaquette |F12|.
    isolation_gap : float
        Minimum direct gap isolating band_idx from its adjacent bands across the BZ.
    """
    KX, KY = build_bz_grid(N_grid, a=a)

    H_F_all = hamiltonian_builder(
        KX, KY, N=N_rep, omega=w, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order
    )
    evals_all, evecs_all = np.linalg.eigh(H_F_all)

    # Direct isolation gap from both neighboring bands
    isolation_gap, _, _ = compute_band_isolation_gap_from_evals(evals_all, band_idx)

    # Link variables and Berry flux calculation
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
    max_flux = float(np.max(np.abs(F12)))

    chern_sum = float(np.sum(F12) / (2.0 * np.pi))
    chern_number = int(np.round(chern_sum))

    return chern_number, chern_sum, max_flux, isolation_gap


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
    max_order=1,
    hamiltonian_builder = build_floquet_hamiltonian
):
    """
    Computes the Chern number of 'band_idx' adaptively.
    Initial grid size scales inversely with frequency to resolve narrow avoided crossings.
    Returns (None, 0.0, isolation_gap, max_flux, N_grid) if:
      1. The band touches an adjacent band (isolation_gap < gap_threshold).
      2. The grid reaches max_grid without achieving stability and max_flux < flux_threshold.
    """
    if N == 0:
        return None, 0.0, 0.0, 0.0, 60

    z_param = (ELECTRON_CHARGE * A_0 * a) / H_BAR
    energy_drive = max(H_BAR * omega, 1e-4)
    energy_hop = max(t, 1e-4)

    # Initial grid scaling heuristic: lower frequency requires finer k-space sampling
    n_freq_heuristic = int(np.ceil(60.0 * (energy_hop / energy_drive)))
    n_driving_heuristic = int(np.ceil((2.0 * np.pi) / max(z_param, 1e-3)))

    estimated_n = max(n_freq_heuristic, n_driving_heuristic)
    N_grid = int(np.clip(estimated_n, 60, 200))
    step_increase = 30

    prev_c_num = None
    c_num, c_sum, max_flux, isolation_gap = 0, 0.0, 0.0, 0.0

    while N_grid <= max_grid:
        c_num, c_sum, max_flux, isolation_gap = evaluate_fhs_on_grid(
            w=omega,
            N_rep=N,
            band_idx=band_idx,
            N_grid=N_grid,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            max_order=max_order,
            hamiltonian_builder = hamiltonian_builder 
        )

        # Abort if band is degenerate or unisolated
        if isolation_gap < gap_threshold:
            print(
                f"    [FHS Abort] band={band_idx}: gap is closed or unresolvable "
                f"({isolation_gap:.2e} < {gap_threshold:.2e}) on {N_grid}x{N_grid}."
            )
            return None, 0.0, isolation_gap, max_flux, N_grid

        flux_ok = max_flux < flux_threshold
        stable_ok = (prev_c_num is not None) and (prev_c_num == c_num)

        # Successfully converged
        if flux_ok and stable_ok:
            return c_num, c_sum, isolation_gap, max_flux, N_grid

        if not flux_ok:
            print(
                f"    [FHS Refinement] band={band_idx}: max_flux={max_flux:.2f} rad >= "
                f"{flux_threshold:.2f} rad on {N_grid}x{N_grid}. Increasing grid..."
            )
        elif not stable_ok and prev_c_num is not None:
            print(
                f"    [FHS Refinement] band={band_idx}: Chern number changed "
                f"{prev_c_num:+d} -> {c_num:+d} between grids. Refining..."
            )

        prev_c_num = c_num
        N_grid += step_increase

    # If the loop completes without meeting the convergence criteria, reject the result
    print(
        f"    [Warning] band={band_idx}: reached max_grid={max_grid} without convergence. "
        f"Last candidate C={c_num}, max_flux={max_flux:.2f} rad, isolation_gap={isolation_gap:.2e}. "
        f"Returning None to prevent unphysical topological numbers."
    )
    return None, 0.0, isolation_gap, max_flux, max_grid


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
    hamiltonian_builder = build_floquet_hamiltonian
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

    # 1. Verify that the target gap above target_band is open across the entire BZ
    _, target_gap, _, _ = compute_band_gaps_spectrum(
        omega=omega,
        M=M,
        target_band=target_band,
        A_0=A_0,
        t=t,
        pol=pol,
        a=a,
        max_order=max_order,
        N_k=80,
        hamiltonian_builder = hamiltonian_builder
    )

    if target_gap is None or target_gap < MIN_RESOLVABLE_GAP:
        gap_str = f"{target_gap:.2e}" if target_gap is not None else "None"
        print(
            f"    [!] Target gap above band {target_band} is closed ({gap_str}). "
            f"Winding number undefined."
        )
        return None

    # 2. Sum Chern numbers of all isolated bands below the open gap
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
            gap_threshold=MIN_RESOLVABLE_GAP,
            max_grid=max_grid,
            max_order=max_order,
            hamiltonian_builder = hamiltonian_builder
        )

        if c_beta is None:
            print(
                f"    [!] Band {beta} could not be resolved reliably. "
                f"Aborting winding number computation."
            )
            return None

        W_sum += c_beta

    return W_sum