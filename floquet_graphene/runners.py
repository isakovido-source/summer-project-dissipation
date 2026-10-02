"""
High-level "recipes" that combine the building blocks in the rest of the
package to answer a specific physics question (e.g. "what are the Chern
numbers at this omega", "how do they change across a frequency sweep").

These are the functions your scripts/ call. If you find yourself writing a
new one-off calculation, it belongs here (or in a new script), not inside
hamiltonians.py / topology.py / gaps.py, which should stay generic building
blocks.
"""

import os
import math

import numpy as np

from .constants import AMPLITUDE, HOPPING, POLARIZATION, LATTICE_CONSTANT, MIN_RESOLVABLE_GAP
from .topology import compute_floquet_chern_adaptive
from .gaps import compute_band_gaps_spectrum
from .plotting import compute_and_save_interactive_plotly, compute_and_save_2d_dirac_cut


def run_calculate_chern(target_omega, N_replicas=None, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT):
    """
    Calculates and prints the Chern numbers for central valence and conduction bands.
    """
    # Total static bandwidth for graphene tight-binding model
    W = 6.0 * t

    # Adaptive Floquet cutoff calculation
    if N_replicas is None:
        N_replicas = max(1, math.ceil(W / target_omega))

    band_val = 2 * N_replicas
    band_cond = 2 * N_replicas + 1

    print("=" * 60)
    print(f" Computing Chern Numbers for omega = {target_omega:.4f} (N={N_replicas})")
    print("=" * 60)

    # 1. Valence band
    c_val, _, gap_val, flux_val, grid_val = compute_floquet_chern_adaptive(
        N=N_replicas, band_idx=band_val, omega=target_omega, A_0=A_0, t=t, pol=pol, a=a
    )

    # 2. Conduction band
    c_cond, _, gap_cond, flux_cond, grid_cond = compute_floquet_chern_adaptive(
        N=N_replicas, band_idx=band_cond, omega=target_omega, A_0=A_0, t=t, pol=pol, a=a
    )

    print("\n" + "-" * 40)
    print(f"Results for omega = {target_omega:.4f}:")
    print("-" * 40)
    c_val_str = f"{c_val:+d}" if c_val is not None else "Undefined (Gapless)"
    c_cond_str = f"{c_cond:+d}" if c_cond is not None else "Undefined (Gapless)"

    print(f"Valence Band ({band_val})   : C = {c_val_str} (Direct Gap = {gap_val:.4e}, Grid = {grid_val}x{grid_val})")
    print(f"Conduction Band ({band_cond}): C = {c_cond_str} (Direct Gap = {gap_cond:.4e}, Grid = {grid_cond}x{grid_cond})")

    return c_val, c_cond


def run_generate_graphs(omega_val, results_dir, N_replicas=None, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT):
    """
    Generates the interactive 3D band plot and the 2D Dirac cut for one (or every)
    Floquet replica order N, saving them under results_dir.

    Note: this used to compute its own output path from __file__; now the
    caller (a script) decides where results go, which keeps this function
    reusable and testable.
    """
    folder_name = f"floquet_results_A0={A_0}_w={omega_val}_t={t}_pol={pol}"
    full_output_path = os.path.join(results_dir, folder_name)
    os.makedirs(full_output_path, exist_ok=True)

    W = 6.0 * t
    if N_replicas is None:
        N_list = range(math.ceil(W / omega_val) + 1)
    else:
        N_list = [N_replicas]

    for N in N_list:
        print(f"=== Processing N = {N} ===")
        band_lower = 2 * N
        band_upper = 2 * N + 1

        c_lower, _, _, _, _ = compute_floquet_chern_adaptive(N, band_lower, omega=omega_val, A_0=A_0, t=t, pol=pol, a=a)
        c_upper, _, _, _, _ = compute_floquet_chern_adaptive(N, band_upper, omega=omega_val, A_0=A_0, t=t, pol=pol, a=a)

        compute_and_save_interactive_plotly(N, omega_val, full_output_path, c_lower, c_upper, A_0=A_0, t=t, a=a, save_html=False)
        compute_and_save_2d_dirac_cut(N, omega_val, full_output_path, c_lower, c_upper, A_0=A_0, t=t, pol=pol, a=a)


def run_omega_sweep(omega_min, omega_max, num_points=50, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION):
    """
    Sweeps drive frequency across a specified interval to detect topological transitions.
    Tracks Chern numbers, direct gaps, and required grid resolutions.
    """
    omega_vals_arr = np.linspace(omega_min, omega_max, num_points)
    results = []

    print("=" * 80)
    print(f"{'omega':>10} | {'N':>3} | {'C_val':>8} | {'C_cond':>8} | {'Gap (Val)':>12} | {'Grid':>9}")
    print("=" * 80)

    for w in omega_vals_arr:
        # Determine adaptive Floquet replica cutoff
        W = 6.0 * t
        N_replicas = max(1, int(np.ceil(W / w)))
        band_val = 2 * N_replicas
        band_cond = 2 * N_replicas + 1

        # 1. Valence band Chern calculation
        c_val, _, gap_val, flux_val, grid_val = compute_floquet_chern_adaptive(
            N=N_replicas, band_idx=band_val, omega=w, A_0=A_0, t=t, pol=pol
        )

        # 2. Conduction band Chern calculation
        c_cond, _, gap_cond, flux_cond, grid_cond = compute_floquet_chern_adaptive(
            N=N_replicas, band_idx=band_cond, omega=w, A_0=A_0, t=t, pol=pol
        )

        c_val_str = f"{c_val:+d}" if c_val is not None else "Gapless"
        c_cond_str = f"{c_cond:+d}" if c_cond is not None else "Gapless"

        print(f"{w:10.6f} | {N_replicas:3d} | {c_val_str:>8} | {c_cond_str:>8} | {gap_val:12.4e} | {grid_val:4d}x{grid_val:<4d}")

        results.append({
            'omega': w,
            'N': N_replicas,
            'c_val': c_val,
            'c_cond': c_cond,
            'gap_val': gap_val,
            'grid': grid_val
        })

    return results


def run_fast_chern_sweep(omega_array, max_grid_cap=220, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT):
    """
    Sweeps through omega values with bounded grid refinement to prevent freezing near gap closings.
    """
    W = 6.0 * t

    print("=" * 80)
    print(f"{'omega':>10} | {'N':>3} | {'C_val':>8} | {'C_cond':>8} | {'Central Gap':>13} | {'Grid':>9}")
    print("=" * 80)

    for w in omega_array:
        N_rep = max(2, int(np.ceil(W / w)))
        b_val = 2 * N_rep
        b_cond = 2 * N_rep + 1

        _, direct_gap, _, _ = compute_band_gaps_spectrum(
            omega=w, M=N_rep, target_band=b_val, A_0=A_0, t=t, pol=pol, a=a, N_k=70
        )

        # Compute valence band Chern with capped grid refinement
        c_v, _, _, _, grid_v = compute_floquet_chern_adaptive(
            N=N_rep, band_idx=b_val, omega=w, A_0=A_0, t=t, pol=pol, a=a,
            flux_threshold=0.85 * np.pi, max_grid=max_grid_cap
        )

        # Compute conduction band Chern with capped grid refinement
        c_c, _, _, _, grid_c = compute_floquet_chern_adaptive(
            N=N_rep, band_idx=b_cond, omega=w, A_0=A_0, t=t, pol=pol, a=a,
            flux_threshold=0.85 * np.pi, max_grid=max_grid_cap
        )

        c_v_str = f"{c_v:+d}" if c_v is not None else "Gapless"
        c_c_str = f"{c_c:+d}" if c_c is not None else "Gapless"
        grid_max = max(grid_v, grid_c)

        print(f"{w:10.5f} | {N_rep:3d} | {c_v_str:>8} | {c_c_str:>8} | {direct_gap:13.4e} | {grid_max:4d}x{grid_max:<4d}")


def run_scan_with_all_gaps(omega_min, omega_max, num_points, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT):
    """
    Executes frequency sweep and returns direct gaps (central, below, above)
    alongside the Chern number, for the caller to plot or inspect.
    """
    W = 6.0 * t
    omega_array = np.linspace(omega_min, omega_max, num_points)

    chern_results = []
    gap_below_results = []
    gap_central_results = []
    gap_above_results = []

    print(f"[*] Commencing sweep across {num_points} points in [{omega_min:.2f}, {omega_max:.2f}]...")

    for idx, w in enumerate(omega_array):
        N_rep = max(1, int(math.ceil(W / w)))
        band_val = 2 * N_rep

        # Compute topological invariant for the valence band
        c_num, _, _, _, grid_pts = compute_floquet_chern_adaptive(
            N=N_rep, band_idx=band_val, omega=w, A_0=A_0, t=t, pol=pol, a=a, max_grid=280
        )
        chern_results.append(c_num)

        # Compute gaps below and above valence band
        g_below, g_central, _, _ = compute_band_gaps_spectrum(
            omega=w, M=N_rep, target_band=band_val, A_0=A_0, t=t, pol=pol, a=a, N_k=100
        )
        # Compute gap above conduction band
        _, g_above, _, _ = compute_band_gaps_spectrum(
            omega=w, M=N_rep, target_band=band_val + 1, A_0=A_0, t=t, pol=pol, a=a, N_k=100
        )

        gap_below_results.append(g_below)
        gap_central_results.append(g_central)
        gap_above_results.append(g_above)

        status = f"{c_num:+d}" if c_num is not None else "UNDEFINED"
        print(f"    [{idx + 1:3d}/{num_points}] w = {w:.4f} | N = {N_rep:2d} | C = {status:>10} | Gap_central = {g_central:.2e} | Gap_below = {g_below:.2e} | Grid = {grid_pts}x{grid_pts}")

    return {
        'omega': omega_array,
        'chern': chern_results,
        'gap_below': gap_below_results,
        'gap_central': gap_central_results,
        'gap_above': gap_above_results,
    }
