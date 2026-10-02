"""
Plotting functions.

NOTE for future refactoring: these functions currently both compute (build
Hamiltonians, diagonalize) and plot/save. A natural next step, once you're
comfortable with the rest of the structure, is to split each of these into
a `compute_...` function that returns plain arrays and a `plot_...` function
that only draws — that makes the numbers reusable and testable without
matplotlib/plotly involved at all.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .constants import AMPLITUDE, HOPPING, POLARIZATION, LATTICE_CONSTANT, MIN_RESOLVABLE_GAP
from .lattice import get_lattice_vectors, dirac_cut_grid
from .hamiltonians import build_floquet_hamiltonian
from .gaps import compute_band_gaps_spectrum
from .topology import compute_floquet_chern_adaptive, compute_winding_number_for_band


def compute_and_save_interactive_plotly(
    N, omega, output_folder, c_lower, c_upper, A_0=AMPLITUDE, t=HOPPING, a=LATTICE_CONSTANT, grid_size=60, save_html=False
):
    kx_vals = np.linspace(-np.pi / a, np.pi / a, grid_size)
    ky_vals = np.linspace(-np.pi / a, np.pi / a, grid_size)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    H_F_all = build_floquet_hamiltonian(KX, KY, N=N, omega=omega, A_0=A_0, t=t, a=a)
    bands = np.linalg.eigvalsh(H_F_all)

    fig = go.Figure()
    c_lower_str = f"{c_lower:+d}" if c_lower is not None else "Undef"
    c_upper_str = f"{c_upper:+d}" if c_upper is not None else "Undef"

    band_lower = 2 * N
    band_upper = 2 * N + 1

    fig.add_trace(go.Surface(
        x=KX, y=KY, z=bands[:, :, band_lower],
        name=f'Central Valence Band {band_lower} (C = {c_lower_str})',
        showscale=False, colorscale='Blues', opacity=0.85
    ))

    fig.add_trace(go.Surface(
        x=KX, y=KY, z=bands[:, :, band_upper],
        name=f'Central Conduction Band {band_upper} (C = {c_upper_str})',
        showscale=False, colorscale='Reds', opacity=0.85
    ))

    title_text = f'Floquet Bands (N={N}, A0={A_0}, w={omega}, t={t})<br><b>C_val = {c_lower_str}, C_cond = {c_upper_str}</b>'
    fig.update_layout(title=title_text, scene=dict(xaxis_title='kx', yaxis_title='ky', zaxis_title='Quasienergy'), width=950, height=750)

    img_name = f"floquet_3D_N={N}_A0={A_0}_w={omega}_C_upper={c_upper_str}.png"
    img_path = os.path.join(output_folder, img_name)
    fig.write_image(img_path, scale=3)
    print(f"    Saved static 3D image: {img_path}")

    if save_html:
        file_name = f"floquet_3D_N={N}_A0={A_0}_w={omega}_C_upper={c_upper_str}.html"
        file_path = os.path.join(output_folder, file_name)
        fig.write_html(file_path)
        print(f"    Saved interactive HTML: {file_path}")


def compute_and_save_2d_dirac_cut(
    N, omega, output_folder, c_lower, c_upper, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT, num_pts=500
):
    a1, a2, _, _ = get_lattice_vectors(a=a)
    KX, KY, ky_min, ky_max = dirac_cut_grid(a=a, num_pts=num_pts)
    ky_vals = KY[:, 0]

    H_F = build_floquet_hamiltonian(KX, KY, N=N, omega=omega, A_0=A_0, t=t, pol=pol, a=a)
    evals = np.linalg.eigvalsh(H_F)[:, 0, :]

    band_lower = 2 * N
    band_upper = 2 * N + 1
    val_floq = evals[:, band_lower]
    cond_floq = evals[:, band_upper]

    k_dot_a1 = KX * a1[0] + KY * a1[1]
    k_dot_a2 = KX * a2[0] + KY * a2[1]
    gamma_0 = -t * (1.0 + np.exp(1j * k_dot_a1) + np.exp(1j * k_dot_a2))[:, 0]
    e_stat = np.abs(gamma_0)

    c_lower_str = f"{c_lower:+d}" if c_lower is not None else "Undef"
    c_upper_str = f"{c_upper:+d}" if c_upper is not None else "Undef"

    plt.figure(figsize=(9, 5.5))
    plt.plot(ky_vals, e_stat, 'k--', alpha=0.45, label="Static Conduction ($A_0=0$)")
    plt.plot(ky_vals, -e_stat, 'k--', alpha=0.45, label="Static Valence ($A_0=0$)")
    plt.plot(ky_vals, cond_floq, 'r-', linewidth=2.0, label=f"Central Conduction (C = {c_upper_str})")
    plt.plot(ky_vals, val_floq, 'b-', linewidth=2.0, label=f"Central Valence (C = {c_lower_str})")

    ky_K_prime = -2.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    ky_K = +2.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    plt.axvline(ky_K_prime, color='darkgreen', linestyle=':', linewidth=1.5, label=r"Dirac Point $K'$")
    plt.axvline(ky_K, color='purple', linestyle=':', linewidth=1.5, label=r"Dirac Point $K$")
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.title(
        f"2D Cut Through Dirac Points ($k_x = 2\\pi/3$)\n"
        f"Floquet Replica Order: N = {N}  |  $A_0={A_0},\\ \\omega={omega},\\ t={t},\\ \\sigma={pol}$\n"
        f"Chern Numbers: $C_{{val}}={c_lower_str},\\ C_{{cond}}={c_upper_str}$",
        fontsize=11
    )
    plt.xlabel(r"$k_y$", fontsize=12)
    plt.ylabel(r"Quasienergy $\varepsilon$", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()

    cut_img_name = f"floquet_2D_cut_N={N}_A0={A_0}_w={omega}_C_upper={c_upper_str}.png"
    cut_img_path = os.path.join(output_folder, cut_img_name)
    plt.savefig(cut_img_path, dpi=300)
    plt.close()
    print(f"    Saved 2D cut image: {cut_img_path}")


def plot_winding_and_gaps_sweep(
    omega_min,
    omega_max,
    num_points,
    pol,
    A_0=AMPLITUDE,
    t=HOPPING,
    max_order=1,
    N_k=70,
    save_plot=True,
    output_filename="floquet_adaptive_sweep.png"
):
    """
    Executes an adaptive-M frequency sweep.
    Calculates M dynamically per frequency: M(omega) = max(2, ceil(6*t / omega))
    to ensure full bandwidth coverage without artificial cutoff jumps.
    """
    omega_array = np.linspace(omega_min, omega_max, num_points)

    gap_0_list = []
    gap_half_list = []
    gap_minus_list = []
    w0_list = []
    whalf_list = []

    print("=" * 95)
    print(f" ADAPTIVE SWEEP: omega in [{omega_min:.3f}, {omega_max:.3f}] ({num_points} pts)")
    print(f" Parameters: A_0={A_0:.3f}, t={t:.3f}, pol={pol:+d}, max_order={max_order}")
    print("=" * 95)
    print(f"{'Idx':>3} | {'omega':>7} | {'M':>2} | {'Gap(0)':>10} | {'W(0)':>7} | {'Gap(w/2)':>10} | {'W(w/2)':>7}")
    print("-" * 95)

    for idx, w in enumerate(omega_array):
        # Adaptive M calculation based on the full physical bandwidth 6*t
        current_M = max(2, int(math.ceil(6.0 * t / w)))

        band_val_0 = 2 * current_M
        band_cond_0 = 2 * current_M + 1

        # Evaluate physical band gaps across the Brillouin zone
        gap_lower, gap_0, _, _ = compute_band_gaps_spectrum(
            omega=w, M=current_M, target_band=band_val_0, A_0=A_0, t=t, pol=pol,
            a=LATTICE_CONSTANT, max_order=max_order, N_k=N_k
        )
        _, gap_upper, _, _ = compute_band_gaps_spectrum(
            omega=w, M=current_M, target_band=band_cond_0, A_0=A_0, t=t, pol=pol,
            a=LATTICE_CONSTANT, max_order=max_order, N_k=N_k
        )

        gap_0_list.append(gap_0)
        gap_half_list.append(gap_upper)
        gap_minus_list.append(gap_lower)

        # Compute topological winding numbers
        w0 = compute_winding_number_for_band(
            target_band=band_val_0, omega=w, A_0=A_0, t=t, pol=pol,
            a=LATTICE_CONSTANT, M=current_M, max_order=max_order
        )
        wh = compute_winding_number_for_band(
            target_band=band_cond_0, omega=w, A_0=A_0, t=t, pol=pol,
            a=LATTICE_CONSTANT, M=current_M, max_order=max_order
        )

        w0_list.append(w0)
        whalf_list.append(wh)

        s_w0 = f"{w0:+d}" if w0 is not None else "Closed"
        s_wh = f"{wh:+d}" if wh is not None else "Closed"

        if (idx + 1) % 5 == 0 or idx == 0 or idx == num_points - 1:
            print(f"{idx+1:3d} | {w:7.4f} | {current_M:2d} | {gap_0:10.2e} | {s_w0:>7} | {gap_upper:10.2e} | {s_wh:>7}")

    gap_0_arr = np.array(gap_0_list)
    gap_half_arr = np.array(gap_half_list)
    gap_minus_arr = np.array(gap_minus_list)
    w0_arr = np.array([np.nan if v is None else float(v) for v in w0_list])
    wh_arr = np.array([np.nan if v is None else float(v) for v in whalf_list])

    # Detect topological transitions based on winding steps
    trans_w0_idx = np.where(np.diff(w0_arr) != 0)[0]
    trans_wh_idx = np.where(np.diff(wh_arr) != 0)[0]

    # Plot aligned double-panel figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 8.5), sharex=True,
        gridspec_kw={'height_ratios': [1.1, 1.0]}
    )

    # Top Panel: Direct Gaps
    ax1.plot(omega_array, gap_0_arr, color='navy', linestyle='-', linewidth=1.6, label=r'Central Gap $\Delta_0$ ($\varepsilon \approx 0$)')
    ax1.plot(omega_array, gap_half_arr, color='crimson', linestyle='--', linewidth=1.5, label=r'Upper Gap $\Delta_{+}$ ($\varepsilon \approx +\hbar\Omega/2$)')
    ax1.plot(omega_array, gap_minus_arr, color='forestgreen', linestyle=':', linewidth=1.3, alpha=0.8, label=r'Lower Gap $\Delta_{-}$ ($\varepsilon \approx -\hbar\Omega/2$)')
    ax1.axhline(MIN_RESOLVABLE_GAP, color='gray', linestyle=':', label='Threshold')
    ax1.set_ylabel(r"Direct Bandgap $\Delta$", fontsize=12)
    ax1.set_title(
        f"Adaptive Floquet Topological Transitions vs $\\omega$\n"
        f"($M(\\omega) = \\lceil 6t/\\omega \\rceil,\\ A_0 = {A_0},\\ t = {t},\\ \\sigma = {pol:+d}$)",
        fontsize=13
    )
    ax1.grid(True, linestyle=':', alpha=0.55)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Bottom Panel: Winding Numbers
    ax2.step(omega_array, w0_arr, where='mid', color='navy', linewidth=1.8, label=r'$W(\varepsilon = 0)$')
    ax2.step(omega_array, wh_arr, where='mid', color='crimson', linewidth=1.8, label=r'$W(\varepsilon = \hbar\Omega/2)$')
    ax2.set_xlabel(r"Driving Frequency $\omega$", fontsize=12)
    ax2.set_ylabel(r"Winding Number $W$", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.55)
    ax2.legend(loc='best', framealpha=0.9)

    # Vertical markers at gap closings
    for i in trans_w0_idx:
        w_crit = 0.5 * (omega_array[i] + omega_array[i+1])
        ax1.axvline(w_crit, color='navy', linestyle=':', alpha=0.45)
        ax2.axvline(w_crit, color='navy', linestyle=':', alpha=0.45)

    for i in trans_wh_idx:
        w_crit = 0.5 * (omega_array[i] + omega_array[i+1])
        ax1.axvline(w_crit, color='crimson', linestyle=':', alpha=0.45)
        ax2.axvline(w_crit, color='crimson', linestyle=':', alpha=0.45)

    plt.tight_layout()
    if save_plot:
        plt.savefig(output_filename, dpi=300)
        print(f"[\u2713] Saved adaptive sweep figure to: {output_filename}")

    plt.show()


def plot_gap_and_chern_scan(scan_results, pol, save_plot=False, output_filename="floquet_gap_chern_scan.png"):
    """
    Plots the output of runners.run_scan_with_all_gaps: direct gaps (central,
    below, above) in one panel and the valence-band Chern number in another.

    This is a worked example of the compute/plot split mentioned at the top
    of this file: run_scan_with_all_gaps() only computes and returns numbers,
    this function only draws them.
    """
    omega_array = np.asarray(scan_results['omega'])
    chern_arr = np.array([np.nan if c is None else float(c) for c in scan_results['chern']])
    gap_below_arr = np.array(scan_results['gap_below'])
    gap_central_arr = np.array(scan_results['gap_central'])
    gap_above_arr = np.array(scan_results['gap_above'])
    is_undefined = np.isnan(chern_arr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})

    # Subplot 1: All three direct gaps
    ax1.plot(omega_array, gap_central_arr, 'k.-', markersize=4, alpha=0.85, label=r'Central Gap $\Delta_0$ ($\varepsilon \approx 0$)')
    ax1.plot(omega_array, gap_below_arr, 'b.-', markersize=4, alpha=0.65, label=r'Gap Below $\Delta_{-}$ ($\varepsilon \approx -\omega/2$)')
    ax1.plot(omega_array, gap_above_arr, 'r--', markersize=3, alpha=0.55, label=r'Gap Above $\Delta_{+}$ ($\varepsilon \approx +\omega/2$)')
    ax1.axhline(MIN_RESOLVABLE_GAP, color='gray', linestyle=':', label=f'Min Resolvable Gap ({MIN_RESOLVABLE_GAP:.1e})')

    ax1.set_ylabel(r'Direct Gap $\Delta$', fontsize=12)
    ax1.set_title(fr'Floquet Band Gaps and Chern Invariant $\sigma={pol:+d}$ (Valence Band) $C(\omega)$', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Subplot 2: Chern Number
    ax2.plot(omega_array, chern_arr, 'o-', color='navy', markersize=5, linewidth=1.4, label=r'Computed $C_{\text{val}}$')
    nan_omegas = omega_array[is_undefined]
    if len(nan_omegas) > 0:
        ax2.scatter(nan_omegas, np.zeros_like(nan_omegas), color='crimson', marker='x', s=50, zorder=5, label='Undefined / Unresolved')

    ax2.set_xlabel(r'Driving Frequency $\omega$', fontsize=12)
    ax2.set_ylabel(r'Chern Number $C$', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    if save_plot:
        plt.savefig(output_filename, dpi=300)
        print(f"[\u2713] Saved gap/Chern scan figure to: {output_filename}")
    plt.show()


def plot_floquet_spectrum_with_invariants(
    omega,
    pol,
    A_0=AMPLITUDE,
    t=HOPPING,
    max_order=1,
    num_k_pts=500,
    save_plot=True,
    output_filename=None
):
    """
    Plots the full 2D band spectrum along the K - K' Dirac cut for a given omega.
    Cutoff M is automatically computed adaptively: M = ceil(6*t / omega).
    Displays Chern numbers for each subband and winding numbers for each gap.
    """
    a = LATTICE_CONSTANT
    M = max(2, int(math.ceil(6.0 * t / omega)))
    total_bands = 2 * (2 * M + 1)

    if output_filename is None:
        output_filename = f"floquet_bands_annotated_w={omega:.3f}_M={M}.png"

    print("=" * 80)
    print(f" COMPUTING ANNOTATED 2D SPECTRUM: omega = {omega:.3f} | Adaptive M = {M} ({total_bands} bands)")
    print("=" * 80)

    KX, KY, ky_min, ky_max = dirac_cut_grid(a=a, num_pts=num_k_pts)
    ky_vals = KY[:, 0]

    H_F = build_floquet_hamiltonian(
        KX, KY, N=M, omega=omega, A_0=A_0, t=t, pol=pol, a=a, max_order=max_order
    )
    evals = np.linalg.eigvalsh(H_F)[:, 0, :]

    # Compute Chern numbers and running Winding numbers
    chern_list = []
    w_running = 0
    winding_list = []

    for beta in range(total_bands):
        c_val, _, _, _, _ = compute_floquet_chern_adaptive(
            N=M, band_idx=beta, omega=omega, A_0=A_0, t=t, pol=pol, a=a, max_grid=300
        )
        c_num = c_val if c_val is not None else 0
        chern_list.append(c_num)
        w_running += c_num
        winding_list.append(w_running)
        print(f"Band {beta:2d}: Chern C = {c_num:+2d} | Gap above: W = {w_running:+2d}")

    # Plotting
    plt.figure(figsize=(10.5, 7.5))

    # Static Dirac cones for reference
    a1, a2, _, _ = get_lattice_vectors(a=a)
    k_dot_a1 = KX * a1[0] + KY * a1[1]
    k_dot_a2 = KX * a2[0] + KY * a2[1]
    gamma_0 = -t * (1.0 + np.exp(1j * k_dot_a1) + np.exp(1j * k_dot_a2))[:, 0]
    e_stat = np.abs(gamma_0)
    plt.plot(ky_vals, e_stat, 'k--', alpha=0.35, label="Static Graphene ($A_0=0$)")
    plt.plot(ky_vals, -e_stat, 'k--', alpha=0.35)

    # Plot all Floquet bands
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, total_bands))
    for b in range(total_bands):
        lbl = f"Band {b} (C={chern_list[b]:+d})" if (b == 2*M or b == 2*M+1) else None
        plt.plot(ky_vals, evals[:, b], color=colors[b], linewidth=1.8, label=lbl)

        # Annotate Chern numbers on the right side
        e_end = evals[-1, b]
        plt.text(ky_max * 1.02, e_end, f"C={chern_list[b]:+d}", verticalalignment='center', fontsize=9, color=colors[b], fontweight='bold')

        # Annotate winding number inside the gap above the band
        if b < total_bands - 1:
            e_gap = 0.5 * (evals[-1, b] + evals[-1, b + 1])
            is_central = (b == 2 * M)
            tag = f" W={winding_list[b]:+d}" + (" (eps=0)" if is_central else "")
            plt.text(ky_max * 1.15, e_gap, tag, verticalalignment='center', fontsize=9, color='black')

    # Dirac point vertical lines
    ky_K_prime = -2.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    ky_K = +2.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    plt.axvline(ky_K_prime, color='green', linestyle=':', label=r"Dirac $K'$")
    plt.axvline(ky_K, color='purple', linestyle=':', label=r"Dirac $K$")
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.7)

    plt.xlabel(r"Momentum $k_y$", fontsize=12)
    plt.ylabel(r"Quasienergy $\varepsilon$", fontsize=12)
    plt.title(f"Floquet Spectrum & Topological Invariants ($K-K'$ Cut)\n($\\omega={omega},\\ M={M},\\ A_0={A_0},\\ t={t},\\ \\sigma={pol:+d}$)", fontsize=12)
    plt.xlim(ky_min, ky_max * 1.35)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()

    if save_plot:
        plt.savefig(output_filename, dpi=300)
        print(f"[\u2713] Saved annotated spectrum to: {output_filename}")

    plt.show()
