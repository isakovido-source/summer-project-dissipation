"""
Human-readable text reports summarizing computed invariants.
"""

from .topology import compute_floquet_chern_adaptive
from .hamiltonians import build_floquet_hamiltonian

def print_spectral_comparison_table(M, omega, A_0, t, pol, a, max_grid=360, hamiltonian_builder = build_floquet_hamiltonian):
    """
    Computes and displays the complete topological spectrum in a clear structured table.
    """
    total_bands = 2 * (2 * M + 1)
    target_band_zero = 2 * M

    print("\n" + "=" * 82)
    print(f" FLOQUET TOPOLOGICAL SPECTRUM TABLE (M={M}, omega={omega:.2f}, z={A_0*a:.2f})")
    print("=" * 82)
    print(f"{'Band':>4} | {'Replica m':>9} | {'Subband':>8} | {'Fig 3 Tag':>10} | {'Chern (C)':>10} | {'Winding (W)':>11} | {'Grid':>9}")
    print("-" * 82)

    # Standard labeling from Fig. 3 of Perez-Piskunow et al.
    fig3_labels = ["C_-4", "C_-3", "C_-2", "C_-1", "C_0", "C_1", "C_2", "C_3", "C_4", "C_5"]

    W_running = 0
    for beta in range(total_bands):
        m_replica = (beta // 2) - M
        subband = "Valence" if (beta % 2 == 0) else "Conduct"
        label = fig3_labels[beta] if beta < len(fig3_labels) else f"C_{beta}"

        c_val, _, _, _, grid_used = compute_floquet_chern_adaptive(
            N=M, 
            band_idx=beta, 
            omega=omega, 
            A_0=A_0, 
            t=t, 
            pol=pol, 
            a=a, 
            max_grid=max_grid,
            hamiltonian_builder = hamiltonian_builder
        )

        c_num = c_val if c_val is not None else 0
        W_running += c_num

        marker = " <== W(eps=0)" if beta == target_band_zero else ""
        print(f"{beta:4d} | {m_replica:9d} | {subband:>8} | {label:>10} | {c_num:+10d} | {W_running:+11d} | {grid_used:4d}x{grid_used:<4d}{marker}")

    print("=" * 82)
    return W_running
