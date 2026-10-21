"""
Construction of Floquet (Sambe-space) Hamiltonians for graphene driven by a
circularly polarized laser.

Includes optimized, strictly Hermitian implementations:
  - build_floquet_hamiltonian         : exact Bessel-function blocks (main model)
  - build_floquet_hamiltonian_taylor  : first-order Taylor expansion (for comparison)
  - build_dirac_floquet_hamiltonian   : low-energy Dirac-cone approximation
"""

import numpy as np
from scipy.special import jv

from .constants import (
    H_BAR,
    ELECTRON_CHARGE,
    POLARIZATION,
    AMPLITUDE,
    HOPPING,
    LATTICE_CONSTANT,
    FERMI_VELOCITY,
)
from .lattice import get_lattice_vectors


def compute_lattice_phases(KX, KY, a=LATTICE_CONSTANT):
    """
    Precomputes the nearest-neighbor exponential phase factors for all k-points.
    This avoids redundant trigonometric/exponential evaluations across Floquet sectors.
    """
    a1, a2, _, _ = get_lattice_vectors(a=a)

    k_dot_a1 = KX * a1[0] + KY * a1[1]
    k_dot_a2 = KX * a2[0] + KY * a2[1]

    phase_a1 = np.exp(1j * k_dot_a1)
    phase_a2 = np.exp(1j * k_dot_a2)

    # Exponentials for the three nearest neighbor vectors
    exp_pos = (np.ones_like(phase_a1, dtype=complex), phase_a1, phase_a2)
    exp_neg = (np.ones_like(phase_a1, dtype=complex), np.conj(phase_a1), np.conj(phase_a2))

    return exp_pos, exp_neg


def compute_gamma_n_fast(exp_pos, exp_neg, n, z, t, pol):
    """
    Evaluates the 2x2 Floquet coupling block elements H_n(k) using precomputed phases:
      H_n(k) = [[ 0,              Gamma_n(k) ],
                [ Gamma_n_lower,  0          ]]
    where Gamma_n_lower(k) = (-1)^n * Gamma_n(-k).
    """
    phi_list = (-np.pi / 2.0, np.pi / 6.0, 5.0 * np.pi / 6.0)
    jn_val = jv(n, z)

    # Fast sum over the 3 nearest neighbors
    phase_sum_pos = sum(
        exp_pos[j] * np.exp(-1j * pol * n * phi_list[j]) for j in range(3)
    )
    gamma_n = -t * jn_val * phase_sum_pos

    if n == 0:
        # Strictly enforce zero-photon block Hermiticity
        gamma_n_lower = np.conj(gamma_n)
    else:
        phase_sum_neg = sum(
            exp_neg[j] * np.exp(-1j * pol * n * phi_list[j]) for j in range(3)
        )
        gamma_n_lower = ((-1.0) ** n) * (-t * jn_val * phase_sum_neg)

    return gamma_n, gamma_n_lower


def build_floquet_hamiltonian(
    KX,
    KY,
    N,
    omega,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    max_order=1,
    h_bar=H_BAR,
):
    """
    Constructs the Sambe Floquet matrix using exact Bessel-function blocks:
      H_F = sum_m (H_0 + m * hbar * omega * I) |m><m|
          + sum_m sum_{n=1}^{max_order} ( H_n |m><m+n| + H_n^dagger |m+n><m| )

    Parameters
    ----------
    KX, KY : np.ndarray
        Reciprocal space grid of shape (Nx, Ny).
    N : int
        Number of Floquet replicas (-N to +N), total blocks = 2*N + 1.
    omega : float
        Drive frequency.
    A_0 : float
        Laser vector potential amplitude.
    t : float
        Nearest-neighbor hopping energy.
    pol : int
        Polarization index (+1 for left-circular, -1 for right-circular).
    a : float
        Carbon-carbon bond length.
    max_order : int
        Maximum multi-photon coupling order retained.
    h_bar : float
        Reduced Planck constant.

    Returns
    -------
    H_F : np.ndarray
        Floquet Hamiltonian tensor of shape (Nx, Ny, 2*(2*N+1), 2*(2*N+1)).
    """
    dim = 2 * (2 * N + 1)
    grid_shape = KX.shape
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)

    z_param = (ELECTRON_CHARGE * A_0 * a) / h_bar
    exp_pos, exp_neg = compute_lattice_phases(KX, KY, a=a)

    # Precompute coupling blocks H_n for n = 0, 1, ..., max_order
    blocks = {}
    for n in range(max_order + 1):
        g_up, g_down = compute_gamma_n_fast(exp_pos, exp_neg, n, z_param, t, pol)
        blocks[n] = (g_up, g_down)

    num_blocks = 2 * N + 1
    photon_energy = h_bar * omega

    # Populate diagonal blocks (m = i - N)
    g0_up, g0_down = blocks[0]
    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        H_F[..., r, r + 1] = g0_up
        H_F[..., r + 1, r] = g0_down
        H_F[..., r, r] += m_index * photon_energy
        H_F[..., r + 1, r + 1] += m_index * photon_energy

    # Populate off-diagonal multi-photon coupling blocks (absorption & emission)
    for n in range(1, max_order + 1):
        gn_up, gn_down = blocks[n]

        for i in range(num_blocks - n):
            r = 2 * i
            c = 2 * (i + n)

            # Absorption block: H_n at (i, i + n)
            H_F[..., r, c + 1] = gn_up
            H_F[..., r + 1, c] = gn_down

            # Emission block: strictly H_n^\dagger at (i + n, i)
            H_F[..., c + 1, r] = np.conj(gn_up)
            H_F[..., c, r + 1] = np.conj(gn_down)

    return H_F


def build_floquet_hamiltonian_taylor(
    KX,
    KY,
    N,
    omega,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=LATTICE_CONSTANT,
    h_bar=H_BAR,
):
    """
    Constructs the tridiagonal-block Sambe Floquet matrix using first-order
    Taylor expansion in vector potential A_0.
    """
    dim = 2 * (2 * N + 1)
    grid_shape = KX.shape
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)

    exp_pos, _ = compute_lattice_phases(KX, KY, a=a)
    _, phase1, phase2 = exp_pos

    # Static tight-binding term
    gamma_0 = -t * (1.0 + phase1 + phase2)

    # First-order Taylor coupling prefactor
    prefactor = -1j * (np.sqrt(3) * ELECTRON_CHARGE * a * t * A_0) / (4.0 * h_bar)

    if pol == +1:
        term1_d, term2_d = np.sqrt(3) + 1j, np.sqrt(3) - 1j
        term1_d_prime, term2_d_prime = np.sqrt(3) - 1j, np.sqrt(3) + 1j
    else:
        term1_d, term2_d = np.sqrt(3) - 1j, np.sqrt(3) + 1j
        term1_d_prime, term2_d_prime = np.sqrt(3) + 1j, np.sqrt(3) - 1j

    gamma_d = prefactor * (term1_d * phase1 + term2_d * phase2)
    gamma_d_prime = prefactor * (term1_d_prime * phase1 + term2_d_prime * phase2)

    num_blocks = 2 * N + 1
    photon_energy = h_bar * omega

    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        # Diagonal 2x2 block
        H_F[..., r, r + 1] = gamma_0
        H_F[..., r + 1, r] = np.conj(gamma_0)
        H_F[..., r, r] += m_index * photon_energy
        H_F[..., r + 1, r + 1] += m_index * photon_energy

        # Upper block (absorption: m -> m+1)
        if i + 1 < num_blocks:
            c = 2 * (i + 1)
            H_F[..., r, c + 1] = gamma_d
            H_F[..., r + 1, c] = np.conj(gamma_d_prime)

            # Lower block (emission: m+1 -> m)
            H_F[..., c + 1, r] = np.conj(gamma_d)
            H_F[..., c, r + 1] = gamma_d_prime

    return H_F


def build_dirac_floquet_hamiltonian(
    KX,
    KY,
    N,
    omega,
    A_0=AMPLITUDE,
    v_F=FERMI_VELOCITY,
    pol=POLARIZATION,
    valley=+1,
    h_bar=H_BAR,
    e=ELECTRON_CHARGE,
):
    """
    Constructs the Sambe Floquet matrix for low-energy Dirac electrons.
    KX, KY represent relative momentum q = k - K.
    """
    dim = 2 * (2 * N + 1)
    grid_shape = KX.shape
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)

    if valley == +1:
        h0_12 = h_bar * v_F * (KX - 1j * KY)
        h0_21 = h_bar * v_F * (KX + 1j * KY)
    else:
        h0_12 = h_bar * v_F * (KX + 1j * KY)
        h0_21 = h_bar * v_F * (KX - 1j * KY)

    coupling = e * v_F * A_0
    xi = valley * pol
    num_blocks = 2 * N + 1
    photon_energy = h_bar * omega

    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        # Diagonal block
        H_F[..., r, r + 1] = h0_12
        H_F[..., r + 1, r] = h0_21
        H_F[..., r, r] += m_index * photon_energy
        H_F[..., r + 1, r + 1] += m_index * photon_energy

        # Absorption (i -> i+1)
        if i + 1 < num_blocks:
            c = 2 * (i + 1)
            if xi == +1:
                H_F[..., r + 1, c] = coupling
                H_F[..., c, r + 1] = np.conj(coupling)
            else:
                H_F[..., r, c + 1] = coupling
                H_F[..., c + 1, r] = np.conj(coupling)

    return H_F