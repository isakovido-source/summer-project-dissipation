"""
Construction of Floquet (Sambe-space) Hamiltonians for graphene driven by a
circularly polarized laser.

Three variants, kept side by side because they were derived and cross-checked
against each other in the report:
  - build_floquet_hamiltonian         : exact Bessel-function blocks (main model)
  - build_floquet_hamiltonian_taylor  : first-order Taylor expansion (for comparison)
  - build_dirac_floquet_hamiltonian   : low-energy Dirac-cone approximation

All functions are pure: given the same arguments they always return the same
array, with no hidden global state besides the physical constants h_bar and e
(which are true universal constants, not "settings").
"""

import numpy as np
from scipy.special import jv

from .constants import H_BAR, ELECTRON_CHARGE, POLARIZATION, AMPLITUDE, HOPPING, LATTICE_CONSTANT, FERMI_VELOCITY
from .lattice import get_lattice_vectors


def compute_gamma_n(KX, KY, n, z, t, pol, a):
    """
    Computes Gamma_n(k) and (-1)^n * Gamma_n(-k) based on the LyX derivation:
    Gamma_n(k) = -t * J_n(z) * sum_j exp(i*k.delta_j) * exp(-i*pol*n*phi_j)
    """
    a1, a2, _, _ = get_lattice_vectors(a=a)

    # Dot products for +k
    k_dot_a1 = KX * a1[0] + KY * a1[1]
    k_dot_a2 = KX * a2[0] + KY * a2[1]

    # Dot products for -k (needed for the B->A element (-1)^n * Gamma_n(-k))
    minus_k_dot_a1 = -k_dot_a1
    minus_k_dot_a2 = -k_dot_a2

    # Exponentials for the 3 neighbors: delta_1 = (-a, 0), delta_2 = delta_1 + a1, delta_3 = delta_1 + a2
    exp_k_pos = [1.0, np.exp(1j * k_dot_a1), np.exp(1j * k_dot_a2)]
    exp_k_neg = [1.0, np.exp(1j * minus_k_dot_a1), np.exp(1j * minus_k_dot_a2)]

    # Angles phi_j in {-pi/2, pi/6, 5*pi/6}
    phi_list = [-np.pi / 2.0, np.pi / 6.0, 5.0 * np.pi / 6.0]

    jn_val = jv(n, z)

    # 1. Gamma_n(k) - hopping A -> B
    phase_sum_pos = sum(exp_k_pos[j] * np.exp(-1j * pol * n * phi_list[j]) for j in range(3))
    gamma_n = -t * jn_val * phase_sum_pos

    # 2. (-1)^n * Gamma_n(-k) - hopping B -> A
    phase_sum_neg = sum(exp_k_neg[j] * np.exp(-1j * pol * n * phi_list[j]) for j in range(3))
    gamma_n_lower = ((-1.0) ** n) * (-t * jn_val * phase_sum_neg)

    return gamma_n, gamma_n_lower


def build_floquet_hamiltonian_taylor(
    KX, KY, N, omega, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT
):
    """
    Constructs the tridiagonal-block Sambe Floquet matrix using first-order Taylor
    expansion, as derived in Section 2.2 of the report.

    KX, KY: 2D numpy arrays of shape (Nx, Ny)
    Returns: Tensor of shape (Nx, Ny, dim, dim)
    """
    a1, a2, _, _ = get_lattice_vectors(a=a)
    dim = 2 * (2 * N + 1)
    grid_shape = KX.shape
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)

    k_dot_a1 = KX * a1[0] + KY * a1[1]
    k_dot_a2 = KX * a2[0] + KY * a2[1]

    phase1 = np.exp(1j * k_dot_a1)
    phase2 = np.exp(1j * k_dot_a2)

    # Static tight-binding term (Section 2.2)
    gamma_0 = -t * (1.0 + phase1 + phase2)

    # First-order Taylor coupling prefactor derived in Section 2.2:
    # prefactor = -i * sqrt(3) * t * e * a * A_0 / (2 * hbar) * (1/2)
    prefactor = -1j * (np.sqrt(3) * ELECTRON_CHARGE * a * t * A_0) / (4.0 * H_BAR)

    if pol == +1:
        term1_d, term2_d = np.sqrt(3) + 1j, np.sqrt(3) - 1j
        term1_d_prime, term2_d_prime = np.sqrt(3) - 1j, np.sqrt(3) + 1j
    else:
        term1_d, term2_d = np.sqrt(3) - 1j, np.sqrt(3) + 1j
        term1_d_prime, term2_d_prime = np.sqrt(3) + 1j, np.sqrt(3) - 1j

    gamma_d = prefactor * (term1_d * phase1 + term2_d * phase2)
    gamma_d_prime = prefactor * (term1_d_prime * phase1 + term2_d_prime * phase2)

    num_blocks = 2 * N + 1
    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        # Diagonal 2x2 block with m * omega on diagonal
        H_F[..., r, r + 1] = gamma_0
        H_F[..., r + 1, r] = np.conj(gamma_0)
        H_F[..., r, r] += m_index * omega
        H_F[..., r + 1, r + 1] += m_index * omega

        # Upper coupling block (absorption of 1 photon: m -> m+1)
        if i + 1 < num_blocks:
            c = 2 * (i + 1)
            H_F[..., r, c + 1] = gamma_d
            H_F[..., r + 1, c] = np.conj(gamma_d_prime)

        # Lower coupling block (emission of 1 photon: m -> m-1)
        if i - 1 >= 0:
            c = 2 * (i - 1)
            H_F[..., r, c + 1] = gamma_d_prime
            H_F[..., r + 1, c] = np.conj(gamma_d)

    return H_F


def build_floquet_hamiltonian(
    KX, KY, N, omega, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=LATTICE_CONSTANT, max_order=1
):
    """
    Constructs the Sambe Floquet matrix using the exact blocks:
    H_n(k) = [[0, Gamma_n(k)], [(-1)^n * Gamma_n(-k), 0]]
    Truncates multiphoton exchange at 'max_order' (default max_order=1 for
    single-photon exchange). This is the main Hamiltonian used throughout
    the project.
    """
    dim = 2 * (2 * N + 1)
    grid_shape = KX.shape
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)
    z_local = (ELECTRON_CHARGE * A_0 * a) / H_BAR

    # Precompute coupling components for order n = 0, 1, ..., max_order
    blocks_dict = {}
    for n in range(max_order + 1):
        g_upper, g_lower = compute_gamma_n(KX, KY, n, z_local, t, pol, a)
        blocks_dict[n] = (g_upper, g_lower)

    num_blocks = 2 * N + 1
    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        # 1. Diagonal block: H_0 + m * omega * I
        g0_upper, g0_lower = blocks_dict[0]
        H_F[..., r, r + 1] = g0_upper
        H_F[..., r + 1, r] = g0_lower
        H_F[..., r, r] += m_index * omega
        H_F[..., r + 1, r + 1] += m_index * omega

        # 2. Off-diagonal blocks: exchange of n photons
        for n in range(1, max_order + 1):
            gn_upper, gn_lower = blocks_dict[n]

            # Absorption (m -> m + n): placed at (i, i + n), corresponding to H_{-n} = H_n^\dagger
            if i + n < num_blocks:
                c = 2 * (i + n)
                H_F[..., r, c + 1] = np.conj(gn_lower)
                H_F[..., r + 1, c] = np.conj(gn_upper)

                # Emission (m + n -> m): strictly the Hermitian conjugate placed below diagonal
                H_F[..., c + 1, r] = gn_lower
                H_F[..., c, r + 1] = gn_upper

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

    KX, KY: 2D numpy arrays of relative momentum q = k - K (shape: (Nx, Ny)).
    N: Number of Floquet replicas (-N to N).
    omega: Drive frequency.
    pol: +1 (left), -1 (right).
    valley: +1 (K point), -1 (K' point).
    Returns: Tensor of shape (Nx, Ny, dim, dim).
    """
    grid_shape = KX.shape
    dim = 2 * (2 * N + 1)
    H_F = np.zeros((*grid_shape, dim, dim), dtype=complex)

    # 1. Static off-diagonal elements in H_0
    # For valley = +1 (K):  H_0 = h_bar * v_F * [[0, k_minus], [k_plus, 0]]
    # For valley = -1 (K'): H_0 = h_bar * v_F * [[0, k_plus], [k_minus, 0]]
    if valley == +1:
        h0_12 = h_bar * v_F * (KX - 1j * KY)
        h0_21 = h_bar * v_F * (KX + 1j * KY)
    else:
        h0_12 = h_bar * v_F * (KX + 1j * KY)
        h0_21 = h_bar * v_F * (KX - 1j * KY)

    # 2. Driving coupling strength
    coupling = e * v_F * A_0
    xi = valley * pol  # Product of valley index and polarization

    num_blocks = 2 * N + 1
    for i in range(num_blocks):
        m_index = i - N
        r = 2 * i

        # --- Diagonal 2x2 Block (H_0 + m * omega * I) ---
        H_F[..., r, r + 1] = h0_12
        H_F[..., r + 1, r] = h0_21
        H_F[..., r, r] += m_index * omega
        H_F[..., r + 1, r + 1] += m_index * omega

        # --- Off-diagonal coupling: H_1 (absorption, i -> i+1) ---
        if i + 1 < num_blocks:
            c = 2 * (i + 1)
            if xi == +1:
                H_F[..., r + 1, c] = coupling
            else:
                H_F[..., r, c + 1] = coupling

        # --- Off-diagonal coupling: H_-1 = H_1^dagger (emission, i -> i-1) ---
        if i - 1 >= 0:
            c = 2 * (i - 1)
            if xi == +1:
                H_F[..., r, c + 1] = coupling
            else:
                H_F[..., r + 1, c] = coupling

    return H_F
