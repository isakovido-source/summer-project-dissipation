import os
import math
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import j0, j1  # Zeroth-order Bessel function for dynamical band renormalization
from scipy.special import jv

# ====================================
#             CONSTANTS
# ====================================


H_BAR = 1.0                 #chosing units such that h_bar = 1
ELECTRON_CHARGE = 1.0       #chosing units such that e = 1  

POLARIZATION = -1
AMPLITUDE = 1.2
HOPPING = 0.2
NEARES_NIEGHBOORS_LATTICE = 1.0      # lattice base vector a length in our untis


FERMI_VELOSOTY = 3 * HOPPING * NEARES_NIEGHBOORS_LATTICE  / (2 * H_BAR)          # fermi dirac v_F = 3ta/(2 * h_bar)
Z_CONST = AMPLITUDE * NEARES_NIEGHBOORS_LATTICE * ELECTRON_CHARGE / H_BAR        # actual drive amplitude, pluggen in Jn(z)


FLUX_THRESHOLD = 0.90 * np.pi
MIN_RESOLVABLE_GAP = 1e-4

# ==============================================================================
# 1. Physical Lattice Definitions
# ==============================================================================
def get_lattice_vectors(a=NEARES_NIEGHBOORS_LATTICE):
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

# ==============================================================================
# 2. Vectorized Floquet Matrix Construction
# ==============================================================================

# The n'th H_n Gamma coeficents to plug into the Hamiltonian
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
    gamma_n_lower = ((-1.0)**n) * (-t * jn_val * phase_sum_neg)
    
    return gamma_n, gamma_n_lower


# First order Talor expantion
def create_Floquet_Graphene_Hamiltonian_Taylor(KX, KY, N, omega, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=NEARES_NIEGHBOORS_LATTICE):
    """
    Constructs the tridiagonal-block Sambe Floquet matrix using first-order Taylor expansion 
    as derived in Section 2.2 of the report.
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


def create_Floquet_Graphene_Hamiltonian(KX, KY, N, omega, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=NEARES_NIEGHBOORS_LATTICE, max_order=1):
    """
    Constructs the Sambe Floquet matrix using the exact blocks:
    H_n(k) = [[0, Gamma_n(k)], [(-1)^n * Gamma_n(-k), 0]]
    Truncates multiphoton exchange at 'max_order' (default max_order=1 for single-photon exchange).
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

def create_Floquet_Dirac_Hamiltonian(KX, KY, N, omega,A_0=AMPLITUDE,
    v_F=FERMI_VELOSOTY,
    pol=POLARIZATION,
    valley=+1,
    h_bar=H_BAR,
    e=ELECTRON_CHARGE,
):
  """Constructs the Sambe Floquet matrix for low-energy Dirac electrons.

  KX, KY: 2D numpy arrays of relative momentum q = k - K (shape: (Nx, Ny)). N:
  Number of Floquet replicas (-N to N). omega: Drive frequency. pol: +1 (left),
  -1 (right). valley: +1 (K point), -1 (K' point). Returns: Tensor of shape (Nx,
  Ny, dim, dim).
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

  # Define H_1 elements (H_-1 is its Hermitian conjugate)
  if xi == +1:
    # Absorbing a photon couples row 2 of block m to row 1 of block m-1
    h1_val = (1, 0)  # element at [1, 0] inside the 2x2 block
  else:
    h1_val = (0, 1)  # element at [0, 1] inside the 2x2 block

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

# ==============================================================================
# 3. Chern Number Calculation (Fukui-Hatsugai-Suzuki Method) & Windin number
# ==============================================================================
def evaluate_fhs_on_grid(
    w,
    N_rep,
    band_idx,
    N_grid,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=NEARES_NIEGHBOORS_LATTICE,
):
  """Evaluates Berry flux and Chern number for 'band_idx' on a fixed N_grid x N_grid mesh."""
  _, _, b1, b2 = get_lattice_vectors(a=a)
  Nx, Ny = N_grid, N_grid

  i_vals = np.arange(Nx)
  j_vals = np.arange(Ny)
  I_grid, J_grid = np.meshgrid(i_vals, j_vals, indexing='ij')

  KX = (I_grid / Nx) * b1[0] + (J_grid / Ny) * b2[0]
  KY = (I_grid / Nx) * b1[1] + (J_grid / Ny) * b2[1]

  H_F_all = create_Floquet_Graphene_Hamiltonian(
      KX, KY, N=N_rep, omega=w, A_0=A_0, t=t, pol=pol, a=a, max_order=1
  )
  evals_all, evecs_all = np.linalg.eigh(H_F_all)

  total_bands = evals_all.shape[-1]

  # Gap check: Edge bands (0 and total_bands-1) touch due to truncation.
  # For internal bands, we check the gap to the adjacent band.
  if band_idx + 1 < total_bands:
    direct_bandgap = np.min(
        evals_all[..., band_idx + 1] - evals_all[..., band_idx]
    )
  else:
    direct_bandgap = np.min(
        evals_all[..., band_idx] - evals_all[..., band_idx - 1]
    )

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

def compute_floquet_chern_adaptive(N, band_idx, omega, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=NEARES_NIEGHBOORS_LATTICE, 
                                   flux_threshold=FLUX_THRESHOLD, gap_threshold=MIN_RESOLVABLE_GAP, max_grid=360):
    """
    Computes the Chern number of 'band_idx' adaptively.
    Initial grid is analytically evaluated from FHS admissibility condition |F12| < pi using full physical constants.
    Aborts immediately if direct_gap < gap_threshold.
    """
    if N == 0:
        return None, 0.0, 0.0, 0.0, 60

    # Dimensionless driving intensity z
    z_param = (ELECTRON_CHARGE * A_0 * a) / H_BAR

    # Condition 1: Direct momentum sampling of avoided crossings
    n_crossings = (2.0 * np.pi) / max(z_param, 1e-4)

    # Condition 2: FHS admissibility on Berry curvature peak
    factor = np.sqrt( (8 * np.pi) / (27 * np.sqrt(3))) # factor ~= 0.733
    n_admissibility = factor * (H_BAR * omega / (max(t, 1e-4)) * max(z_param**2, 1e-4))

    # Initial grid determination from the most restrictive analytical condition
    estimated_n = int(np.ceil(max(n_crossings, n_admissibility)))
    N_grid = int(np.clip(estimated_n, 60, max_grid))
    step_increase = 40

    while N_grid <= max_grid:
        c_num, c_sum, max_flux, direct_gap = evaluate_fhs_on_grid(
            w=omega, N_rep=N, band_idx=band_idx, N_grid=N_grid,
            A_0=A_0, t=t, pol=pol, a=a
        )

        # Abort immediately if the band touches an adjacent band
        if direct_gap < gap_threshold:
            return None, 0.0, direct_gap, max_flux, N_grid

        # Check FHS admissibility condition (|F12| < flux_threshold < pi)
        if max_flux < flux_threshold:
            return c_num, c_sum, direct_gap, max_flux, N_grid

        print(f"    [FHS Refinement] band={band_idx}: max_flux={max_flux:.2f} rad >= {flux_threshold:.2f} rad on {N_grid}x{N_grid}. Increasing grid...")
        N_grid += step_increase

    print(f"    [Warning] Reached max_grid={max_grid} for band {band_idx} with max_flux={max_flux:.2f} rad.")
    return c_num, c_sum, direct_gap, max_flux, N_grid

#function of the winding number W(E) 
def compute_winding_number_for_band(
    target_band,
    omega,
    A_0=AMPLITUDE,
    t=HOPPING,
    pol=POLARIZATION,
    a=NEARES_NIEGHBOORS_LATTICE,
    M=None,
    max_grid=300,
    max_order=1
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
# ==============================================================================
# 4. Compute the bandgaps and gap closing
# ==============================================================================

def compute_direct_gap(omega_val, N, A_0=0.15, t=0.2, pol=+1, a=1.0, N_k=120):
    """
    Computes the minimum direct gap across the entire BZ between bands 2N and 2N+1.
    """
    _, _, b1, b2 = get_lattice_vectors(a=a)
    i_vals = np.arange(N_k)
    j_vals = np.arange(N_k)
    I_grid, J_grid = np.meshgrid(i_vals, j_vals, indexing='ij')
    
    KX = (I_grid / N_k) * b1[0] + (J_grid / N_k) * b2[0]
    KY = (I_grid / N_k) * b1[1] + (J_grid / N_k) * b2[1]
    
    H_F = create_Floquet_Graphene_Hamiltonian(KX, KY, N=N, omega=omega_val, A_0=A_0, t=t, pol=pol, a=a)
    evals = np.linalg.eigvalsh(H_F)
    
    band_val = 2 * N
    band_cond = 2 * N + 1
    gaps = evals[..., band_cond] - evals[..., band_val]
    
    min_idx = np.unravel_index(np.argmin(gaps), gaps.shape)
    return gaps[min_idx], (KX[min_idx], KY[min_idx])

def find_topological_critical_frequency(omega_bracket, N, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=1.0, tol=1e-5, N_k_base=100):
    """
    Finds the exact transition frequency omega_c where the direct gap closes.
    """
    omega_a, omega_b = omega_bracket
    print(f"[*] Starting search for gap closing in [{omega_a:.6f}, {omega_b:.6f}] (N={N})...")
    
    def objective(w, n_grid):
        gap, _ = compute_direct_gap(w, N=N, A_0=A_0, t=t, pol=pol, a=a, N_k=n_grid)
        return gap

    res = minimize_scalar(
        lambda w: objective(w, N_k_base),
        bracket=(omega_a, (omega_a + omega_b) / 2.0, omega_b),
        bounds=omega_bracket,
        method='bounded',
        options={'xatol': tol}
    )
    
    omega_candidate = res.x
    N_k_fine = int(N_k_base * 1.8)
    final_gap, k_closing = compute_direct_gap(omega_candidate, N=N, A_0=A_0, t=t, pol=pol, a=a, N_k=N_k_fine)
    
    print(f"[✓] Transition identified:")
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
    a=NEARES_NIEGHBOORS_LATTICE,
    max_order=1,
    N_k = 100
):
    """
    Computes direct band gaps across the entire BZ immediately below and above 'target_band'.
    
    Returns:
    --------
    gap_below    : float or None (gap to target_band - 1, None if target_band == 0)
    gap_above    : float or None (gap to target_band + 1, None if target_band == total_bands - 1)
    k_min_below  : tuple (kx, ky) of the minimum gap location below
    k_min_above  : tuple (kx, ky) of the minimum gap location above
    """
    _, _, b1, b2 = get_lattice_vectors(a=a)
    i_vals = np.arange(N_k)
    j_vals = np.arange(N_k)
    I_grid, J_grid = np.meshgrid(i_vals, j_vals, indexing='ij')
    KX = (I_grid / N_k) * b1[0] + (J_grid / N_k) * b2[0]
    KY = (I_grid / N_k) * b1[1] + (J_grid / N_k) * b2[1]

    H_F = create_Floquet_Graphene_Hamiltonian(
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

# ==============================================================================
# 5. Plotting Functions 
# ==============================================================================

def compute_and_save_interactive_plotly(N, omega, output_folder, c_lower, c_upper, A_0=AMPLITUDE, t=HOPPING, a=1.0, grid_size=60, save_html=False):
    kx_vals = np.linspace(-np.pi / a, np.pi / a, grid_size)
    ky_vals = np.linspace(-np.pi / a, np.pi / a, grid_size)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    H_F_all = create_Floquet_Graphene_Hamiltonian(KX, KY, N=N, omega=omega, A_0=A_0, t=t, a=a)
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

def compute_and_save_2d_dirac_cut(N, omega, output_folder, c_lower, c_upper, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=1.0, num_pts=500):
    a1, a2, _, _ = get_lattice_vectors(a=a)
    kx_val = 2.0 * np.pi / (3.0 * a)
    ky_min = -4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    ky_max = +4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    
    ky_vals = np.linspace(ky_min, ky_max, num_pts)
    kx_vals = np.full_like(ky_vals, kx_val)
    
    KX = kx_vals[:, np.newaxis]
    KY = ky_vals[:, np.newaxis]
    
    H_F = create_Floquet_Graphene_Hamiltonian(KX, KY, N=N, omega=omega, A_0=A_0, t=t, pol=pol, a=a)
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
            a=NEARES_NIEGHBOORS_LATTICE, max_order=max_order, N_k=N_k
        )
        _, gap_upper, _, _ = compute_band_gaps_spectrum(
            omega=w, M=current_M, target_band=band_cond_0, A_0=A_0, t=t, pol=pol, 
            a=NEARES_NIEGHBOORS_LATTICE, max_order=max_order, N_k=N_k
        )
        
        gap_0_list.append(gap_0)
        gap_half_list.append(gap_upper)
        gap_minus_list.append(gap_lower)

        # Compute topological winding numbers
        w0 = compute_winding_number_for_band(
            target_band=band_val_0, omega=w, A_0=A_0, t=t, pol=pol, 
            a=NEARES_NIEGHBOORS_LATTICE, M=current_M, max_order=max_order
        )
        wh = compute_winding_number_for_band(
            target_band=band_cond_0, omega=w, A_0=A_0, t=t, pol=pol, 
            a=NEARES_NIEGHBOORS_LATTICE, M=current_M, max_order=max_order
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
        print(f"[✓] Saved adaptive sweep figure to: {output_filename}")

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
    a = NEARES_NIEGHBOORS_LATTICE
    M = max(2, int(math.ceil(6.0 * t / omega)))
    total_bands = 2 * (2 * M + 1)
    
    if output_filename is None:
        output_filename = f"floquet_bands_annotated_w={omega:.3f}_M={M}.png"
        
    print("=" * 80)
    print(f" COMPUTING ANNOTATED 2D SPECTRUM: omega = {omega:.3f} | Adaptive M = {M} ({total_bands} bands)")
    print("=" * 80)
    
    # 1D Momentum cut through K and K'
    kx_fixed = 2.0 * np.pi / (3.0 * a)
    ky_min = -4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    ky_max = +4.0 * np.pi / (3.0 * np.sqrt(3.0) * a)
    
    ky_vals = np.linspace(ky_min, ky_max, num_k_pts)
    kx_vals = np.full_like(ky_vals, kx_fixed)
    
    KX = kx_vals[:, np.newaxis]
    KY = ky_vals[:, np.newaxis]
    
    H_F = create_Floquet_Graphene_Hamiltonian(
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
        print(f"[✓] Saved annotated spectrum to: {output_filename}")
        
    plt.show()

# ==============================================================================
# 6. High-level Execution Functions
# ==============================================================================
def run_calculate_chern(target_omega, N_replicas=None, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=1.0):
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
        N=N_replicas,
        band_idx=band_val,
        omega=target_omega,
        A_0=A_0,
        t=t,
        pol=pol,
        a=a
    )

    # 2. Conduction band
    c_cond, _, gap_cond, flux_cond, grid_cond = compute_floquet_chern_adaptive(
        N=N_replicas,
        band_idx=band_cond,
        omega=target_omega,
        A_0=A_0,
        t=t,
        pol=pol,
        a=a
    )

    print("\n" + "-" * 40)
    print(f"Results for omega = {target_omega:.4f}:")
    print("-" * 40)
    c_val_str = f"{c_val:+d}" if c_val is not None else "Undefined (Gapless)"
    c_cond_str = f"{c_cond:+d}" if c_cond is not None else "Undefined (Gapless)"

    print(f"Valence Band ({band_val})   : C = {c_val_str} (Direct Gap = {gap_val:.4e}, Grid = {grid_val}x{grid_val})")
    print(f"Conduction Band ({band_cond}): C = {c_cond_str} (Direct Gap = {gap_cond:.4e}, Grid = {grid_cond}x{grid_cond})")

    return c_val, c_cond

def run_generate_graphs(omega_val, N_replicas=None, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=1.0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_graphs_dir = os.path.join(script_dir, "Graphs Graphene Floquet")
    folder_name = f"floquet_results_A0={A_0}_w={omega_val}_t={t}_pol={pol}"
    full_output_path = os.path.join(base_graphs_dir, folder_name)
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

# There are things here to fix
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
            N=N_replicas,
            band_idx=band_val,
            omega=w,
            A_0=A_0,
            t=t,
            pol=pol
        )

        # 2. Conduction band Chern calculation
        c_cond, _, gap_cond, flux_cond, grid_cond = compute_floquet_chern_adaptive(
            N=N_replicas,
            band_idx=band_cond,
            omega=w,
            A_0=A_0,
            t=t,
            pol=pol
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

def run_fast_chern_sweep(omega_array, max_grid_cap=220, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=NEARES_NIEGHBOORS_LATTICE):
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
            N=N_rep,
            band_idx=b_val,
            omega=w,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            flux_threshold=0.85 * np.pi,
            max_grid=max_grid_cap
        )

        # Compute conduction band Chern with capped grid refinement
        c_c, _, _, _, grid_c = compute_floquet_chern_adaptive(
            N=N_rep,
            band_idx=b_cond,
            omega=w,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            flux_threshold=0.85 * np.pi,
            max_grid=max_grid_cap
        )

        c_v_str = f"{c_v:+d}" if c_v is not None else "Gapless"
        c_c_str = f"{c_c:+d}" if c_c is not None else "Gapless"
        grid_max = max(grid_v, grid_c)

        print(f"{w:10.5f} | {N_rep:3d} | {c_v_str:>8} | {c_c_str:>8} | {direct_gap:13.4e} | {grid_max:4d}x{grid_max:<4d}")

# make graph of chern number of the frquency C(omega)
def run_scan_with_all_gaps(omega_min, omega_max, num_points, A_0=AMPLITUDE, t=HOPPING, pol=POLARIZATION, a=NEARES_NIEGHBOORS_LATTICE):
    """
    Executes frequency sweep and plots direct gaps (central, below, above) alongside the Chern number.
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
            N=N_rep,
            band_idx=band_val,
            omega=w,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            max_grid=280
        )
        chern_results.append(c_num)

        # Compute gaps below and above valence band
        g_below, g_central, _, _ = compute_band_gaps_spectrum(
            omega=w,
            M=N_rep,
            target_band=band_val,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            N_k=100
        )
        # Compute gap above conduction band
        _, g_above, _, _ = compute_band_gaps_spectrum(
            omega=w,
            M=N_rep,
            target_band=band_val + 1,
            A_0=A_0,
            t=t,
            pol=pol,
            a=a,
            N_k=100
        )

        gap_below_results.append(g_below)
        gap_central_results.append(g_central)
        gap_above_results.append(g_above)

        status = f"{c_num:+d}" if c_num is not None else "UNDEFINED"
        print(f"    [{idx + 1:3d}/{num_points}] w = {w:.4f} | N = {N_rep:2d} | C = {status:>10} | Gap_central = {g_central:.2e} | Gap_below = {g_below:.2e} | Grid = {grid_pts}x{grid_pts}")

    # Prepare data for plotting
    chern_arr = np.array([np.nan if c is None else float(c) for c in chern_results])
    gap_below_arr = np.array(gap_below_results)
    gap_central_arr = np.array(gap_central_results)
    gap_above_arr = np.array(gap_above_results)
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
    plt.show()

# run calculate winding numbers
def print_spectral_comparison_table(M, omega, A_0, t, pol, a, max_grid=360):
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
            N=M, band_idx=beta, omega=omega, A_0=A_0, t=t, pol=pol, a=a, max_grid=max_grid
        )
        
        c_num = c_val if c_val is not None else 0
        W_running += c_num
        
        marker = " <== W(eps=0)" if beta == target_band_zero else ""
        print(f"{beta:4d} | {m_replica:9d} | {subband:>8} | {label:>10} | {c_num:+10d} | {W_running:+11d} | {grid_used:4d}x{grid_used:<4d}{marker}")

    print("=" * 82)
    return W_running

# ==============================================================================
# 7. Now the actual main() and running
# ==============================================================================
def main():

    """"
    # 1. Run adaptive frequency sweep across the full wide interval
    plot_winding_and_gaps_sweep(
        omega_min=0.15,
        omega_max=0.80,
        num_points=120,
        pol=POLARIZATION,
        save_plot=True,
        output_filename="floquet_wide_sweep.png"
    )
    """""

    # 2. Plot complete 2D Dirac cut with annotated Chern and Winding numbers at omega = 0.36
    plot_floquet_spectrum_with_invariants(
        omega=0.36,
        pol=POLARIZATION,
        save_plot=True,
        output_filename="floquet_spectrum_omega_0.36.png"
    )

if __name__ == "__main__":
    main()
    main()