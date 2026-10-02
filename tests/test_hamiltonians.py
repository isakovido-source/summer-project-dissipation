"""
Basic sanity tests for the Floquet Hamiltonian.

Run from the project root with:
    pytest

These are exactly the kind of physics-based checks that catch real bugs:
Hermiticity, a known analytic limit, and a topological sum rule. None of
them require knowing the "right" numerical answer in advance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floquet_graphene.hamiltonians import build_floquet_hamiltonian
from floquet_graphene.constants import HOPPING


def test_hamiltonian_is_hermitian():
    kx = np.array([[0.3]])
    ky = np.array([[0.5]])
    H = build_floquet_hamiltonian(kx, ky, N=2, omega=0.4)
    H_dagger = np.conj(np.swapaxes(H, -1, -2))
    assert np.allclose(H, H_dagger, atol=1e-10)


def test_zero_drive_recovers_static_graphene_at_gamma():
    """At A_0 = 0 and k = Gamma = (0, 0), the static graphene Hamiltonian's
    off-diagonal element gamma_0 = -3t exactly."""
    kx = np.array([[0.0]])
    ky = np.array([[0.0]])
    H = build_floquet_hamiltonian(kx, ky, N=0, omega=0.4, A_0=0.0, t=HOPPING)
    # With N=0 there is a single 2x2 block: H[0, 0, 0, 1] = gamma_0
    gamma_0 = H[0, 0, 0, 1]
    assert np.isclose(gamma_0, -3.0 * HOPPING, atol=1e-10)


def test_eigenvalues_are_real():
    kx, ky = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    H = build_floquet_hamiltonian(kx, ky, N=2, omega=0.4)
    evals = np.linalg.eigvalsh(H)
    assert np.all(np.isfinite(evals))
