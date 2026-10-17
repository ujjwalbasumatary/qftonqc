"""Reproduce the Chapter 9 cutoff table with NumPy only.

H = p^2/2 + x^2/2 + coupling*x^4, with hbar = mass = frequency = 1.
The SHO matrix projects x^4 before truncation. Grid kinetic energy is spectral.
Run: python anharmonic_cutoff_comparison.py
"""
import numpy as np


def sho_hamiltonian(dimension, coupling=0.1):
    n = np.arange(dimension, dtype=float)
    x4 = np.diag(0.75 * (2*n*n + 2*n + 1))
    for level in range(dimension - 2):
        element = 0.5 * (2*level + 3) * np.sqrt((level + 1)*(level + 2))
        x4[level, level + 2] = x4[level + 2, level] = element
    for level in range(dimension - 4):
        element = 0.25 * np.sqrt(np.prod(np.arange(level + 1, level + 5)))
        x4[level, level + 4] = x4[level + 4, level] = element
    return np.diag(n + 0.5) + coupling*x4


def grid_hamiltonian(dimension, field_max=5.0, coupling=0.1, twist=False):
    if dimension < 2 or dimension % 2:
        raise ValueError("Use an even grid dimension, as in the qubit encodings.")
    # field_max is HALF THE PERIOD for either grid convention.
    spacing = 2*field_max/dimension
    shift = 0.5 if twist else 0.0
    indices = np.arange(dimension) - dimension/2 + shift
    x = spacing*indices
    p = np.pi/field_max*indices
    F = np.exp(-1j*np.outer(p, x))/np.sqrt(dimension)
    assert np.allclose(F.conj().T @ F, np.eye(dimension), atol=1e-12)
    return F.conj().T @ np.diag(p*p/2) @ F + np.diag(x*x/2 + coupling*x**4)


def main():
    coupling, field_max = 0.1, 5.0
    print('H = p^2/2 + x^2/2 + 0.1 x^4')
    print('method     dimension         E0             E1             E2')
    for method in ('SHO', 'periodic'):
        for dimension in (8, 16, 32):
            H = (sho_hamiltonian(dimension, coupling) if method == 'SHO' else
                 grid_hamiltonian(dimension, field_max, coupling))
            energies = np.linalg.eigvalsh(H)[:3]
            print(f'{method:8s} {dimension:7d}  ' + '  '.join(f'{e:.10f}' for e in energies))
    reference = np.linalg.eigvalsh(sho_hamiltonian(128, coupling))[:3]
    checks = {
        'SHO96': np.linalg.eigvalsh(sho_hamiltonian(96, coupling))[:3],
        'periodic64_L10': np.linalg.eigvalsh(grid_hamiltonian(64, 5, coupling))[:3],
        'periodic64_L12': np.linalg.eigvalsh(grid_hamiltonian(64, 6, coupling))[:3],
        'twisted64_L10': np.linalg.eigvalsh(grid_hamiltonian(64, 5, coupling, twist=True))[:3],
    }
    print('Reference SHO128: ' + '  '.join(f'{e:.12f}' for e in reference))
    for label, energies in checks.items():
        error = np.max(np.abs(energies-reference))
        print(f'{label}: max low-energy difference {error:.3e}')
        assert error < 1e-9
    variational = np.array([np.linalg.eigvalsh(sho_hamiltonian(d, coupling))[:3]
                            for d in (8,16,32)])
    assert np.all(np.diff(variational,axis=0) <= 1e-11)
    assert np.all(variational >= reference - 1e-10)
    # Independent check of the analytic x^4 elements with padded ladder matrices.
    d = 8
    x = np.diag(np.sqrt(np.arange(1, d+4)/2),1)
    x += x.T
    projected = np.linalg.matrix_power(x,4)[:d,:d]
    assert np.allclose(sho_hamiltonian(d,coupling),
                       np.diag(np.arange(d)+0.5)+coupling*projected)
    print('All matrix, variational, and independent cutoff checks passed.')


if __name__ == '__main__':
    main()
