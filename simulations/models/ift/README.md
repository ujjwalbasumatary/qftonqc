# Ising field theory

The spin-chain Hamiltonian is

$$
H=-\sum_j\left(\sigma_j^z\sigma_{j+1}^z
 +g_x\sigma_j^x+g_z\sigma_j^z\right),
\qquad
\eta_{\rm latt}=\frac{g_x-1}{|g_z|^{8/15}}.
$$

Near $(g_x,g_z)=(1,0)$, its low-energy spectrum approaches Ising field
theory. The scattering construction follows
[arXiv:2411.13645](https://arxiv.org/abs/2411.13645). A uniform matrix product
state (MPS) represents the vacuum, and a tangent tensor represents a
one-particle excitation. Two localized packets built from these excitations
are evolved in a finite window.

[`scripts/spectrum.jl`](scripts/spectrum.jl) calculates the lowest
tangent-space excitation energy over a chosen interval of momenta.

[`scripts/collide_fixed_momentum.jl`](scripts/collide_fixed_momentum.jl) uses
one excitation tensor for each packet, evaluated at momenta $+k$ and $-k$.
It saves the vacuum-subtracted bond-energy density and
$\langle\sigma_n^z\rangle-\langle\sigma^z\rangle_{\rm vac}$.

[`scripts/collide_momentum_grid.jl`](scripts/collide_momentum_grid.jl)
calculates excitation tensors across the Brillouin zone and Fourier sums them
into a packet. Each tensor currently receives an independent phase choice. If
neighboring momenta receive incompatible phases, their Fourier sum can shift,
distort, or delocalize the packet.

The two collision programs save local expectation values. To obtain the
probability of an outgoing channel, project the evolved state onto separated
multi-particle states in that channel. Summing the probabilities over the
channels included in the calculation then checks how much of the final
state those channels account for.

[`scripts/state_io.jl`](scripts/state_io.jl) loads the MPS states saved by
the fixed-momentum program. [`scripts/particle_basis.jl`](scripts/particle_basis.jl)
calculates excitation tensors in the left and right gauges on the saved
vacuum. [`scripts/two_particle_overlap.jl`](scripts/two_particle_overlap.jl)
has the contractions with localized excitation pairs, their norms, and their
Gram matrices. The docstrings give the tensor conventions and normalization.
The particle-production page in `docs/src/physics/particle-production.md`
explains how the momentum dependence enters the overlaps.

[`notebooks/exact_diagonalization_and_qiskit.ipynb`](notebooks/exact_diagonalization_and_qiskit.ipynb)
builds the open-chain Ising Hamiltonian by exact diagonalization and uses its
ground state to initialize a Qiskit circuit.

[`notebooks/spectrum_and_local_quench.ipynb`](notebooks/spectrum_and_local_quench.ipynb)
finds the uniform-MPS ground state, calculates the tangent-space dispersion,
and evolves a local disturbance. The same local-quench calculation is written
as a Julia script in [`examples/local_quench.jl`](examples/local_quench.jl).

The Mathematica TFIM notebook is in
[`course/mathematica/`](../../../course/mathematica/).
