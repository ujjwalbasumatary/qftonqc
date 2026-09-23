# Bosonized Schwinger model

The reference calculation uses

$$
H=\chi\sum_x\left[
 \frac{\pi_x^2}{2}
 +\frac{(\phi_x-\phi_{x-1})^2}{2}
 +\frac{\mu^2\phi_x^2}{2}
 -\lambda\cos(\beta\phi_x-\theta)
\right].
$$

[Belyansky *et al.*](https://arxiv.org/abs/2307.02522) construct a uniform-MPS
vacuum, quark and meson quasiparticles, incoming wave packets, and overlaps
with outgoing particle states.

[`scripts/source_quench.jl`](scripts/source_quench.jl) contains the calculation
currently implemented here. It diagonalizes the onsite Hamiltonian in a
truncated oscillator basis, keeps its lowest eigenstates, finds the ground
state of a finite open chain with a source on the central sites, changes the
source strength, and evolves the state with TDVP.

The program saves $\langle\phi_n(t)\rangle$ as `field`. The key `flux`
contains the same array. In the normalization of the reference,
$E_T/e=\phi/\sqrt{\pi}$.

The source quench does not contain incoming quark or meson wave packets. Those
states require a uniform-MPS vacuum, topological and nontopological
quasiparticle tensors, a momentum-dependent phase choice, and the packet
gluing described in the supplement of the reference.

[`notebooks/onsite_basis_and_ground_state.ipynb`](notebooks/onsite_basis_and_ground_state.ipynb)
constructs the truncated onsite basis and the finite-chain ground state with a
central source.

[`notebooks/source_quench_and_string_breaking.ipynb`](notebooks/source_quench_and_string_breaking.ipynb)
changes that source, evolves the MPS, and plots the field as the disturbance
moves through the chain.

The Mathematica string-breaking notebook is in
[`course/mathematica/`](../../../course/mathematica/).
