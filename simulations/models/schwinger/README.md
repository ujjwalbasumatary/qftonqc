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

[Belyansky *et al.*](https://arxiv.org/abs/2307.02522) represent the vacuum by
a uniform matrix product state (MPS). They construct quark and meson
quasiparticles above this vacuum, prepare incoming wave packets, and
calculate overlaps with outgoing particle states.

[`scripts/source_quench.jl`](scripts/source_quench.jl) contains the calculation
of the response to a change in a local source. It diagonalizes the onsite
Hamiltonian in a truncated oscillator basis and retains its lowest
eigenstates as the basis for each MPS site. It then finds the ground state
of a finite open chain with a source on the central sites. After the source
strength changes, the state evolves under the new Hamiltonian using the
time-dependent variational principle (TDVP).

The program saves $\langle\phi_n(t)\rangle$ as `field`. The key `flux`
contains the same array. At $\beta=\sqrt{4\pi}$, the normalization of the
reference gives $E_T/e=\phi/\sqrt{\pi}$.

Preparing incoming quark or meson wave packets requires a uniform-MPS
vacuum and topological and nontopological quasiparticle tensors. The phases
of those tensors must be chosen as a function of momentum, and the two
packets must be joined into an incoming state. This construction is described
in the supplement of the reference.

[`notebooks/onsite_basis_and_ground_state.ipynb`](notebooks/onsite_basis_and_ground_state.ipynb)
constructs the truncated onsite basis and the finite-chain ground state with a
central source.

[`notebooks/source_quench_and_string_breaking.ipynb`](notebooks/source_quench_and_string_breaking.ipynb)
changes that source, evolves the MPS, and plots the field as the disturbance
moves through the chain.

The Mathematica string-breaking notebook is in
[`course/mathematica/`](../../../course/mathematica/).
