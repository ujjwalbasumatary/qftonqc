# Lattice $\phi^4$ theory

The scalar calculation uses lattice spacing $a=1$ and

$$
H=\sum_n\left[
 \frac{\pi_n^2}{2}
 +\frac{(\phi_n-\phi_{n+1})^2}{2}
 +\frac{\mu_0^2\phi_n^2}{2}
 +\frac{\lambda_0}{4!}\phi_n^4
\right].
$$

[`scripts/collide_wavepackets.jl`](scripts/collide_wavepackets.jl) finds a
vacuum represented by a uniform matrix product state (MPS). It calculates
tangent-space excitations on an evenly spaced momentum grid. The number
of grid points sets the number of sites in each packet interval. The
excitation tensors are summed to form two localized packets in separate
intervals. The two intervals are then joined into a window and evolved using
two-site TDVP, the time-dependent variational principle applied to adjacent
MPS tensors.

The program saves the energy density assigned to each bond, along with
$\langle\phi_n^2\rangle-\langle\phi^2\rangle_{\rm vac}$ at each site. These local
expectation values describe the energy and the change in the second moment
of the field along the chain. Particle probabilities would require overlaps
of the late-time MPS with separated one- and multi-particle states. Those
overlaps must be compared across calculations in which the oscillator cutoff, time step,
bond dimension, window length, and packet width are varied one at a time.

The notebooks contain calculations of the ground state and its entanglement,
as well as the evolution of local excitations and wave packets.

| Notebook | Calculation |
| --- | --- |
| [`notebooks/entanglement/finite_chain_ground_state.ipynb`](notebooks/entanglement/finite_chain_ground_state.ipynb) | finite-chain ground states and their entanglement |
| [`notebooks/entanglement/uniform_ground_state.ipynb`](notebooks/entanglement/uniform_ground_state.ipynb) | uniform-MPS ground states as the oscillator cutoff and couplings are varied |
| [`notebooks/scattering/finite_chain_local_excitations.ipynb`](notebooks/scattering/finite_chain_local_excitations.ipynb) | two local field insertions in a finite MPS |
| [`notebooks/scattering/window_local_excitations.ipynb`](notebooks/scattering/window_local_excitations.ipynb) | two local field insertions in a window MPS |
| [`notebooks/scattering/local_observables.ipynb`](notebooks/scattering/local_observables.ipynb) | energy-density and $\phi^2$ plots from saved evolutions |
| [`notebooks/wavepackets/momentum_space_packets.ipynb`](notebooks/wavepackets/momentum_space_packets.ipynb) | two packets assembled from tangent tensors on a momentum grid |

The Julia files in [`examples/ground_states/`](examples/ground_states/) and
[`examples/local_excitations/`](examples/local_excitations/) contain the same
ground-state and local-insertion calculations without the notebook interface.
