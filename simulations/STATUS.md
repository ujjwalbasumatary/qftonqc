# Status of the scattering calculations

The Julia tests cover the oscillator matrix elements, the exact
free dispersions, the momentum grid, the Gaussian weights, and the number of
excitations in one- and two-particle MPS windows. The model programs can find
MPS ground states, construct packets, evolve them with TDVP, and save local
expectation values.

A local expectation value is not a particle-number measurement. An additional
ray in an energy-density plot shows that energy has left the collision region
with a different velocity. Identifying the ray with a particle species
requires agreement with that species' dispersion. Its probability requires an
overlap with a separated outgoing particle state.

## Quantities that should accompany a long run

The JLD2 files contain local observables and sampling times. The Ising and
phi-four programs put parameters in the filenames, while the Schwinger
program also saves the parsed argument dictionary. The following quantities
still have to be added:

- the Git commit and Julia package versions;
- the norm and energy at every saved time;
- the bond dimension and discarded weight along the window;
- the distance between the outgoing signal and each window boundary;
- MPS checkpoints from which an interrupted evolution can resume.

These quantities distinguish errors from the time step, the bond cutoff, and
the finite window. They should be stored with the state rather than inferred
from the final plot.

## Ising field theory

The lattice Hamiltonian is

$$
H=-\sum_j\left(\sigma_j^z\sigma_{j+1}^z
 +g_x\sigma_j^x+g_z\sigma_j^z\right),
\qquad
\eta_{\rm latt}=\frac{g_x-1}{|g_z|^{8/15}}.
$$

The spectrum program finds a one-site VUMPS ground state and the lowest branch
returned by the tangent-space excitation calculation. The fixed-momentum
collision program places one packet at $+k$ and one at $-k$, joins them
through the vacuum bond matrix, and evolves the resulting window with
two-site TDVP. It saves the vacuum-subtracted bond-energy density and
$\langle\sigma_n^z\rangle-\langle\sigma^z\rangle_{\rm vac}$.

The following calculations are still required for the comparison with
[arXiv:2411.13645](https://arxiv.org/abs/2411.13645).

1. Calculate the vacuum energy, correlation length, Schmidt values, and
   excitation energies at several vacuum bond dimensions. The quoted digits
   should remain unchanged when the bond dimension is increased.
2. At $g_z=0$, compare the tangent-space dispersion with
   $$
   \epsilon(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
   $$
   A two-packet evolution in the same limit should leave unit probability in
   the two-particle sector.
3. For each incoming packet, calculate its norm, mean position, position
   variance, mean momentum, momentum variance, energy, and group velocity.
   Increase the support until the packet amplitude at its ends is negligible
   compared with its maximum.
4. Compare evolutions after halving the time step, increasing the evolution
   bond dimension, increasing the window, and changing the packet width. Each
   comparison should be made before any value is quoted from the collision.
5. Construct separated outgoing one- and multi-particle states. Project the
   late-time MPS onto these states and check that the sum of the resolved
   probabilities approaches one as the momentum grid and separation cut are
   refined.
6. Calculate the elastic time delay from two nearby incoming momenta. The
   reference uses momentum differences as small as $10^{-3}$; a displacement
   read from a heatmap is not the same calculation.

The reference used correlation lengths of roughly 4–10 lattice sites, packet
widths of 70–120 sites, and windows of 1000–2000 sites. Smaller windows are
appropriate while the packet construction and outgoing projections are being
checked.

## Bosonized Schwinger model

The Hamiltonian used in
[arXiv:2307.02522](https://arxiv.org/abs/2307.02522) is

$$
H=\chi\sum_x\left[
 \frac{\pi_x^2}{2}
 +\frac{(\phi_x-\phi_{x-1})^2}{2}
 +\frac{\mu^2\phi_x^2}{2}
 -\lambda\cos(\beta\phi_x-\theta)
\right].
$$

The current program diagonalizes the onsite Hamiltonian in a finite oscillator
basis, retains its lowest eigenstates, finds a finite-chain ground state with
a source on the central sites, changes the source strength, and evolves the
state with TDVP. It saves $\langle\phi_n(t)\rangle$. It does not construct
incoming quark or meson states.

The comparison with the reference requires the following additions.

1. Increase the oscillator cutoff until the retained onsite energies and the
   matrix elements of $\phi$ and $\phi^2$ stop changing at the quoted
   digits. Repeat this comparison while increasing the number of retained
   onsite states.
2. At $\lambda=0$, compare the excitation energy with
   $$
   \omega(p)=\chi\sqrt{\mu^2+4\sin^2(p/2)}.
   $$
3. Find the uniform-MPS vacua. Quarks require tangent tensors connecting
   different vacua; mesons require tangent tensors over the same vacuum.
4. Choose compatible tensor phases across the momentum grid, form localized
   packets, and calculate their energy, momentum, velocity, charge, and field
   profile before evolution.
5. Enlarge the time-evolution window before an outgoing signal reaches its
   edge. Compare the result while varying the time step, bond dimension,
   onsite cutoff, number of retained onsite states, packet separation, and
   packet width.
6. Project the final state onto separated quark and meson states. Repeat the
   projection with a finer momentum grid and a larger separation between
   particles. The sum over the resolved sectors should approach one.

The reference begins quark evolutions near bond dimension 20 and increases it
towards 50. Its meson evolutions begin near 40 and increase towards 100. The
onsite calculation uses an oscillator cutoff near 2000 and retains 12 states.
These values do not replace the comparisons listed above.

## Lattice $\phi^4$ theory

The $\phi^4$ program constructs two packets and saves a local bond-energy
assignment together with
$\langle\phi_n^2\rangle-\langle\phi^2\rangle_{\rm vac}$. No particular
published scattering result has yet been selected for comparison. Before
extracting particle probabilities, the calculation needs the same packet
measurements, parameter variations, and outgoing-state projections described
above.
