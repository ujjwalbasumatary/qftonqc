# Status of the scattering calculations

The Julia tests cover the oscillator matrix elements, the exact
free dispersions, the momentum grid, the Gaussian weights, and the number of
excitations in one- and two-particle MPS windows. The model programs can find
MPS ground states, construct packets, evolve them with TDVP, and save local
expectation values.

A band in an energy-density plot follows the energy carried away from the
collision region. Its motion gives a velocity that can be compared with a
particle species' dispersion. Calculating the probability of an outgoing
channel requires overlaps with separated particle states of the corresponding
species and multiplicity.

## Additional information to save during evolution

The JLD2 files contain local observables and sampling times. The Ising and
$\phi^4$ programs put parameters in the filenames, while the Schwinger
program also saves the parsed argument dictionary. The output still needs
the following additions.

- the Git commit and Julia package versions;
- the norm and energy at every saved time;
- the bond dimension and discarded weight along the window;
- the distance between the outgoing signal and each window boundary;
- saved MPS states from which an interrupted evolution can resume.

Saving these quantities at the same times as the local observables allows
comparisons between runs with different time steps, bond dimensions, and
window sizes. For example, the norm and energy at each saved time give their
drift over the evolution, while the distance to each boundary records how
close an outgoing packet has come to the edge of the window.

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
   excitation energies at several vacuum bond dimensions, comparing their
   values as the bond dimension increases.
2. At $g_z=0$ and $g_x>1$, compare the tangent-space dispersion with
   $$
   \epsilon(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
   $$
   The spectrum command uses the same vacuum on both sides of the excitation
   tensor. It excludes $g_z=0$, $g_x<1$, where a single fermion is a kink
   between different ordered vacua, and the critical point $(g_x,g_z)=(1,0)$,
   where its mass-normalized ratios are undefined. In the free theory, an
   exact two-particle state retains unit probability in the two-particle
   sector; the packet preparation and evolution can be compared against
   that expectation.
3. For each incoming packet, calculate its norm, mean position, position
   variance, mean momentum, momentum variance, energy, and group velocity.
   Increase the support until the packet amplitude at its ends is negligible
   compared with its maximum.
4. Compare evolutions after halving the time step, increasing the evolution
   bond dimension, increasing the window, and changing the packet width.
   Compare the energy and magnetization profiles at matching sites and
   physical times.
5. Construct separated outgoing one- and multi-particle states. Project the
   late-time MPS onto these states and check that the sum of the probabilities
   for the included channels approaches one as the momentum grid and
   separation cut are refined.
6. Calculate the elastic time delay by projecting the same evolved state onto
   reference states at two nearby momenta. Subtract the relative phase from
   free propagation to isolate the scattering phase difference. The reference
   uses momentum differences as small as $10^{-3}$.

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
a source on the central sites, then evolves this state with TDVP after changing
the source strength. It saves $\langle\phi_n(t)\rangle$. Preparing incoming
quarks or mesons requires the vacuum and excitation calculations listed below.

The comparison with the reference requires the following additions.

1. Increase the oscillator cutoff and compare the retained onsite energies
   and matrix elements of $\phi$ and $\phi^2$. Then increase the number of
   retained onsite states and repeat the comparison.
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
   edge. Compare the field profile and packet motion while varying the time
   step, bond dimension, onsite cutoff, number of retained onsite states,
   packet separation, and packet width.
6. Project the final state onto separated quark and meson states. Repeat the
   projection with a finer momentum grid and a larger separation between
   particles, comparing how closely the sum of probabilities over the included
   sectors approaches one.

The reference begins quark evolutions near bond dimension 20 and increases it
towards 50. Its meson evolutions begin near 40 and increase towards 100. The
onsite calculation uses an oscillator cutoff near 2000 and retains 12 states.

## Lattice $\phi^4$ theory

The $\phi^4$ program constructs two packets and saves a local bond-energy
assignment together with
$\langle\phi_n^2\rangle-\langle\phi^2\rangle_{\rm vac}$. No particular
published scattering result has yet been selected for comparison. The incoming
packet widths and momenta still need to be measured, and the evolved energy
and field profiles need to be compared at different time steps, bond
dimensions, and window sizes. Projecting the late-time MPS onto separated
outgoing particle states then gives the channel probabilities.
