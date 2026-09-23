# Particle production and outgoing states

A collision begins with two separated incoming particles. After they meet,
the state can contain a superposition of outgoing channels, distinguished by
their particle species, momenta, and multiplicities. For example, ``11\to12``
creates a heavier species while keeping two outgoing particles, whereas
``11\to111`` increases the particle number from two to three. The phrase
“particle production” can refer to either process, so the outgoing species
and multiplicity distinguish the two.

The current collision programs save local energy and field expectation
values. Extracting scattering probabilities also requires projecting the
outgoing MPS onto particle states; these projections remain to be implemented.
The Schwinger program prepares a ground state with a central source and changes
that source; its initial state is described on the
[Schwinger-model page](schwinger.md).

## Energy carried away from a collision

For the Ising and scalar-field windows, the saved energy density is

```math
\delta e_n(t)=\langle\psi(t)|h_{n,n+1}|\psi(t)\rangle
-\langle\Omega|h_{n,n+1}|\Omega\rangle,
```

where ``|\Omega\rangle`` is the corresponding uniform vacuum. A localized
packet produces a band in position and time. Once the packet is separated
from other excitations, its velocity can be compared with the dispersion of
a candidate particle species,

```math
v_a(p)=\frac{dE_a(p)}{dp}.
```

The sign of ``v_a`` determines its direction and its magnitude determines the
change in position per unit time. Each momentum component evolves with a
phase ``e^{-iE_a(p)t}``. Over a narrow range of momenta, the linear variation
of ``E_a(p)`` translates the envelope at velocity ``v_a``. If this velocity
varies appreciably across the packet's momentum distribution, different
components move apart and the packet spreads. Broadening can therefore occur
even during the propagation of a single particle.

A local expectation value includes contributions from every channel in the
quantum state, so several visible bands can belong to different possible
outcomes. The number of bands therefore need not equal the number of particles
in any one outcome. Their heights depend on the channel probabilities as well
as particle energies, packet widths, and matrix elements of the chosen
observable. Section III of the
[Ising scattering paper](https://arxiv.org/abs/2411.13645) discusses
this distinction using outgoing states containing different species.

Two-point energy correlations retain information about which regions carry
energy together. For instance, one can evaluate

```math
C_{nm}(t)=\langle\delta h_n\delta h_m\rangle_t
-\langle\delta h_n\rangle_t\langle\delta h_m\rangle_t,
\qquad
\delta h_n=h_{n,n+1}-\langle h_{n,n+1}\rangle_{\rm vac}I.
```

At sufficiently separated sites the bond operators act on disjoint sites.
Correlations between outgoing regions supplement the one-point profiles;
the current output files contain only the one-point observables listed in
[data and figures](../data.md).

## Energetically allowed channels

For two packets narrowly centred at ``+k`` and ``-k`` on branch 1, the incoming
energy is approximately ``E_{\rm in}=2E_1(k)``. On the lattice, an outgoing
channel must satisfy

```math
\sum_{r=1}^{N_{\rm out}}E_{a_r}(p_r)=E_{\rm in},\qquad
\sum_{r=1}^{N_{\rm out}}p_r=0\pmod{2\pi}.
```

These equations use excitation energies above the same vacuum. Internal
symmetries can further restrict the channel. Near a continuum limit with
dispersion minima at zero momentum, the sum of outgoing rest masses gives
the threshold at zero total momentum. In this limit, the example channels
``11\to12`` and ``11\to111`` have thresholds ``m_1+m_2`` and ``3m_1``,
respectively. Away from that limit, minimize the sum of lattice dispersions
subject to the momentum constraint.

Energy and momentum conservation determine which outgoing configurations are
allowed. The interactions determine the amplitude for reaching each one.
Enough incoming energy to cross a threshold therefore permits a channel to
open, while its probability still depends on the scattering dynamics.

Finite packets have an energy distribution, so a packet centred near a
threshold can contain components on both sides of it. The distribution
determines how much of the incoming packet has enough energy to enter the
channel. The [Ising page](ising.md) gives the lattice Hamiltonian and units
used by the spectrum calculation.

## Projections onto outgoing particles

Let ``\mathcal H_\alpha^{\rm out}`` denote a channel with specified species
and multiplicity, including its allowed momenta. Given an orthonormal basis
``\{|\alpha,r\rangle\}`` of separated outgoing states, its probability is

```math
P_\alpha(t)=
\frac{\sum_r|\langle\alpha,r|\psi(t)\rangle|^2}
{\langle\psi(t)|\psi(t)\rangle}.
```

The sum includes the momentum or position measure used to normalize that
basis. For identical particles, ordering the positions is one way to avoid
counting the same configuration more than once. The excitation tensors and
their normalization must be consistent with the tensors used to prepare the
incoming state.

A finite collection of outgoing wave packets will generally overlap. If
``G_{rs}=\langle\chi_r|\chi_s\rangle`` and
``b_r=\langle\chi_r|\psi\rangle``, the probability in their span is

```math
P_{\rm span}=\frac{b^\dagger G^+b}{\langle\psi|\psi\rangle},
```

where ``G^+`` is the Moore–Penrose inverse on the retained independent
directions. Simply summing ``|b_r|^2`` would count overlap between the basis
states more than once. Forming ``G^+`` requires deciding which eigenvalues of
``G`` to discard. Repeating the projection with different cutoffs shows how
much that choice changes the probability. Save the discarded eigenvalues
alongside the probability to record which part of the spectrum was omitted
from the inverse.

The outgoing particles need time to separate. Restricting their insertion
positions to a minimum separation excludes configurations that still interact
appreciably. Compare the channel probabilities at larger minimum separations
and at several late times, keeping all outgoing packets inside the window.
Section II.4 of the [Ising paper](https://arxiv.org/abs/2411.13645) constructs probabilities from
such overlaps and normalizes its Fourier sums against the incoming sector.

Mutually orthogonal, exhaustive channels have probabilities summing to one.
In a finite calculation, the missing weight
``1-\sum_\alpha P_\alpha`` can include omitted species or multiplicities,
overlapping particles, basis truncation, and evolution error. To distinguish
these contributions, compare the channel probabilities after enlarging the
outgoing basis, reducing the time step, increasing the bond dimension, and
enlarging the window. Assigning the missing weight to a particular channel
requires an explicit projection onto that channel.

For the Julia programs, the remaining work is to retain the evolved MPS,
construct the separated outgoing basis from identified excitation branches,
and evaluate these overlaps. The saved local-observable arrays cannot be
used to reconstruct the required many-body state. The relevant numerical
comparisons are collected on the [comparison page](../comparisons.md).
