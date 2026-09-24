# Particle production and outgoing states

A collision begins with two separated incoming particles. After they meet,
the state can contain a superposition of outgoing channels, distinguished by
their particle species, momenta, and multiplicities. For example, ``11\to12``
creates a heavier species while keeping two outgoing particles, whereas
``11\to111`` increases the particle number from two to three. The phrase
“particle production” can refer to either process, so the outgoing species
and multiplicity distinguish the two.

The current collision programs save local energy and field expectation
values. The Ising overlap functions also calculate projections onto pairs
and triples of localized excitation tensors. Identifying the particle species represented
by those tensors requires their excitation energies and momentum dependence.
The fixed-momentum Ising program saves the initial MPS and states at selected
times, together with the preparation vacuum and excitation tensors. You can
load these files for the overlap calculation described below; see
[Data and figures](../data.md).
The Schwinger program prepares a ground state with a central source and changes
that source; its initial state is described on the
[Schwinger-model page](schwinger.md).

## Energy carried away from a collision

For the Ising and scalar-field windows, the saved energy density is

```math
\delta e_n(t)=\langle\psi(t)|h_{n,n+1}|\psi(t)\rangle
-\langle\Omega|h_{n,n+1}|\Omega\rangle,
```

where ``|\Omega\rangle`` is the corresponding uniform vacuum. The states
in this expression have unit norm. A localized
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
energy together. For instance, we can evaluate

```math
C_{nm}(t)=\langle\delta h_n\delta h_m\rangle_t
-\langle\delta h_n\rangle_t\langle\delta h_m\rangle_t,
\qquad
\delta h_n=h_{n,n+1}-\langle h_{n,n+1}\rangle_{\rm vac}I.
```

At sufficiently separated sites the bond operators act on disjoint sites.
Correlations between outgoing regions supplement the one-point profiles.
They can be calculated from the saved MPS; the observable arrays alone do
not contain them. The [320-site example](../demonstrations/ising-collision.md)
includes three-point energy correlations after a collision.

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
directions. The Gram matrix accounts for overlaps between the basis states;
simply summing ``|b_r|^2`` need not give the probability in their span.
When forming ``G^+``, we choose a cutoff below which
to discard eigenvalues of ``G``. Repeating the projection with different
cutoffs shows how much that choice changes the probability. Save the
discarded eigenvalues alongside the probability to record which part of
the spectrum was omitted from the inverse.

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

### Calculating two-particle overlaps in Julia

`simulations/models/ift/scripts/particle_basis.jl` has the functions for
calculating excitation tensors on a saved uniform vacuum. Its
`right_gauge_tensor` function converts a left-gauged excitation to the right
gauge at the same momentum, preserving the momentum eigenstate and its phase.
For each ordered pair of positions, use a left-gauged tensor at the left
insertion and a right-gauged tensor at the right insertion. With these choices,
states at different ordered positions are orthogonal.

`simulations/models/ift/scripts/two_particle_overlap.jl` has the contractions
with the evolved MPS. `two_particle_overlaps` returns the complex overlap for
every allowed pair of positions, and `localized_pair_norms` calculates the
norm of each reference pair. `two_particle_weight` divides the squared
overlaps by those norms and by the evolved state's squared norm before
summing over positions. For one fixed choice of the two excitation tensors,
this gives the weight in their localized-pair subspace.

A tensor calculated at one momentum only approximates a particle branch
over a range of momenta. You can account for its momentum dependence by
calculating tensors at several momenta and expanding them in a common
reference basis. `orthonormal_tensor_basis` constructs that basis, while
`pair_basis_grams` calculates overlaps between reference pairs at the same
positions. The reference vectors are linear combinations of excitation
tensors; their indices do not label particle species.

Write the physical tensors as ``B_L(k)=\sum_a c_{L,a}(k)B_{L,a}`` and
``B_R(q)=\sum_b c_{R,b}(q)B_{R,b}``. For raw position overlaps
``O_{ab}(n,m)``, their momentum-space overlap is

```math
A(k,q)=\frac{1}{L\sqrt{\langle\psi|\psi\rangle}}
\sum_{m-n\ge g} e^{-i(kn+qm)}
\sum_{a,b}c_{L,a}(k)^*c_{R,b}(q)^*O_{ab}(n,m),
```

where ``g`` is the minimum separation and the discrete momenta have spacing
``2\pi/L``. The factor ``1/L`` is the product of the two discrete Fourier
normalizations. Keep the complex coefficients when forming this sum;
summing probabilities over the reference vectors would describe a different
subspace. Rescaling a reference pair also requires rescaling its expansion
coefficient so that the physical tensor remains unchanged.

At large separation, the Gram matrix of orthonormal reference pairs
approaches the identity. Compare it with the identity at the separations
used in the projection, and check the incoming two-particle weight with the
same momentum grid. Repeating the outgoing projection at several late times
and minimum separations shows whether the separation cut still excludes
part of a packet. A higher excitation eigenvalue alone does not identify a
stable species; compare its energy with the multiparticle thresholds and
repeat the excitation calculation with a larger vacuum bond dimension.

### Calculating three-particle overlaps in Julia

`simulations/models/ift/scripts/three_particle_overlap.jl` has the
implementation for three ordered insertions at ``n<m<r``. The first and
middle excitation tensors are left-gauged, and the last is right-gauged.
`three_particle_overlaps` returns the complex amplitudes, grouped by the
outer positions ``(n,r)``. Its `minimum_separation` argument applies to both
``m-n`` and ``r-m``.

Different outer-position pairs are orthogonal in these gauges. With the
outer positions held fixed, two states whose middle insertions differ can
overlap. `middle_position_gram` calculates this matrix, and
`three_particle_weight` uses its inverse on the independent directions to
evaluate ``b^\dagger G^+b/\langle\psi|\psi\rangle``. The returned values
include the smallest and largest Gram eigenvalues and the overlap discarded
by the eigenvalue cutoff.

This calculation keeps one excitation tensor at each of the three
insertions. A particle's tensor changes with momentum, so a full ``111``
projection also requires the momentum-dependent construction described for
pairs above. Finite-separation reference states with two and three
insertions can overlap with one another. `pair_triple_cross_gram` calculates
``c_m=\langle n,m,r|n,r\rangle`` for a specified reference pair; the quantity
``c^\dagger G^+c/\langle n,r|n,r\rangle`` measures how much of that pair
lies in the three-insertion subspace.

The [collision example](../demonstrations/ising-collision.md) gives the
measured weights at several evolved times and separations, together with a
Julia example for calculating them from saved states. Projections onto four
or more particles remain to be implemented. The other collision programs
save only local-observable arrays, from which the many-body state cannot be
reconstructed.
