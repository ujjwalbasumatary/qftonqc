# Numerical comparisons

To compare two evolutions with different time steps or bond dimensions,
evaluate the saved energy or field expectation values at the same sites and
physical times. For a local observable ``O_n(t)``, the largest difference
between runs ``a`` and ``b`` within a region ``R`` is

```math
\Delta_O(t)=\max_{n\in R}|O_n^{(a)}(t)-O_n^{(b)}(t)|,
```

where ``R`` is the same spatial region in both runs. This absolute difference
remains meaningful where a vacuum-subtracted observable crosses zero.
Comparing packet positions and widths alongside the full profile helps
distinguish a change in shape from a displacement. Even a small change in
velocity can accumulate into a visible displacement at late times.

## Vacuum and excitation energies

The uniform-vacuum bond dimension sets the number of Schmidt values that the
MPS can retain. Repeat the ground-state calculation with several values of
`--bond_dimension` and compare the energy per site, correlation length,
Schmidt spectrum, and excitation energies at the momenta used by the packets.
All energies must use the same Hamiltonian convention.

A residual from the ground-state or excitation solver measures how closely
the computed state satisfies the variational equations at the chosen bond
dimension. Increasing the bond dimension enlarges the space of states
available to the solver. Comparing the masses and group velocities as both
the solver tolerance and bond dimension are varied separates these two sources
of numerical error.

At zero longitudinal field, the transverse-field Ising chain supplies the
exact lattice dispersion, with nearest-neighbour coupling set to one,

```math
\epsilon(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
```

`ift_free_fermion_dispersion` evaluates this expression. For the spectrum
program, make this comparison at ``g_z=0`` and ``g_x>1``, using the same
momenta in both calculations. At ``g_z=0`` and ``g_x<1``, a single fermion is a kink
connecting the two ordered vacua. The program uses the same vacuum on both
sides of its excitation tensor and rejects that case. It also excludes the
critical point ``(g_x,g_z)=(1,0)``, where its mass-normalized energy ratios
are undefined. The [Ising page](physics/ising.md) explains the vacuum choice.

In the quadratic bosonic limit, the corresponding comparison is with

```math
\omega(p)=\chi\sqrt{\mu^2+4\sin^2(p/2)},
```

as returned by `schwinger_free_lattice_dispersion`. For the source-quench
Hamiltonian with zero cosine coefficient, its bulk dispersion is instead
``\sqrt{m^2+4\kappa\sin^2(p/2)}``, in the program's units. This is an
infinite-chain formula; an open finite chain has discrete standing-wave modes.
Parameter conventions are given on the [Schwinger page](physics/schwinger.md).

## Onsite bosonic spaces

The Schwinger source quench has two separate truncations. `--d` is the number
of oscillator states used to diagonalize the onsite Hamiltonian.
`--d_trunc` is the number of its eigenstates retained on each lattice site.
First increase `d` at fixed `d_trunc` and compare retained eigenvalues and
projected operators. Then increase `d_trunc` and repeat the many-body
calculation, comparing ground-state energies and the evolved field.

Eigenvectors have arbitrary phases and can rotate within degenerate
subspaces. Before comparing matrix elements from two diagonalizations, align
the eigenvector phases and the bases chosen within degenerate subspaces, or
compare the degenerate subspaces as a whole. The stored onsite residual
measures the error in the eigendecomposition within the chosen oscillator
space. Increasing `d` tests the effect of truncating that space.

The [scalar-field program](physics/phi4.md) retains `--local_dim` oscillator
states directly. Increasing this dimension changes the onsite Hilbert space;
compare the ground-state energies and evolved field expectation values between
these runs. The shared oscillator operators are projections of operators in
the infinite oscillator space. In particular, ``P_d\phi^2P_d`` generally
differs from ``(P_d\phi P_d)^2`` at the upper edge of the basis. The
[function reference](reference/functions.md) gives the matrix elements used
by the code.

## Packet preparation

For each incoming packet, compare its norm, centre, width, energy, momentum
distribution, and motion before the collision. Propagating a single packet
helps distinguish spreading already present in the initial state from a
change caused by the interaction with the other packet.

Increasing the initial separation at fixed packet shape tests how the other
packet affects its motion before the collision. Enlarging each support
reduces the effect of cutting off the envelope at its ends; compare the
packet energy and initial profile as the support grows. Broader packets
reduce the momentum spread. In the fixed-momentum construction, repeating
the calculation at larger widths also tests the approximation of using one
tangent tensor throughout that spread.

For a momentum-grid packet, compare the actual sampled momenta and weights.
The requested spacing is rounded, and `--sigma` uses a different convention
from the shared `gaussian_weights` function. Refining `--delta_p` also changes
the support and window size in the current programs. A comparison between
two values of `--delta_p` therefore includes changes to the momentum sum,
packet support, and window size. Keeping the actual grid and both lengths
with each run makes these changes explicit. Details are on the
[wave-packet page](physics/wave-packets.md).

## Time step and evolution bond dimension

At fixed bond dimension, TDVP approximates the Schrödinger equation by
projecting ``-iH|\psi\rangle`` onto the changes of state obtainable by varying
the MPS tensors. These changes form the tangent space at ``|\psi\rangle``.
Reducing the time step integrates these equations more accurately; increasing
the bond dimension enlarges the set of states they can describe. The
projection and the integration scheme are derived in the
[TDVP paper by Haegeman et al.](https://arxiv.org/abs/1408.5056).

Compare a run with time step ``\Delta t`` against one with ``\Delta t/2``
at their common physical times. The Ising and scalar-field option
`--total_time` counts samples, including the initial sample. To keep the
same final time while halving the step, change ``T`` to ``2T-1``. The
Schwinger option specifies a physical duration, so it stays unchanged.

The collision programs use two-site TDVP and `truncrank(D_evolution)`.
Each update evolves a pair of neighbouring tensors and separates them again
by a singular-value decomposition. Their common bond can grow up to
`--evolution_bond_dimension`, whose default is twice the vacuum bond
dimension. Once the rank exceeds this limit, the additional Schmidt values
are discarded. The discarded weight depends on their magnitudes as well as
their number.

Scattering can entangle outgoing packets, so more Schmidt states can be
needed across a cut even as the packets move apart. Increasing the evolution
bond dimension allows more of these correlations to remain in the MPS.
Compare the evolved energy and field expectation values at successively
larger ranks, especially after the packets overlap.

The Schwinger source quench uses one-site TDVP, so its bond dimension remains
fixed during evolution. Repeat the preparation and evolution at larger
`--D` and compare the field ``\langle\phi_n(t)\rangle`` at matching sites
and times.

For a time-independent evolution Hamiltonian, the exact norm and total energy
are constant. Recording both at each saved time gives their drift over the
evolution. For a source quench, evaluate the energy with the post-quench
Hamiltonian from ``t=0`` onward. The preparation energy belongs to a different
Hamiltonian. Even a TDVP trajectory that conserves energy can depend on the
bond dimension, so the comparison of local observables at different bond
dimensions is still needed.

## Window size and elapsed time

An MPS window evolves a finite region inside fixed exterior vacuum tensors.
Enlarge the window while preserving packet widths, momenta, and initial
separation. Compare observables in the common interior, accounting for any
shift in site labels. The finite open Schwinger chain can also reflect waves
from its ends, so compare the chain length at fixed central source region.

The distance to a boundary and the largest populated group velocity give an
estimate of the time available before a signal arrives there. Packet tails
and spreading affect when the boundary begins to change the interior
observables. Comparing the energy or field profile between windows of
different sizes tests when these effects become visible. Local bond-energy
arrays omit bonds crossing the window boundary, so their sum need not equal
the conserved total excitation energy.

## Outgoing probabilities

Once outgoing projections are implemented, vary the momentum basis, species
included, minimum particle separation, and time of projection. Compare how
each channel probability and their sum change under these variations. Look
for a time interval in which the probabilities remain approximately constant
as the particles separate, before boundary effects appear. The
[particle-production page](physics/particle-production.md) defines the
projection and the treatment of an overlapping basis.

The current programs save local observables, times, and, for the Schwinger
quench, onsite and DMRG residuals. Saving the norm and discarded weights at
each time, saving the complete MPS, and calculating outgoing probabilities
still require additions to the programs. Keep the command, source revision,
Julia package versions, and the differences between runs with the data,
using separate output directories as described under
[running the calculations](running.md) and [data and figures](data.md).
