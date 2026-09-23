# Numerical comparisons

A result should be accompanied by the change observed when its numerical
parameters are varied. Different parameters affect different parts of these
calculations: the vacuum, the incoming state, time evolution, the finite
window, and eventually the outgoing-state projections.

Choose the quantity to compare before changing a parameter. For local
observables measured at matching sites and times, one useful difference is

```math
\Delta_O(t)=\max_{n\in R}|O_n^{(a)}(t)-O_n^{(b)}(t)|,
```

where ``R`` is the same spatial region in both runs. Absolute differences
remain meaningful where a vacuum-subtracted observable crosses zero.
For packet motion, compare positions and widths as well as the full profile;
a small change in velocity can accumulate into a visible displacement at
late times.

## Vacuum and excitation energies

The uniform-vacuum bond dimension sets the number of Schmidt values that the
MPS can retain. Repeat the ground-state calculation with several values of
`--bond_dimension` and compare the energy per site, correlation length,
Schmidt spectrum, and excitation energies at the momenta used by the packets.
All energies must use the same Hamiltonian convention.

A small residual from the ground-state or excitation solver measures how
closely that solver has reached its variational solution. Increasing the bond
dimension tests whether the variational space itself is sufficient. Compare
both effects before assigning digits to a mass or group velocity.

At zero longitudinal field, the transverse-field Ising chain supplies the
exact lattice dispersion, with nearest-neighbour coupling set to one,

```math
\epsilon(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
```

`ift_free_fermion_dispersion` evaluates this expression. Compare the excitation
branch with it at the same momenta in the gapped phase and with the same
vacuum and boundary sector. In the quadratic bosonic limit, the corresponding
comparison is with

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
subspaces. Matrix-element comparisons therefore require consistent phases
and a matching basis, or comparison of the corresponding subspaces. The
stored onsite residual only tests the eigendecomposition in the chosen
oscillator space.

The [scalar-field program](physics/phi4.md) retains `--local_dim` oscillator
states directly. Increase this dimension and compare the same observables.
The shared oscillator operators are projections of the infinite-space
operators: ``P_d\phi^2P_d`` generally differs from ``(P_d\phi P_d)^2`` at the
upper edge of the basis. The [function reference](reference/functions.md)
gives the matrix elements used by the code.

## Packet preparation

For each incoming packet, compare its norm, centre, width, energy, momentum
distribution, and motion before the collision. Propagating a single packet
helps distinguish spreading already present in the initial state from a
change caused by the interaction with the other packet.

Increase the initial separation at fixed packet shape, and increase each
support until the envelope at its ends contributes negligibly to the chosen
observable. Repeat with broader packets to reduce the momentum spread. In
the fixed-momentum construction, this also tests the approximation of using
one tangent tensor throughout that spread.

For a momentum-grid packet, compare the actual sampled momenta and weights.
The requested spacing is rounded, and `--sigma` uses a different convention
from the shared `gaussian_weights` function. Refining `--delta_p` also changes
the support and window size in the current programs. Record those changes
together; a difference between two runs cannot then be assigned solely to
the momentum sum. Details are on the [wave-packet page](physics/wave-packets.md).

## Time step and evolution bond dimension

Compare a run with time step ``\Delta t`` against one with ``\Delta t/2``
at their common physical times. The Ising and scalar-field option
`--total_time` counts samples, including the initial sample. To keep the
same final time while halving the step, change ``T`` to ``2T-1``. The
Schwinger option specifies a physical duration, so it stays unchanged.

The collision programs use two-site TDVP and `truncrank(D_evolution)`.
Increasing `--evolution_bond_dimension` allows more Schmidt values during
the collision and subsequent separation. Its default is twice the vacuum
bond dimension. A maximum rank specifies how many values may be kept; it
does not specify the discarded weight. Compare the evolved observables at
successively larger ranks, especially after the packets overlap.

The Schwinger source quench uses one-site TDVP, so its bond dimension remains
fixed during evolution. Repeat the preparation and evolution at larger
`--D` to test the effect of that restriction.

For a time-independent evolution Hamiltonian, the exact norm and total energy
are constant. Record their drift alongside the local observables. For a
source quench, evaluate the energy with the post-quench Hamiltonian from
``t=0`` onward. The preparation energy belongs to a different Hamiltonian.
Energy conservation by itself does not determine the error in a projected
TDVP trajectory; the bond-dimension comparison tests that separately.

## Window size and elapsed time

An MPS window evolves a finite region inside fixed exterior vacuum tensors.
Enlarge the window while preserving packet widths, momenta, and initial
separation. Compare observables in the common interior, accounting for any
shift in site labels. The finite open Schwinger chain can also reflect waves
from its ends, so compare the chain length at fixed central source region.

The distance to a boundary and the largest populated group velocity give an
estimate of the time available before a signal arrives there. Packet tails
and spreading make a direct comparison between window sizes preferable to
using this estimate as a strict cutoff. Local bond-energy arrays also omit
bonds crossing the window boundary; their sum is not automatically the
conserved total excitation energy.

## Outgoing probabilities

Once outgoing projections are implemented, vary the momentum basis, species
included, minimum particle separation, and time of projection. Compare each
channel probability and their sum. A plateau in time while particles
separate is useful only within the interval before boundary effects appear.
The [particle-production page](physics/particle-production.md) defines the
projection and the treatment of an overlapping basis.

The current programs save local observables, times, and, for the Schwinger
quench, onsite and DMRG residuals. Norm histories, discarded weights, complete
MPS checkpoints, and outgoing probabilities still need to be added. Keep the
command, source revision, Julia package versions, and comparison results with
the data, using separate output directories as described under
[running the calculations](running.md) and [data and figures](data.md).
