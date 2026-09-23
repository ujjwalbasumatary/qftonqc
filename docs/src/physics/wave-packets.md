# Wave packets

The Ising and scalar-field collision programs start from a uniform MPS vacuum.
A localized excitation is made by replacing a vacuum tensor with an excitation
tensor and summing over its position. Two incoming packets occupy separate
regions of a finite window, with the uniform vacuum continuing outside it.

The functions that assemble these packet tensors are in
[`QFTSimulations.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/src/QFTSimulations.jl).
The [function reference](../reference/functions.md) gives argument types,
dimensions, and normalization conventions.

## Vacuum and excitation tensors

For a one-site uniform MPS, the left- and right-canonical tensors obey

```math
\sum_s (A_L^s)^\dagger A_L^s=I,\qquad
\sum_s A_R^s(A_R^s)^\dagger=I,\qquad
A_L^s C=C A_R^s.
```

Here ``s`` labels the local physical basis and ``C`` is the centre matrix on a
virtual bond. Its singular values are the Schmidt values across that bond.
The arrays `AL`, `AR`, and each excitation tensor `B` have dimensions
`(D, d, D)`, with the indices ordered as left bond, physical index, and right
bond.

A tangent-space excitation of momentum ``p`` has the form

```math
|\Phi_p(B)\rangle
=\sum_n e^{ipn}|\ldots A_L A_L B_n A_R A_R\ldots\rangle.
```

Replacing one tensor changes the many-body wavefunction through its virtual
bonds. The resulting spin or field profile extends over neighbouring sites,
typically over a distance of order the vacuum correlation length. The site
``n`` locates this excitation, while the phase ``e^{ipn}`` combines all its
possible positions into a state of definite momentum. This construction is
described in Sec. II.2 of the
[Ising scattering paper](https://arxiv.org/html/2411.13645v1#S2.SS2).

The effective excitation Hamiltonian determines ``B(p)`` and its energy above
the vacuum. To assign a species to the excitation, the same branch must be
identified across momenta. Each collision program uses the branch returned by
the excitation solver, without tracking several species through crossings.

For a finite packet, `packet[n]` contains the whole site-dependent tensor,
including its envelope and phase. The function
`single_particle_packet_tensors` represents the sum over insertion positions
with upper-triangular tensors,

```math
M_n^s=\begin{pmatrix}A_L^s&B_n^s\\0&A_R^s\end{pmatrix}.
```

The first tensor is the block row ``(A_L\;B_1)`` and the last is the block
column ``(B_N\;A_R)^T``. Any path through their product crosses from the upper
block to the lower block exactly once. Consequently, every term in the state
contains one ``B_n`` insertion. The internal bond dimension is ``2D`` and the
two outer bonds have dimension ``D``. A support containing one site consists
only of its ``B_1`` tensor.

## Two incoming packets

`two_particle_packet_tensors` constructs each packet separately, carrying out
the sum over insertion positions within each support before joining the two
supports. Every term then contains one insertion in the left support and one
in the right support. The bond between the supports has dimension ``D``.

The two supports are joined with ``C^{-1}`` because the first packet ends in a
right-canonical vacuum segment and the second begins with a left-canonical
segment. The relation

```math
A_R^s C^{-1}=C^{-1}A_L^s
```

converts between these descriptions. In the code, `C \ id(domain(C))` solves
for the inverse of `C`, which multiplies the final right bond of the left
packet. Very small Schmidt values make the joined tensor sensitive to rounding
errors in this inverse.

This construction fixes the number of tangent insertions exactly. The packets
represent two incoming particles when their separation is large compared with
their widths and the vacuum correlation length. At smaller separations, the
excitations interact and need not define an exact particle-number sector of
the interacting Hamiltonian.

The shared constructors leave the state unnormalized. The collision programs
normalize the assembled `WindowMPS` before time evolution.

The two insertions specify the initial state. During evolution, `timestep`
updates the tensors throughout the window using the full Hamiltonian. These
updates allow the state to leave its initial two-packet form and acquire
components in other particle sectors when the interactions permit them.

## Fixed central momenta

The Ising program
[`collide_fixed_momentum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_fixed_momentum.jl)
uses one excitation tensor at each of ``+\kappa`` and ``-\kappa``. The two
packets have tensors

```math
B_L(n)=e^{i\kappa(n-n_L)}e^{-(n-n_L)^2/\sigma_x^2}B(+\kappa),
\qquad
B_R(n)=e^{-i\kappa(n-n_R)}e^{-(n-n_R)^2/\sigma_x^2}B(-\kappa).
```

The command-line option `--sigma` is ``\sigma_x`` in lattice-site units.
The left support is `1:L÷2`, its centre is `n_center`, and the right support
is `L÷2+1:L`, with centre `L-n_center`. Truncating each Gaussian to its support
changes the packet if its tails are still appreciable at the support ends.

For the continuous, unbounded envelope, the squared amplitude has position
standard deviation ``\sigma_x/2`` and momentum standard deviation
``1/\sigma_x``. Finite support, lattice sampling, and the excitation tensor's
spatial structure affect the position and momentum moments of the prepared
state. The width of that state therefore needs to be calculated after packet
preparation, rather than read directly from the Gaussian envelope.

The approximation here is to use ``B(\kappa)`` throughout the packet's
momentum spread. Comparing ``B(p)`` over that spread, or comparing with a
packet assembled from several momenta, tests this choice. The position
envelope follows Eq. (14) of the
[Ising scattering paper](https://arxiv.org/abs/2411.13645).

## Packets from a momentum grid

The Ising
[`collide_momentum_grid.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_momentum_grid.jl)
and [scalar-field calculation](phi4.md) use a separate excitation tensor at
each grid momentum. Their sum gives the site-dependent tensor

```math
B_n=\sum_{j=1}^{N}
e^{-\delta p_j^2/\sigma_p^2}e^{ip_j(n-n_0)}B(p_j),\qquad
p_j=-\pi+(j-1)\Delta p,\qquad \Delta p=\frac{2\pi}{N}.
```

The displacement ``\delta p_j`` is wrapped to ``[-\pi,\pi)`` about the selected
central grid point. The requested `--delta_p` is rounded to an integer number
of points through ``N=\operatorname{round}(2\pi/\mathtt{delta\_p})``. Each
packet occupies ``N`` sites and the full window occupies ``2N`` sites. A finer
grid therefore also enlarges the window in these programs.

Here `--sigma` is ``\sigma_p`` in lattice-momentum units. The continuous
Gaussian has squared-amplitude momentum standard deviation ``\sigma_p/2``.
If ``B`` is independent of momentum and the sum is replaced by an integral
over the whole momentum axis, the Fourier transform has position standard
deviation ``1/\sigma_p``. In the programs, the finite sum is periodic over
``N`` sites, and both ``B(p)`` and its phase affect localization.

Each program makes the first component of each ``B(p)`` real by an independent
phase rotation. This phase choice can change abruptly when the component
passes through zero. Overlaps between tensors at neighbouring momenta reveal
such changes; the initial field or energy profile shows how they affect the
localization of the resulting packet.

The sums contain no separate ``\Delta p`` or normalization factor; the full
state is normalized after the supports are joined. The shared function
`gaussian_weights` uses ``e^{-\delta p^2/(2\sigma^2)}`` instead, with explicit
choices for normalizing the weights. Equal momentum envelopes require
``\sigma_p=\sqrt{2}\sigma`` between these conventions. The latter convention
appears in Eq. (S40) of the
[Schwinger paper supplement](https://arxiv.org/html/2307.02522).

Packet separation, support, and width comparisons are described under
[numerical comparisons](../comparisons.md). Their connection to outgoing
particle probabilities is described on the
[particle-production page](particle-production.md).
