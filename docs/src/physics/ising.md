# Ising field theory

The Ising calculation starts with a spin chain. Each site has two states, and
the Hamiltonian is

```math
H=-\sum_n\left[J\sigma_n^z\sigma_{n+1}^z
  +g_x\sigma_n^x+g_z\sigma_n^z\right].
```

The matrices ``\sigma^\alpha`` are Pauli matrices, with eigenvalues ``\pm1``.
The lattice spacing and ``\hbar`` are one. The spectrum and fixed-momentum
programs set ``J=1``; the momentum-grid program also accepts `--J`.
The spectrum arguments `--gx` and `--gz` set the two fields. In the collision
programs the corresponding arguments are `--h_x` and `--h_z`.

For positive ``J``, the interaction lowers the energy when neighbouring
spins point in the same ``z`` direction. The transverse field ``g_x`` mixes
the two ``\sigma^z`` states on each site and competes with this alignment.
The longitudinal field ``g_z`` favours one of the two ``z`` orientations.
At ``g_z=0``, reversing every ``z`` spin leaves the Hamiltonian unchanged;
this is its ``\mathbb Z_2`` symmetry. A nonzero longitudinal field breaks
that symmetry explicitly.

For ``J=1``, the critical point is ``(g_x,g_z)=(1,0)``. Approaching it at fixed

```math
\eta_{\rm latt}=\frac{g_x-1}{|g_z|^{8/15}}
```

gives the Ising field theory studied by
[Jha et al.](https://arxiv.org/abs/2411.13645). The ratio ``\eta_{\rm latt}``
is defined through the spin-chain couplings. The paper distinguishes it from
the continuum coupling ratio
``\eta=\tau/|h|^{8/15}``. Energies and times printed by the programs remain in
lattice units. With a nonunit ``J``, the corresponding dimensionless fields
are ``g_x/J`` and ``g_z/J``.

## Vacuum and excitation energies

[`spectrum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/spectrum.jl)
finds a translation-invariant vacuum using a uniform matrix product state
(MPS). A single tensor is repeated along the infinite chain. Its bond
dimension, selected by `--bond-dimension`, limits how much entanglement the
state can contain. VUMPS varies that tensor to minimize the energy per site.

An excitation is obtained by replacing a vacuum tensor ``A`` by ``B(p)`` and
summing over the insertion position with phase ``e^{ipn}``, giving the state

```math
|\Phi_p(B)\rangle=\sum_n e^{ipn}
|\cdots A\,B(p)_n\,A\cdots\rangle.
```

States of this form define the tangent space in which MPSKit solves the
Hamiltonian eigenvalue problem. The program requests one excitation branch
at each momentum and reports its energy relative to the vacuum. Identifying
it with a stable particle requires following the same branch as momentum and
bond dimension change.

The excitation tensor describes a disturbance of the correlated vacuum.
Even though it replaces one tensor, the changes in local expectation values
extend over neighbouring sites. A local spin flip generally creates a
superposition of excitation energies. Solving the tangent-space eigenvalue
problem selects a state with a definite momentum and an approximate
excitation energy. This is the particle used to build a packet in the
[construction of Jha et al.](https://arxiv.org/html/2411.13645v1#S2.SS2).

The spectrum output includes ``m_1=E(0)`` and ``E(p)/m_1``. For two particles
on the same branch with opposite momenta, the central collision energy is
``E_{\rm cm}=2E(p)``. The `--target-cm-ratio` option selects the sampled
momentum nearest the requested ``2E(p)/m_1``; the actual value is printed.
The program writes its table to the terminal.

At ``g_z=0`` the code also compares the excitation energies with

```math
E_{\rm free}(p)=2\sqrt{1+g_x^2-2g_x\cos p}.
```

At ``g_z=0``, the spin chain can be expressed as a quadratic fermion
Hamiltonian. The occupations of its independent quasiparticle modes are
conserved, so two incoming quasiparticles cannot produce additional ones.
For generic nonzero values of both ``g_x-1`` and ``g_z``, the corresponding
continuum theory is interacting and nonintegrable. Scattering can then
populate additional particle sectors when energy and momentum conservation
allow them, as
discussed in the [reference paper](https://arxiv.org/html/2411.13645v1#S1).
The ratios involving ``m_1`` require a nonzero gap and therefore exclude the
critical point itself.

## Two incoming packets

[`collide_fixed_momentum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_fixed_momentum.jl)
uses the tangent tensors at ``+k`` and ``-k``. It multiplies each by a Gaussian
over its half of a finite window, so that the tensor at site ``n`` is

```math
B_L(n)=e^{ik(n-n_L)}e^{-(n-n_L)^2/\sigma^2}B(+k),\qquad
B_R(n)=e^{-ik(n-n_R)}e^{-(n-n_R)^2/\sigma^2}B(-k).
```

Here `--kappa` sets ``k``, `--sigma` sets ``\sigma``, and `--n_center` sets
``n_L``. The right centre is ``n_R=L-n_L``. For the continuous, unbounded
Gaussian, the squared envelope has position standard deviation ``\sigma/2``;
lattice sampling and finite support affect the actual packet variance.
Each half contains one excitation insertion, summed over its possible
positions within that half. The two halves are joined to form the incoming
two-particle state, and the complete window is normalized.

[`collide_momentum_grid.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_momentum_grid.jl)
instead calculates ``B(p)`` on a grid and Fourier sums the tensors. Its
`--sigma` is a width in momentum, and `--mom` sets the central momentum.
The requested `--delta_p` determines an integer ``N``; the actual spacing is
``2\pi/N``, and the window contains ``2N`` sites. Each momentum tensor receives
an independent phase choice in this program. Those phases enter the Fourier
sum and can affect the packet's position and shape. The construction and its
phase dependence are described in [Wave packets](wave-packets.md).

A localized packet contains a range of momenta and hence a range of
excitation energies. Their relative phases change during evolution, causing
the packet to move and spread. For a narrow distribution around momentum
``k``, its velocity is approximately ``dE/dp`` evaluated at ``k``. The
opposite central momenta are chosen so that the two separated packets move
toward each other.

The vacuum extends to infinity on either side of the window. Two-site TDVP,
the time-dependent variational principle applied to adjacent MPS tensors,
evolves the window. `--evolution_bond_dimension` sets the largest retained
bond rank; its default is twice the vacuum bond dimension. Increasing this
rank allows more entanglement to develop during the collision.

## Energy and spin expectation values

Both collision programs measure the symmetric bond density

```math
h_{n,n+1}=-J\sigma_n^z\sigma_{n+1}^z
-\frac12(g_x\sigma_n^x+g_z\sigma_n^z)
-\frac12(g_x\sigma_{n+1}^x+g_z\sigma_{n+1}^z).
```

The array `energy_exp` stores
``\langle h_{n,n+1}\rangle_t-\langle h_{n,n+1}\rangle_{\rm vac}``, while
`s_z_exp` stores
``\langle\sigma_n^z\rangle_t-\langle\sigma^z\rangle_{\rm vac}``.
Subtracting the vacuum value makes the undisturbed background zero. Localized
energy moving away from the collision gives tracks in the energy-density plot.
The spin array records the change in longitudinal magnetization as these
excitations pass through the chain.
Channel probabilities require projections of the evolved state onto outgoing
particle states; see [Particle production](particle-production.md).

For a window of ``L`` sites and ``T`` saved samples, both arrays have ``T``
rows and ``L`` columns. The measured bonds occupy columns `1:L-1` of
`energy_exp`, leaving its last column unused. The stored times are
``t_r=(r-1)\Delta t``, including the initial state. Thus `--total_time` gives
the number of saved samples, and the final time is ``(T-1)\Delta t``.

PNG figures and JLD2 arrays are written under `results/ift/`. Commands are in
[Running the calculations](../running.md), and the saved keys are listed in
[Data and figures](../data.md). Comparisons at different time steps, bond
dimensions, packet widths, and window lengths are described in
[Numerical comparisons](../comparisons.md).
