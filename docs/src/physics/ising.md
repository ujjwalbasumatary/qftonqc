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

For ``J=1``, the critical point is ``(g_x,g_z)=(1,0)``. Approaching it at fixed

```math
\eta_{\rm latt}=\frac{g_x-1}{|g_z|^{8/15}}
```

gives the Ising field theory studied by
[Jha et al.](https://arxiv.org/abs/2411.13645). This is a lattice parameter:
the paper distinguishes it from the continuum coupling ratio
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
summing over the insertion position with phase ``e^{ipn}``:

```math
|\Phi_p(B)\rangle=\sum_n e^{ipn}
|\cdots A\,B(p)_n\,A\cdots\rangle.
```

MPSKit solves the Hamiltonian eigenvalue problem in this tangent space. The
program requests one branch at each momentum and reports its energy relative
to the vacuum. Identifying it with a stable particle requires following that
branch as momentum and bond dimension change.

The spectrum output includes ``m_1=E(0)`` and ``E(p)/m_1``. For two particles
on the same branch with opposite momenta, the central collision energy is
``E_{\rm cm}=2E(p)``. The `--target-cm-ratio` option selects the sampled
momentum nearest the requested ``2E(p)/m_1``; the actual value is printed.
The program writes its table to the terminal.

At ``g_z=0`` the code also compares the excitation energies with

```math
E_{\rm free}(p)=2\sqrt{1+g_x^2-2g_x\cos p}.
```

The ratios involving ``m_1`` require a nonzero gap, so the critical point
itself is unsuitable for this mass-normalized scan.

## Two incoming packets

[`collide_fixed_momentum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_fixed_momentum.jl)
uses the tangent tensors at ``+k`` and ``-k``. It multiplies each by a Gaussian
over its half of a finite window:

```math
B_L(n)=e^{ik(n-n_L)}e^{-(n-n_L)^2/\sigma^2}B(+k),\qquad
B_R(n)=e^{-ik(n-n_R)}e^{-(n-n_R)^2/\sigma^2}B(-k).
```

Here `--kappa` sets ``k``, `--sigma` sets ``\sigma``, and `--n_center` sets
``n_L``. The right centre is ``n_R=L-n_L``. For the continuous, unbounded
Gaussian, the squared envelope has position standard deviation ``\sigma/2``;
lattice sampling and finite support affect the actual packet variance.
Each half contains one excitation
insertion, summed over its possible positions. The two halves are joined to
form the incoming two-particle state, and the complete window is normalized.

[`collide_momentum_grid.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_momentum_grid.jl)
instead calculates ``B(p)`` on a grid and Fourier sums the tensors. Its
`--sigma` is a width in momentum, and `--mom` sets the central momentum.
The requested `--delta_p` determines an integer ``N``; the actual spacing is
``2\pi/N``, and the window contains ``2N`` sites. Each momentum tensor receives
an independent phase choice in this program. Those phases enter the Fourier
sum and can affect the packet's position and shape. The construction and its
phase dependence are described in [Wave packets](wave-packets.md).

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
Channel probabilities require projections of the evolved state onto outgoing
particle states; see [Particle production](particle-production.md).

Both arrays have ``T`` rows and ``L`` columns. The last energy column is
unused; the measured bonds occupy columns `1:L-1`. The stored times are
``t_r=(r-1)\Delta t``, including the initial state. Thus `--total_time` gives
the number of saved samples, and the final time is ``(T-1)\Delta t``.

PNG figures and JLD2 arrays are written under `results/ift/`. Commands are in
[Running the calculations](../running.md), and the saved keys are listed in
[Data and figures](../data.md). Comparisons at different time steps, bond
dimensions, packet widths, and window lengths are described in
[Numerical comparisons](../comparisons.md).
