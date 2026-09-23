# Lattice phi-four theory

[`collide_wavepackets.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/phi4/scripts/collide_wavepackets.jl)
evolves two scalar-particle wave packets on an infinite vacuum. The lattice
Hamiltonian is

```math
H=\sum_n\left[
 \frac{\pi_n^2}{2}+\frac{(\phi_n-\phi_{n+1})^2}{2}
 +\frac{\mu_0^2\phi_n^2}{2}+\frac{\lambda_0}{4!}\phi_n^4
\right].
```

The lattice spacing and ``\hbar`` are one. `--mu_sq` sets the bare coefficient
``\mu_0^2``, and `--lambda` sets ``\lambda_0``. The particle mass is obtained
from the excitation gap of the interacting chain; it need not equal
``\sqrt{\mu_0^2}``.

The ``\pi_n^2/2`` term is the field's kinetic energy, and the gradient term
penalizes differences between neighbouring field values. When
``\lambda_0=0`` and ``\mu_0^2>0``, the chain consists of coupled harmonic
oscillators whose normal modes evolve independently. The quartic interaction
makes the oscillators anharmonic and couples those modes, allowing particle
excitations to scatter. For positive ``\lambda_0``, it also makes large field
values costly in energy. The Hamiltonian is unchanged when both ``\phi`` and
``\pi`` change sign, giving a ``\mathbb Z_2`` field-reflection symmetry.

## Oscillator states and the vacuum

Each site is represented by the first ``d`` harmonic-oscillator states. The
field and its conjugate momentum are expressed as

```math
\phi=\frac{a+a^\dagger}{\sqrt2},\qquad
\pi=\frac{a-a^\dagger}{i\sqrt2}.
```

`--local_dim` selects ``d``. The program fills the matrix elements of
``\phi``, ``\phi^2``, ``\pi^2``, and ``\phi^4`` directly. If ``P_d`` projects
onto the retained states, these are matrices of ``P_dOP_d`` for each
operator ``O``. Taking powers of the truncated field matrix ``P_d\phi P_d``
would omit contributions from intermediate oscillator levels above the
cutoff. The directly evaluated quadratic and quartic matrices retain those
contributions.

The vacuum is a uniform matrix product state with one tensor repeated along
the chain. `--bond_dimension` selects its bond dimension. At the chosen
``d`` and bond dimension, VUMPS varies the tensor to minimize the energy per
site, with a requested tolerance of ``10^{-12}``. Compare the vacuum and its
excitation gap after increasing ``d`` and the bond dimension separately.

The oscillator levels label states on a single site. A particle of the
interacting chain is a collective excitation above the correlated MPS
vacuum. Its energy depends on the coupling between sites and the quartic
interaction, which is why the particle gap must be calculated together with
the vacuum.

## Momentum-space packets

The program solves the tangent-space excitation problem above this vacuum
on the grid

```math
p_i=-\pi+(i-1)\Delta p,\qquad
\Delta p=\frac{2\pi}{N},\qquad i=1,\ldots,N.
```

The requested `--delta_p` is used to choose
``N=\operatorname{round}(2\pi/\mathtt{delta\_p})``. The actual spacing is
printed. Each solution provides an excitation tensor ``B(p_i)``, which
replaces a vacuum tensor in the one-particle ansatz. The collision program
uses the default branch returned by MPSKit.

For a central momentum ``k`` selected by `--momentum`, the packet tensor at
site ``n`` is

```math
B_n=\sum_i e^{-\delta p_i^2/\sigma^2}
 e^{ip_i(n-n_0)}B(p_i).
```

Here ``\delta p_i`` is the shortest displacement around the Brillouin zone
from the grid point nearest ``k``, and `--sigma` gives the momentum width.
For the continuous, unbounded Gaussian, the squared envelope has momentum
standard deviation ``\sigma/2``. The sampled variance also depends on the
grid spacing and wrapping at the Brillouin-zone boundary.
The second packet uses the grid point nearest ``-k``. Each packet occupies
an ``N``-site interval and is centred at local site ``N\div2``; the two
intervals form a window of ``2N`` sites. The normalized window contains one
excitation insertion in each interval.

The phase of each ``B(p_i)`` is chosen independently by making its first
array component real. Since these phases enter the Fourier sum, they can
change the packet's position and shape. The resulting spatial profile and
its dependence on momentum spacing are part of the comparisons described
in [Wave packets](wave-packets.md).

The Hamiltonian remains fixed throughout this calculation. Time dependence
comes from the localized initial state, whose momentum components have
different energies and acquire different phases. The packets propagate
according to the excitation dispersion, and their overlap brings the two
excitations into the region where they interact.

Two-site TDVP evolves the window while its exterior remains the uniform
vacuum. `--evolution_bond_dimension` sets the largest bond rank retained
during evolution. A value of zero, the default, selects twice the vacuum
bond dimension. `--total_time` is the number ``T`` of saved samples, so the
last saved time is ``(T-1)\mathtt{time\_step}``.

## Local observables

The energy density used by the program assigns the onsite term to the left
site of each bond and has the form

```math
h_{n,n+1}=\frac{\pi_n^2}{2}+\frac{\mu_0^2\phi_n^2}{2}
 +\frac{\lambda_0\phi_n^4}{24}+\phi_n^2-\phi_n\phi_{n+1}.
```

Summing this density on the infinite chain gives the Hamiltonian above. Its
local assignment differs from splitting the gradient equally between the
two sites, so that convention matters when comparing individual bond values.

The arrays stored after each time step are

```math
\begin{aligned}
\mathtt{energy\_exp}[r,n]
 &=\langle h_{n,n+1}\rangle_{t_r}
   -\langle h_{n,n+1}\rangle_{\rm vac},\\
\mathtt{phi\_sq\_exp}[r,n]
 &=\langle\phi_n^2\rangle_{t_r}-\langle\phi^2\rangle_{\rm vac}.
\end{aligned}
```

Both arrays have one row for each of the ``T`` saved times and one column
for each of the ``L=2N`` sites, giving shape ``(T,L)``. The measured bonds
occupy columns `1:L-1` of the energy array, and its last column remains zero.
All columns of `phi_sq_exp` are measured. The arrays show the spatial
distribution of excess energy and the change in the mean-square field.
The quantity ``\langle\phi_n^2\rangle`` includes both fluctuations and the
square of the mean field; it equals the variance when
``\langle\phi_n\rangle=0``. Because ``\phi_n^2`` is even under field
reflection, it can reveal a localized excitation even when the state has
this symmetry and its mean field vanishes. Subtracting the vacuum value
removes the vacuum's contribution to this second moment. Particle probabilities
require overlaps with separated outgoing states, as described in
[Particle production](particle-production.md).

Each observable has a PNG figure and a JLD2 file under `results/phi4/`.
The filenames omit the central momentum, so use different `--output_dir`
values when comparing runs that differ only in `--momentum`.
[Data and figures](../data.md) gives the saved keys and loading examples.

The accompanying
[notebooks](https://github.com/ujjwalbasumatary/qftonqc/tree/main/simulations/models/phi4/notebooks)
also calculate finite-chain and uniform ground states, their entanglement,
and the evolution of local field insertions. The
[Julia examples](https://github.com/ujjwalbasumatary/qftonqc/tree/main/simulations/models/phi4/examples)
contain the ground-state and local-insertion calculations as scripts.
