# A Schwinger source quench

In the bosonized Schwinger model, the local field has an infinite-dimensional
Hilbert space. We first approximate it with a finite oscillator basis, then
keep a few low-energy onsite states for the MPS calculation. This example
uses a 12-site open chain and four retained states per site.

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/schwinger.jl
```

[`schwinger.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/examples/schwinger.jl)
has the example. It calls `get_elems` and `build_hamiltonian` from
`source_quench.jl`, then shows the ground-state calculation and time
evolution explicitly.

## The onsite basis

In the program's notation, the Hamiltonian is

```math
H(J)=\sum_{n=1}^{L}\left[
\frac{\pi_n^2+m^2\phi_n^2}{2}
+\mu^2\bigl(1-\cos(\beta\phi_n-\theta)\bigr)\right]
+\frac{\kappa}{2}\sum_{n=1}^{L-1}(\phi_n-\phi_{n+1})^2
+J\sum_{n\in S}\phi_n.
```

We choose ``m=1``, ``\mu=0.2``, ``\beta=\sqrt{4\pi}``, ``\theta=0``, and
``\kappa=1``. For ``L=12``, the source region is ``S=\{4,5,6,7,8\}``.
The parameter `mu` therefore gives cosine coefficient ``\mu^2=0.04``.
The notation and its relation to the field-theory couplings are explained
on the [Schwinger page](../physics/schwinger.md).

```julia
include("simulations/examples/schwinger.jl")
Quench = SchwingerExample.Quench
basis = Quench.get_elems(4; d=24, beta=sqrt(4pi), mu=0.2, m=1.0, theta=0.0)
basis.energies
```

Here `d=24` is the number of oscillator states used to diagonalize the
single-site term. The first argument, four, is the number of its lowest
eigenvectors retained afterwards. These are different approximations.
Increasing `d` changes the onsite diagonalization; increasing the retained
dimension gives each site more states during the many-body calculation.

`basis.phi` and `basis.phi_sq` are the field and its square projected into
the retained basis. In particular, `basis.phi_sq` is formed by projecting
``\phi^2`` rather than squaring the projected ``\phi``. Intermediate states
outside the retained basis contribute to the matrix elements of ``\phi^2``.

The complete script repeats the onsite calculation with `d=32` and prints
the change in the four retained energies. This comparison tests the
oscillator cutoff at fixed retained dimension. To study the latter, repeat
the chain calculation with more than four retained onsite states.
For these parameters, the four onsite energies at `d=24` are approximately
0.53811816, 1.54896267, 2.52583068, and 3.53995534. Increasing `d` to 32
changes each by less than ``1.7\times10^{-7}``.

## Preparing the state in a source

We first find the ground state with source strength ``J_0=0.2``. A positive
source lowers the energy of negative field values on the source sites.

```julia
using MPSKit, TensorKit, LinearAlgebra, Random
BLAS.set_num_threads(1)
Random.seed!(20260926)
H0 = Quench.build_hamiltonian(12, 4, basis.phi, basis.phi_sq, basis.onsite;
    source_strength=0.2, kappa=1.0)
state = FiniteMPS(12, ℂ^4, ℂ^8)
state, _, residual = find_groundstate(state, H0, DMRG())
normalize!(state)
```

The bond dimension is eight. The density-matrix renormalization group (DMRG)
varies the MPS to minimize the expectation value of ``H(J_0)``. The resulting
field profile is displaced in the source
region and extends into neighbouring sites through the gradient coupling.

## Removing the source

At ``t=0`` we set ``J_1=0``. The prepared state is no longer an eigenstate of
the new Hamiltonian, so its field expectation values evolve.

```julia
H1 = Quench.build_hamiltonian(12, 4, basis.phi, basis.phi_sq, basis.onsite;
    source_strength=0.0, kappa=1.0)
algorithm = TDVP()
envs = environments(state, H1)
state, envs = timestep(state, H1, 0.0, 0.1, algorithm, envs)
real(expectation_value(state, 6 => basis.phi))
```

This example uses one-site TDVP, retaining the bond dimensions of the
prepared MPS. `envs` contains the contractions outside the tensors being
updated; passing it to the next step lets MPSKit reuse those contractions.

The complete example evolves to ``t=1`` in intervals of 0.1. `field.csv`
contains lattice time, site, and ``\langle\phi_n\rangle``. `quench.jld2`
contains the same field array, the norm squared, the expectation value of
``H(J_1)``, both onsite spectra, and the parameters. Field values are saved
directly; there is no subtraction of the initial profile.

To run this interactively and read the central field at every saved time,
use

```julia
result = SchwingerExample.run_example()
result.times
result.field[:, 6]
```

For the supplied example, the central field changes from approximately
``-0.18272`` at ``t=0`` to ``-0.15550`` at ``t=0.5`` and ``-0.08211`` at
``t=1``. Removing the source lets the initially negative displacement move
back towards zero over this interval.

You can extend the evolution with `run_example(final_time=2.0)` or reduce
the time step with `run_example(dt=0.05)`. The saved energy is always
measured with ``H(J_1)``, including at ``t=0``, so values at different times
refer to the same Hamiltonian.
