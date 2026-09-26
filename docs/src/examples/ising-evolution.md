# Time evolution and saved states

We now evolve the two packets from the preceding example. This command
prepares them again, so you can run it independently.

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl evolution
```

The parameters are ``h_x=1.5``, ``h_z=0``, ``L=64``, ``\kappa=0.6``,
``\sigma=6``, and packet centres 16 and 48. We use a vacuum bond dimension
of four and allow the bond dimension to grow to twelve during evolution.
The time step is ``\Delta t=0.1`` and the final lattice time is ``t=12``.

For the exact dispersion, the group velocity is ``v(p)=dE(p)/dp``. At
``p=0.6`` its magnitude is approximately 1.93 sites per unit lattice time.
The packet centres are 32 sites apart, so they should meet near
``t=32/(2v)\simeq8.3``. This estimate uses the central momentum; different
momenta within each packet have different velocities and cause it to spread.

## A TDVP time step

The time-dependent variational principle approximates Schrödinger evolution
within the space of MPS. We use the two-site method so that the bond
dimensions can grow. After each two-site update, a singular-value
decomposition retains at most twelve states on the bond.

You can perform one time step interactively with

```julia
include("simulations/examples/ising.jl")
using MPSKit, TensorKit
prep = IsingExamples.prepare_packets()
algorithm = TDVP2(; trscheme=truncrank(12))
state, _ = timestep(prep.state, prep.ham, 0.0, 0.1, algorithm)
measured = IsingExamples.measure(state, prep)
```

The time arguments are the starting time and the interval, so this call
advances from ``t=0`` to ``t=0.1``. `truncrank(12)` specifies a maximum
retained dimension; it does not specify an allowed discarded weight.

For the complete example, call

```julia
result = IsingExamples.evolution()
```

The terminal prints the lattice time, norm squared, and summed internal bond
energy after every interval. We do not renormalize the state between steps.
The same numbers go into `evolution.csv` as the calculation proceeds.
MPSKit divides local expectation values by the current norm squared. The
saved MPS retains the norm printed in the terminal.

## Reading the saved arrays

`observables.jld2` contains `times`, `energy`, `spin`, `norm_squared`, and
the parameters. The energy array has 121 rows and 63 columns. Row one is
``t=0``; the last row is ``t=12``. Column ``n`` contains ``\delta e_n`` on
bond ``(n,n+1)``. The spin array has 64 columns and contains
``\langle\sigma_n^z\rangle-\langle\sigma^z\rangle_\Omega``.
At ``h_z=0``, the Hamiltonian and this two-packet state preserve spin-flip
symmetry, while ``\sigma^z`` changes sign under it. Its expectation value
therefore stays near zero in the numerical calculation. Use the energy
distribution to examine the motion of these packets.

```julia
using JLD2
data = load(result.data_path)
data["times"]
data["energy"][end, :]
initial_energy = sum(data["energy"][1, :])
final_energy = sum(data["energy"][end, :])
(final_energy - initial_energy) / initial_energy
```

This sum includes the bonds inside the window. While the packets remain
far from its ends, its change is useful for comparing choices of time step
and bond dimension. To halve the time step, use
`IsingExamples.evolution(dt=0.05)`. To change the maximum bond dimension,
edit `evolution_bond_dimension` in `prepare_packets`.

With the supplied parameters and seed, the summed energy changes from
3.54168142 at ``t=0`` to 3.54170397 at ``t=12``, an increase of about
0.00064%. The final norm squared is 0.99999107. These are comparisons you
can repeat after changing the numerical parameters.

## Reloading an MPS

The example saves states at ``t=0,3,6,9,12`` in its `states/` directory.
These files include the vacuum, Hamiltonian, preparation tensors, parameters,
and local observables alongside the MPS. The program prints the final path.

```julia
saved = IsingExamples.load_ift_state(last(result.paths))
state = saved["state"]
saved["time"]
saved["parameters"]
```

In a new Julia session, include `simulations/examples/ising.jl` and replace
`last(result.paths)` with the printed file path. You can then measure another
observable without repeating the evolution. For example,

```julia
sx, _, _ = IsingExamples.Collision.get_ops()
real(expectation_value(state, 32 => sx))
```

The complete script reloads the final state, measures the central bond energy
again, and prints its difference from the value stored during evolution.
For the supplied example, that difference is zero. The five MPS files
occupy about 5.7 MB in total.
Keep the Julia environment used to write the state when reloading these
objects. The [data page](../data.md) describes the saved file format.
