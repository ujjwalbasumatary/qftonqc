# Working on the calculations

The model programs in `simulations/models/*/scripts/` contain the Hamiltonians,
prepare states, and calculate spectra or time evolution. The shared functions in
`simulations/src/QFTSimulations.jl` construct oscillator matrices, momentum
grids, and packet tensors. The notebooks in each model directory contain
separate calculations; editing a notebook does not change the corresponding
command-line program. The [program reference](reference/programs.md) links
each script and describes its functions.

Use `--project=simulations` when running or editing the calculations in Julia.
The environment selected by `--project=docs` contains the packages for building
this website. Installation and run commands are given in
[Running the calculations](running.md).

## Following an Ising collision through the code

In `simulations/models/ift/scripts/collide_fixed_momentum.jl`,
`parse_cmdline()` reads the arguments into a dictionary. At the end of the
file, that dictionary is passed to `main(parsed_args)`. Near the beginning
of `main`, the entries become variables such as `D`, `D_evolution`, `L`,
`T`, `dt`, `κ`, `σ`, `h_x`, and `h_z`.

`get_ham(h_x, h_z)` constructs the infinite MPO `ham`, and `prep_gs(D, ham)`
finds the uniform vacuum `ψ_gs` with VUMPS. The call to
`get_QPstate(ψ_gs, ham, [κ, -κ])` returns the excitation energies and states
at the two packet momenta. `get_B_tensor_list(states)` extracts the tensors
stored in `B_list`.

`create_stacked_tensor` multiplies these tensors by their Gaussian envelopes
and uses the shared `two_particle_packet_tensors` function to place one
excitation in each half of the window. The returned `wavepacket_window`
contains the site tensors. `WindowMPS(ψ_gs, wavepacket_window)` combines
them with the exterior vacuum to form `ψ_window`, which is normalized before
the expectation values are evaluated.

The arrays `energy_exp` and `s_z_exp` are allocated before the evolution loop.
Their first rows contain the initial expectation values after subtracting
the values in `ψ_gs`. The loop over `t_step in 2:T` advances `ψ_window` with
`TDVP2` and fills the subsequent rows. Its rank limit is set by
`truncrank(D_evolution)`. Once the loop finishes, `times` is constructed and
the arrays are plotted and written to JLD2 files.

The momentum-grid Ising program and the scalar-field program follow the same
sequence, but `main()` reads its own arguments. Before preparing the packets,
it constructs `momenta` with `commensurate_momentum_grid` and calculates
an excitation tensor at each momentum. `create_B_packet` forms the Fourier
sum for each site, and `create_stacked_tensor` joins the two packets.
The scalar program saves `phi_sq_exp` in place of `s_z_exp`.

## Following the Schwinger source quench

In `simulations/models/schwinger/scripts/source_quench.jl`, `main()` reads and
checks the arguments, then calls `get_elems` to construct `basis`. This named
tuple contains the retained onsite Hamiltonian, projected field operators,
onsite energies, and projection residual. Both Hamiltonians use these same
operators. `preparation_hamiltonian` has source strength `J0`, and
`quench_hamiltonian` has source strength `J1`.

DMRG finds the ground state of `preparation_hamiltonian` and stores it in
`state`. The program fills the initial row of `field`, then evolves `state`
with `quench_hamiltonian` using one-site TDVP. The array `times` is constructed
by `simulation_times` before the loop, and each step uses the difference
between consecutive entries. At the end, `jldsave` writes the field, times,
arguments, and residuals to one file. `main()` also returns the field, times,
and file paths as a named tuple.

## Changing a Hamiltonian

The Ising collision Hamiltonians are defined in `get_ham`. The scalar
Hamiltonian is also defined in `get_ham`, using the operators returned by
`matrix_elems(d)`. Each collision script constructs `ham_density` separately
inside `main` for the local energy measurements. If an interaction is changed
in `get_ham`, change `ham_density` to include it with the same coefficient.
The Ising density splits each onsite term between adjacent bonds; the scalar
density assigns it to the left site. Their expressions are given in
[Data and figures](data.md).

The energy subtraction is stored in `gs_value_energy`. It is calculated
from `ψ_gs` using the same `ham_density` measured in the evolving state.
After changing the Hamiltonian, prepare the vacuum again so that both the
packet tensors and the subtracted energy refer to that Hamiltonian.
The Ising spectrum program constructs its MPO inside `main(args)` rather
than importing either collision script's `get_ham`; update that definition
as well when comparing the spectrum with an altered collision Hamiltonian.

For the Schwinger program, the onsite Hamiltonian is `onsite_matrix` inside
`get_elems`. The gradient and source terms are added in `build_hamiltonian`.
A change to the onsite Hamiltonian changes its eigenvectors and hence the
retained basis. A new onsite observable can be projected in `get_elems` using
the same `retained_vectors` used for `phi_projected` and `phi_sq_projected`,
then included in the returned named tuple for use in `main`.

## Adding a local observable

The collision scripts evaluate an observable both before the first step and
after every call to `timestep`. For example, the fixed-momentum Ising program
stores its initial spin expectation value with

```julia
s_z_exp[1, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
```

and uses the following expression after a step.

```julia
s_z_exp[t_step, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
```

To add another one-site observable, define its operator alongside `σ_z`,
allocate an array beside `s_z_exp`, and fill it at both locations. If its
vacuum value is to be subtracted, calculate that value from `ψ_gs` beside
`gs_value_s_z`. The existing loop stops at `L-1` because it also evaluates
bond energies, so the one-site value at `L` is filled separately. Include
that site for the new observable as well.

For a two-site observable, the bond-energy calculation gives the operator
placement used by `expectation_value`.

```julia
energy_exp[t_step, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
```

Only the internal bonds `1:L-1` are measured. The energy array has `L`
columns, but its last column is unused. Array shapes and site conventions
are listed in [Data and figures](data.md).

In the Schwinger program, the corresponding expression uses the retained
onsite operator from `basis` and stores the field without vacuum subtraction.

```julia
field[step, site] = real(expectation_value(state, site => basis.phi))
```

Add the new array to the JLD2 save call after the evolution loop, alongside
`times`. The Ising and scalar scripts use `@save`; the Schwinger script uses
`jldsave`. Add a figure if needed, and describe the operator, any subtraction,
and the array indices on the data page so the saved values can be interpreted
without reading the loop.

## Checking an edit

The scripts reuse names such as `get_ham`, `prep_gs`, and `main`. Run them in
separate Julia processes, or include each file in a separate Julia module
when calling its functions interactively. Including a script defines its
functions without starting `main`, because the final call is guarded by
`abspath(PROGRAM_FILE) == @__FILE__`. The fixed-momentum collision and spectrum
programs take a dictionary in `main`; the other three programs read `ARGS`
inside `main()`.

After editing the shared functions, run their tests from the repository root.

```bash
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.test()'
```

These tests compare oscillator matrix elements, dispersion formulas,
momentum grids, Gaussian weights, and particle numbers in a product-vacuum
packet construction. They do not run the model scripts. After changing a
script, use its `--help` command to check the arguments and run a short
evolution into a separate `--output_dir`. Load the resulting JLD2 file to
check the saved keys, array dimensions, and initial and final times. For a
new observable, compare the first saved row with a direct expectation value
in the prepared state. Comparisons between time steps, bond dimensions,
basis sizes, and window lengths are described in
[Numerical comparisons](comparisons.md).

## Calculating outgoing probabilities

The collision scripts currently save local expectation values. To calculate
outgoing probabilities, retain `ψ_window` at the chosen late times, construct
outgoing particle states from identified excitation branches, and evaluate
their overlaps with it. Saving the complete state or performing those
projections inside `main` requires adding code before the state is discarded
when the function returns. The required overlaps and normalization are
described in [Particle production](physics/particle-production.md); they
have not yet been implemented in the programs.
