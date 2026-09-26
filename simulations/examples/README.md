# Worked examples

These scripts accompany the
[worked examples in the documentation](https://ujjwalbasumatary.github.io/qftonqc/examples/).
Run the commands below from the repository root after installing the packages
in `simulations/Project.toml` with the checked-in Manifest.

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl spectrum
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl packets
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl evolution
julia --startup-file=no --threads=1 --project=simulations simulations/examples/overlaps.jl
julia --startup-file=no --threads=1 --project=simulations simulations/examples/schwinger.jl
```

Each command prepares its own data and prints the path of a new directory
inside `results/examples/`. Previous results are left in place. The Ising
examples use a 64-site window at most; the Schwinger chain has 12 sites.
Both use one computation thread and one BLAS thread.

For an interactive calculation, start Julia with
`julia --startup-file=no --threads=1 --project=simulations`, then enter

```julia
include("simulations/examples/ising.jl")
result = IsingExamples.spectrum()
```

Including a file defines its functions without running the example. Keeping
this Julia session open avoids compiling the same methods again when you
change parameters. The first calculation can take several minutes to compile.

The Ising examples call the Hamiltonian, vacuum, excitation, and packet
functions in `models/ift/scripts/collide_fixed_momentum.jl`. The Schwinger
example calls the onsite-basis and Hamiltonian functions in
`models/schwinger/scripts/source_quench.jl`. The overlap example uses a
six-site state with specified excitation counts and calls the pair and
triple contraction functions in `models/ift/scripts/`.
