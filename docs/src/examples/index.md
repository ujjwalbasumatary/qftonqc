# Worked examples

These examples use the same Hamiltonians, packet constructions, and overlap
functions as the longer calculations. You can run them on a laptop. The
Ising window has 64 sites, and the Schwinger chain has 12 sites. Each example
prepares its own data; you do not need the results of a workstation run.

We begin with the Ising vacuum and its single-particle spectrum, prepare two
packets, and evolve them for a short time. A separate six-site example explains
how the overlap functions distinguish states with two and three excitations.
For the Schwinger model, we prepare a ground state in the presence of a source
and evolve it after removing that source.

- [The Ising vacuum and spectrum](ising-spectrum.md)
- [Preparing two packets](ising-packets.md)
- [Time evolution and saved states](ising-evolution.md)
- [Two- and three-excitation overlaps](overlaps.md)
- [A Schwinger source quench](schwinger.md)

## Running an example

Use Julia 1.12.6 with the package versions in `simulations/Manifest.toml`.
From the repository root, install the packages once with

```bash
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.instantiate()'
```

Each page gives a command for the complete example and explains the Julia
functions used in it. You can also start Julia with

```bash
julia --startup-file=no --threads=1 --project=simulations
```

and enter the Julia sections interactively. Including an example file defines
its functions without starting a calculation. When you change a parameter
and run the function again in the same Julia session, Julia can reuse the
code it has already compiled.

The examples use one Julia computation thread and one BLAS thread. Installing
packages and compiling MPSKit for the first time can take several minutes.
The numerical calculations are much shorter once that compilation is done.
A new Julia process can compile these methods again. For repeated changes
to an example, keep the same interactive session open.

## Time and memory use

The following measurements were made on a laptop with an AMD Ryzen 5 5600H,
using Julia 1.12.6 and one computation thread. Each example
was started in a new Julia process and then repeated in that same process.
The first-run times include loading the example and compiling its methods;
package installation is excluded. Peak memory is for the Julia process
across both calls, including compilation.

| Example | First run | Repeat in the same Julia session | Peak memory |
| --- | --- | --- | --- |
| Ising spectrum | 3 min 51 s | 0.54 s | 2.5 GiB |
| Two incoming packets | 4 min 22 s | 0.30 s | 2.8 GiB |
| Ising evolution to ``t=12`` | 6 min 15 s | 47 s | 3.2 GiB |
| Two- and three-excitation overlaps | 11 s | Less than 0.01 s | 0.52 GiB |
| Schwinger evolution to ``t=1`` | 3 min 16 s | 0.50 s | 2.8 GiB |

These are timings for the supplied parameters. To measure the Ising
evolution on your own machine, start an interactive Julia session as above
and run

```julia
@time include("simulations/examples/ising.jl")
@time IsingExamples.evolution()
@time IsingExamples.evolution()
Sys.maxrss() / 2.0^30
```

The final expression reports the largest amount of RAM used by this Julia
process, in GiB.
The two calls create separate output directories. The terminal prints an
explanation of the compilation delay when you launch an Ising or Schwinger example
from the command line, followed by lattice-time progress during evolution.

## Files written by the examples

Every run creates a new directory inside `results/examples/` and prints its
path. For example, the spectrum goes into a directory named
`ising-spectrum-XXXXXX`, where the last six characters distinguish runs.
Running an example again leaves the previous output in place. Git ignores
these directories.

CSV and TOML files can be read in a text editor. The Ising and Schwinger
examples also save Julia arrays in JLD2 files, and the Ising evolution saves
the MPS itself. We reload one of these states in the
[time-evolution example](ising-evolution.md).

The scripts are in
[`simulations/examples/`](https://github.com/ujjwalbasumatary/qftonqc/tree/main/simulations/examples).
The [larger Ising calculations](../demonstrations/ising-collision.md) compare
collisions at zero and nonzero longitudinal field.
