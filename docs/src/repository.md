# Repository

The Julia calculations are in `simulations/`. The teaching notebooks and
student submissions are in `course/`. You can collect data and figures from
simulation runs in `results/`.

```text
simulations/
  Project.toml             Julia dependencies
  Manifest.toml            package versions
  src/QFTSimulations.jl     shared operators and MPS constructions
  test/                    tests of the shared functions
  models/
    ift/                   Ising spectrum and two-packet evolution
    schwinger/             bosonized Schwinger source quench
    phi4/                  scalar ground states and packet evolution
course/
  tutorials/               teaching notebooks, numbered by topic
  examples/                anharmonic-oscillator spectrum
  mathematica/             TFIM and string-breaking notebooks
  projects/                student reports and accompanying calculations
results/                   data and figures from local runs
docs/
  src/                     text of this website
  make.jl                  builds the website with Documenter
  build/                   generated website; ignored by Git
```

## Julia calculations

Each directory under
[`simulations/models/`](https://github.com/ujjwalbasumatary/qftonqc/tree/main/simulations/models)
contains programs in `scripts/`, notebooks in `notebooks/`, and, where
available, shorter calculations in `examples/`.
The Hamiltonians and the states evolved by those programs are described on
the [Ising](physics/ising.md), [Schwinger](physics/schwinger.md), and
[scalar-field](physics/phi4.md) pages. Instructions for installing the
packages and running the programs are in [Running the calculations](running.md).

The shared module,
[`QFTSimulations.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/src/QFTSimulations.jl),
contains the oscillator operators, momentum grids, Gaussian weights, and
matrix-product-state tensors used by the model programs. Its docstrings
appear in the [Julia reference](reference/functions.md).

## Course material, results, and documentation

The teaching notebooks and student projects are listed on the
[course page](course.md), with links to their reports and code. The Python
and Mathematica notebooks run independently of the Julia package.

The contents of `results/` are ignored by Git, apart from its README.
The arrays written by the Julia programs and the quantities plotted from
them are described on the [data page](data.md).

This website is built from the Markdown files in `docs/src/` and the Julia
docstrings. The source lives in the same repository as the calculations.
Instructions for building a local copy are in
[`docs/README.md`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/docs/README.md).
