# Quantum field theory on a quantum computer

This repository contains material from HE381, *Quantum Field Theory on a
Quantum Computer*, taught by Prof. Aninda Sinha at IISc in the fall of 2025,
and the tensor-network calculations that grew out of the course.

The Julia calculations follow two references:

- [Real-Time Scattering in Ising Field Theory using Matrix Product States](https://arxiv.org/abs/2411.13645),
- [High-Energy Collision of Quarks and Mesons in the Schwinger Model](https://arxiv.org/abs/2307.02522).

The Ising calculation constructs two incoming tangent-space excitations and
evolves them with TDVP. It saves local energy and spin expectation values.

The Schwinger calculation evolves a finite-chain ground state after a local
source quench and saves the field expectation value at each site.

## Files

```text
simulations/             Julia package and current calculations
  models/ift/            Ising spectrum and collisions
  models/schwinger/      Schwinger source quench
  models/phi4/           lattice phi-four calculation
course/tutorials/        notebooks used in class
course/examples/         longer numerical examples
course/projects/         student reports, notebooks, and presentations
results/                 data and figures from local runs
docs/                    documentation website
```

Each model directory contains its Julia programs together with related
notebooks and examples. Student submissions are under `course/projects/`.

## Julia environment

The calculations require Julia 1.12 or a later 1.x release. From the
repository root, run

```bash
julia --startup-file=no --project=simulations -e \
  'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.test()'
```

The first MPSKit compilation can take several minutes. Commands for the model
calculations are in [`simulations/README.md`](simulations/README.md). Outputs
can be stored in `results/ift/`, `results/schwinger/`, and `results/phi4/`;
Git ignores their contents.

The remaining calculations are listed in
[`simulations/STATUS.md`](simulations/STATUS.md).
The documentation source is in [`docs/`](docs/), including commands for
building and viewing the website locally.
