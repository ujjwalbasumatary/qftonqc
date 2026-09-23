# Model calculations

Run these programs from the repository root with `--project=simulations`.
Each program accepts `--help`.

| Script | Calculation |
| --- | --- |
| `ift/scripts/spectrum.jl` | Finds the uniform MPS vacuum and calculates excitation energies at the requested momenta. |
| `ift/scripts/collide_fixed_momentum.jl` | Evolves two Ising packets built from excitation tensors at `+k` and `-k`. |
| `ift/scripts/collide_momentum_grid.jl` | Constructs two Ising packets by summing excitation tensors over a momentum grid, then evolves their collision. |
| `schwinger/scripts/source_quench.jl` | Finds a finite-chain ground state with a central source, changes the source strength, and follows the field expectation value. |
| `phi4/scripts/collide_wavepackets.jl` | Evolves two scalar-field packets and measures the local energy and $\phi^2$ expectation value. |

Outputs go to `results/<model>/` unless `--output_dir` is supplied. Git ignores
the contents of the results directory.

The `notebooks/` and `examples/` directories contain ground-state, spectrum,
and local-excitation calculations. Each model's README describes the files
in that directory. The command-line programs listed above use the shared
functions in `simulations/src/`.
