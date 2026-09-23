# Model calculations

Run these programs from the repository root with `--project=simulations`.
Each program accepts `--help`.

| Script | Calculation |
| --- | --- |
| `ift/scripts/spectrum.jl` | uMPS vacuum and tangent-space dispersion |
| `ift/scripts/collide_fixed_momentum.jl` | two Ising packets built at momenta `+k` and `-k` |
| `ift/scripts/collide_momentum_grid.jl` | two Ising packets Fourier summed over a momentum grid |
| `schwinger/scripts/source_quench.jl` | finite-chain source quench in the bosonized Schwinger Hamiltonian |
| `phi4/scripts/collide_wavepackets.jl` | two-packet lattice $\phi^4$ evolution |

Outputs go to `results/<model>/` unless `--output_dir` is supplied. Git ignores
the contents of the results directory.

The `notebooks/` and `examples/` directories contain additional calculations
for reading and modification. The command-line programs listed above use the
shared package in `simulations/src/`.
