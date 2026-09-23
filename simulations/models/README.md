# Simulation entry points

Run every script from the repository root with `--project=julia`. Use `--help`
to inspect its parameters.

| Script | Purpose | Status |
| --- | --- | --- |
| `ift/spectrum.jl` | uMPS vacuum and tangent-space dispersion scan | Maintained diagnostic |
| `ift/collide_fixed_momentum.jl` | Two separated IFT packets at `+k` and `-k` | Preferred IFT collision driver |
| `ift/collide_momentum_grid.jl` | IFT packets assembled from a full commensurate momentum grid | Maintained but more expensive diagnostic |
| `schwinger/source_quench.jl` | Finite-chain five-site source quench | Exploratory; not the paper's scattering protocol |
| `phi4/collide_wavepackets.jl` | Two-packet lattice phi-four evolution | Exploratory |

Collision and quench outputs default to `julia/results/<model>/`. Override the
root with `--output_dir`; raw outputs are intentionally ignored by Git.

Files under `../legacy/` are neither imported nor tested as supported entry
points.
