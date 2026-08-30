# Julia tensor-network simulations

This directory contains Julia/MPSKit experiments for Ising field theory (IFT),
the bosonized Schwinger model, and lattice phi-four theory. The environment and
core numerical helpers are reproducible; the IFT and Schwinger programs are not
yet complete reproductions of the papers.

| Area | Current state | What is still needed |
| --- | --- | --- |
| `src/`, `test/` | Tested oscillator operators, free dispersions, momentum grids, Gaussian weights, and one-particle/two-particle packet construction | Keep these tests green as simulation code is extended |
| `ift/` | Runnable uMPS ground state, tangent-space excitations, two incoming packets, TDVP evolution, and local heatmaps | Asymptotic particle projection, channel probabilities, phase shifts/time delays, convergence scans, and checkpoints |
| `schwinger/` | Runnable finite-chain five-site source quench and electric-flux heatmap | Paper Hamiltonian interface, uMPS vacua, topological quark/meson packets, window growth, asymptotic projection, and paper observables |
| `phi4/` | Exploratory ground-state and scattering scripts | A separately specified validation and reproduction target |

The scientific targets are [the IFT scattering paper
(arXiv:2411.13645)](https://arxiv.org/pdf/2411.13645) and [the Schwinger-model
scattering paper (arXiv:2307.02522)](https://arxiv.org/abs/2307.02522). See
[`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) for the staged route and acceptance
criteria.

## Set up Julia on Windows

The checked-in `Project.toml` and `Manifest.toml` target Julia 1.12. From
PowerShell at the repository root:

```powershell
$Julia = "$env:LOCALAPPDATA\Programs\Julia-1.12.6\bin\julia.exe"
& $Julia --startup-file=no --project=.\julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
& $Julia --startup-file=no --project=.\julia -e 'using Pkg; Pkg.test()'
```

The first MPSKit/TensorKit precompile can take several minutes. Subsequent runs
reuse the depot. If Julia is installed elsewhere, change only `$Julia`.

The same Windows executable can be called from WSL:

```bash
JULIA_EXE=/mnt/c/Users/USER/AppData/Local/Programs/Julia-1.12.6/bin/julia.exe
"$JULIA_EXE" --startup-file=no --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
"$JULIA_EXE" --startup-file=no --project=julia -e 'using Pkg; Pkg.test()'
```

The simulation drivers resolve their default `plots/` and `data/` directories
from the script location, so launching them from the repository root or their
own directory produces the same output paths.

## Run the exploratory programs

Use `--help` before a run; CLI defaults are development values, not a claim of
paper-level convergence.

IFT fixed-momentum smoke run (the preferred small diagnostic):

```powershell
Push-Location .\julia\ift
& $Julia --startup-file=no --project=.. .\scattering_infinite_fixed_m.jl `
    -D 10 --evolution_bond_dimension 20 `
    -L 80 -n 20 -k 0.3 -s 8 -T 20 -t 0.05 -x 1.06 -z 0.01
Pop-Location
```

IFT momentum-grid smoke run:

```powershell
Push-Location .\julia\ift
& $Julia --startup-file=no --project=.. .\scattering_infinite.jl `
    -D 10 -p 0.2 -m 0.3 -s 0.1 -T 20 -t 0.05 -x 1.06 -z 0.01
Pop-Location
```

In these scripts, `T` is the number of stored samples, so the final physical
time is approximately `(T - 1) * dt`. The fixed-momentum run writes JLD2 arrays
and a heatmap below `ift/fixed_mom/`; the momentum-grid run writes below
`ift/data/` and `ift/plots/`.

Small Schwinger source-quench diagnostic:

```powershell
Push-Location .\julia\schwinger
& $Julia --startup-file=no --project=.. .\schwinger_string_breaking.jl `
    -L 40 -r 8 -D 12 -d 120 -b 3.5449077018 -u 0.7071067812 `
    -k 1.0 -m 0.3162277660 -t 3.1415926536 -T 20 -s 0.05
Pop-Location
```

This parameter choice maps the paper's `mu^2 = 0.1`, `lambda = 0.5`,
`beta = sqrt(4*pi)`, and `theta = pi` to the current script as follows:

| Paper | Current script |
| --- | --- |
| field mass `mu^2 = 0.1` | `--m sqrt(0.1)` |
| cosine strength `lambda = 0.5` | `--mu sqrt(0.5)`, because the code uses `mu^2` |
| gradient coefficient `1` | `--kappa 1` |
| `beta = sqrt(4*pi)` | `--beta 3.5449077018` |
| `theta` | `--theta` |

The script additionally quenches a five-site linear source from `J0` to `J1`.
That source protocol is not the incoming-particle construction in the paper.
The small `d=120`, retained dimension `d_trunc=8`, and bond dimension `D=12`
above are only for a smoke run. Paper-oriented onsite construction starts near
`d=2000`, `d_trunc=12` and must be convergence checked. Output is written to
`schwinger/plots/` and `schwinger/data/`.

## What is practical on this laptop

The current WSL allocation exposes an AMD Ryzen 5 5600H (6 physical cores) and
about 7.4 GiB of RAM. That is enough to see a qualitative IFT inelastic event,
but not to reproduce the paper's production campaign interactively. First use
the spectrum utility to choose a momentum for the desired center-of-mass
energy:

```powershell
& $Julia --startup-file=no --project=.\julia .\julia\scripts\ift_spectrum.jl `
    -D 6 -n 9 --gx 1.06 --gz 0.006 --k-max 0.7 --target-cm-ratio 6
```

At this coupling `eta_latt` is about `0.919`; a low-bond scan places
`E_cm approximately 6 m1` near `k approximately 0.35-0.4`. A deliberately
reduced first-look run is:

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\ift\scattering_infinite_fixed_m.jl `
    -D 6 --evolution_bond_dimension 12 `
    -L 120 -n 30 -k 0.38 -s 10 -T 251 -t 0.1 -x 1.06 -z 0.006
```

This can reveal additional outgoing energy-density tracks, but it is not a
particle-production measurement: the packets are narrow and the truncations
are intentionally aggressive. Promote a promising event to roughly
`L=200-400`, packet width `20-40`, `Dmax=24-32`, and `dt=0.05`, then compare at
least two settings. The paper's converged scale was typically `N=1000-2000`,
`Dmax=64`, and width `70-120`; it reports about 10,000 RK4 steps per week on an
8-physical-core AWS instance. On this laptop that final tier is a batch job of
days to weeks and may exceed the comfortable RAM margin.

For the Schwinger track, the finite source quench above is laptop-sized. The
paper's actual quark runs grow `D=20` to `D'=50`, while meson runs grow `D=40`
to `D'=100` with a retained onsite dimension of 12. The former may be possible
as a long reduced run; the latter and systematic channel projections are much
better suited to a larger-memory workstation.

## Data and run discipline

Existing JLD2 files mostly contain observable matrices; filenames encode the
parameters. Before launching expensive scans:

1. Run `Pkg.test()` and a small smoke case.
2. Use a new output directory for each scan; never treat an existing PNG as
   evidence of convergence.
3. Record the git commit, full argument set, Julia/package versions, wall time,
   random seed (when applicable), norm/energy drift, and truncation diagnostics.
4. Keep raw JLD2 data and convergence tables. Generate figures from data in a
   separate analysis step.
5. Do not rely on the current scripts for restart: state checkpoints and resume
   support remain completion tasks.

The `plot_and_push.sh` files are legacy convenience scripts with repository
side effects. Review them before use; they are not part of the simulation or
validation workflow.
