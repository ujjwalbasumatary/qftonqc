# Julia tensor-network simulations

This is the reproducible Julia/MPSKit workspace for the Ising field theory
(IFT), bosonized Schwinger-model, and lattice phi-four experiments. The package
foundation and packet-construction tests are working; the IFT and Schwinger
programs are not yet full reproductions of the target papers.

The scientific targets are the [IFT scattering
paper](https://arxiv.org/pdf/2411.13645) and the [Schwinger scattering
paper](https://arxiv.org/abs/2307.02522). See
[`docs/COMPLETION_PLAN.md`](docs/COMPLETION_PLAN.md) for the staged acceptance
criteria.

## Directory layout

```text
julia/
├── Project.toml, Manifest.toml   reproducible Julia environment
├── src/                          shared, tested numerical building blocks
├── test/                         unit and MPS integration tests
├── scripts/
│   ├── ift/                      supported IFT diagnostics and collisions
│   ├── schwinger/                exploratory finite-chain source quench
│   └── phi4/                     exploratory phi-four collision driver
├── results/                      ignored local run outputs
├── docs/                         completion plan and scientific criteria
└── legacy/                       unsupported notebooks and old scripts
```

Only `src/`, `test/`, and `scripts/` are part of the maintained execution path.
The material under `legacy/` is retained for reference and may contain stale
paths, duplicated code, or outdated state construction.

| Area | Current state | Still needed |
| --- | --- | --- |
| `src/`, `test/` | Tested oscillator operators, free dispersions, commensurate grids, Gaussian weights, and exact one-/two-particle packet construction | Keep tests green while extracting more shared model code |
| `scripts/ift/` | uMPS vacuum, tangent-space excitations, two incoming packets, TDVP2 evolution, and local heatmaps | Checkpoint/resume, convergence scans, asymptotic projectors, channel probabilities, and phase shifts/time delays |
| `scripts/schwinger/` | Finite-chain five-site source quench and field heatmap | Paper Hamiltonian interface, uMPS vacua, quark/meson packets, adaptive window, and outgoing projection |
| `scripts/phi4/` | Corrected exploratory two-packet driver | A separately specified validation and reproduction target |

## Set up Julia on Windows

The checked-in environment targets Julia 1.12. From PowerShell at the
repository root:

```powershell
$Julia = "$env:LOCALAPPDATA\Programs\Julia-1.12.6\bin\julia.exe"
& $Julia --startup-file=no --project=.\julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
& $Julia --startup-file=no --project=.\julia -e 'using Pkg; Pkg.test()'
```

The first MPSKit/TensorKit precompile can take several minutes. Later runs reuse
the Julia depot. If Julia is installed elsewhere, change only `$Julia`.

The same Windows executable can be called from WSL:

```bash
JULIA_EXE=/mnt/c/Users/USER/AppData/Local/Programs/Julia-1.12.6/bin/julia.exe
"$JULIA_EXE" --startup-file=no --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
"$JULIA_EXE" --startup-file=no --project=julia -e 'using Pkg; Pkg.test()'
```

## Run the maintained entry points

Run commands from the repository root. Use `--help` before an expensive job;
the defaults are development settings and do not imply convergence. Drivers
write beneath `julia/results/<model>/` by default. Pass `--output_dir` to isolate
a particular scan or run.

IFT fixed-momentum smoke run (the preferred small diagnostic):

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\scripts\ift\collide_fixed_momentum.jl `
    -D 10 --evolution_bond_dimension 20 `
    -L 80 -n 20 -k 0.3 -s 8 -T 20 -t 0.05 -x 1.06 -z 0.01
```

IFT momentum-grid smoke run:

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\scripts\ift\collide_momentum_grid.jl `
    -D 10 -p 0.2 -m 0.3 -s 0.1 -T 20 -t 0.05 -x 1.06 -z 0.01
```

For these IFT drivers, `T` is the number of stored samples, so the final
physical time is `(T - 1) * dt`.

Small Schwinger source-quench diagnostic:

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\scripts\schwinger\source_quench.jl `
    -L 40 -r 8 -D 12 -d 120 -b 3.5449077018 -u 0.7071067812 `
    -k 1.0 -m 0.3162277660 -t 3.1415926536 -T 20 -s 0.05
```

This maps the paper's `mu^2 = 0.1`, `lambda = 0.5`,
`beta = sqrt(4*pi)`, and `theta = pi` onto the exploratory script as follows:

| Paper | Source-quench argument |
| --- | --- |
| field mass `mu^2 = 0.1` | `--m sqrt(0.1)` |
| cosine strength `lambda = 0.5` | `--mu sqrt(0.5)`, because this script uses `mu^2` |
| gradient coefficient `1` | `--kappa 1` |
| `beta = sqrt(4*pi)` | `--beta 3.5449077018` |
| `theta` | `--theta` |

The source-quench protocol is not the incoming-particle construction in the
paper. The small `d=120`, retained dimension `d_trunc=8`, and bond dimension
`D=12` above are smoke-test values. Paper-oriented onsite construction starts
near `d=2000`, `d_trunc=12` and requires convergence checks.

## What is practical on this laptop

The current WSL allocation exposes an AMD Ryzen 5 5600H (6 physical cores) and
about 7.4 GiB of RAM. That is enough for a qualitative IFT inelastic-event
search, but not an interactive reproduction of the paper's production
campaign. First choose momentum from a low-cost spectrum scan:

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\scripts\ift\spectrum.jl `
    -D 6 -n 9 --gx 1.06 --gz 0.006 --k-max 0.7 --target-cm-ratio 6
```

At this coupling `eta_latt` is about `0.919`; a low-bond scan places
`E_cm ≈ 6 m1` near `k ≈ 0.35-0.4`. A deliberately reduced first-look run is:

```powershell
& $Julia --startup-file=no --project=.\julia `
    .\julia\scripts\ift\collide_fixed_momentum.jl `
    -D 6 --evolution_bond_dimension 12 `
    -L 120 -n 30 -k 0.38 -s 10 -T 251 -t 0.1 -x 1.06 -z 0.006
```

This can reveal additional outgoing energy-density tracks, but it is not yet a
particle-production measurement. Promote a promising event to roughly
`L=200-400`, packet width `20-40`, `Dmax=24-32`, and `dt=0.05`, then compare at
least two settings. The paper's converged scale was typically `N=1000-2000`,
`Dmax=64`, and width `70-120`; it reports about 10,000 RK4 steps per week on an
8-physical-core AWS instance. On this laptop that final tier is a batch job of
days to weeks and may exceed the comfortable RAM margin.

For the Schwinger track, the finite source quench is laptop-sized. The paper's
quark runs grow `D=20` toward `D'=50`, while meson runs grow `D=40` toward
`D'=100` with retained onsite dimension 12. Reduced quark runs may fit; meson
runs and systematic projections are better suited to a larger-memory machine.

## Data and run discipline

Generated JLD2 and PNG files are intentionally not versioned on this branch.
Historical artifacts remain available on `main` and in Git history at
`015e907`; the old IFT and phi-four collision files predate the corrected
two-particle sector and are not valid two-particle-scattering evidence.

Before launching expensive scans:

1. Run `Pkg.test()` and a small smoke case.
2. Give each scan a distinct `--output_dir`.
3. Record the Git commit, full arguments, Julia/package versions, wall time,
   random seed when applicable, norm/energy drift, and truncation diagnostics.
4. Keep raw data and convergence tables outside Git; version compact manifests
   and plotting/analysis code.
5. Do not rely on the current drivers for restart: state checkpoint/resume is a
   remaining completion task.
