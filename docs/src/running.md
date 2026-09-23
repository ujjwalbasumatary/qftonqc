# Running the calculations

Run the commands below from the repository root. The simulations use Julia
1.12 or a later 1.x release and share the environment in `simulations/`.

```bash
julia --startup-file=no --project=simulations -e \
  'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.test()'
```

The first command installs the packages listed in `Manifest.toml`. The second
runs the tests for oscillator matrix elements, free dispersions, momentum
grids, Gaussian weights, and the packet tensor construction.

Each model program accepts `--help`. For example,

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl --help
```

## Ising spectrum

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/spectrum.jl \
  -D 6 -n 9 --gx 1.06 --gz 0.006 --k-max 0.7 --target-cm-ratio 6
```

This finds a vacuum described by a uniform MPS with bond dimension 6, then
calculates one tangent-space excitation branch at nine momenta. The terminal
output includes the mass ``m_1=E(0)``, a table of ``k``, ``E(k)``, and
``E(k)/m_1``, and the sampled momentum closest to ``2E(k)/m_1=6``.

## Ising collision

The fixed-momentum program builds each packet from one excitation tensor,
evaluated at ``+k`` for one packet and ``-k`` for the other. The command below
uses ``k=0.38``.

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl \
  -D 6 --evolution_bond_dimension 12 \
  -L 120 -n 30 -k 0.38 -s 10 -T 251 -t 0.1 -x 1.06 -z 0.006
```

The window has 120 sites, with packet centres at sites 30 and 90. The
amplitude envelope is ``\exp[-(n-n_0)^2/10^2]``. `T` counts saved times,
including the initial state, so this command ends at ``t=(251-1)0.1=25``.
The energy and spin arrays and their PNG figures go into `results/ift/`.

In the momentum-grid program, each packet is a Fourier sum of excitation
tensors calculated across the Brillouin zone. You can run it with the
following command.

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_momentum_grid.jl \
  -D 10 -p 0.2 -m 0.3 -s 0.1 -T 20 -t 0.05 -x 1.06 -z 0.01
```

Here `-p` requests the momentum spacing and `-s` is the momentum-envelope
width. The actual spacing is ``2\pi/N``, where ``N`` is the nearest integer
to ``2\pi/0.2``; the window contains ``2N`` sites. The two packet conventions
are explained in [Vacua and wave packets](physics/wave-packets.md).

## Schwinger source quench

```bash
julia --startup-file=no --project=simulations \
  simulations/models/schwinger/scripts/source_quench.jl \
  -L 40 -r 8 -D 12 -d 120 -b 3.5449077018 -u 0.7071067812 \
  -k 1.0 -m 0.3162277660 -t 3.1415926536 -T 1.0 -s 0.05
```

This uses 120 oscillator states to diagonalize the onsite Hamiltonian and
keeps its eight lowest eigenstates at each of 40 sites. The program finds
the ground state with source strength `J0`, changes it to `J1`, and saves
``\langle\phi_n(t)\rangle`` during evolution. `--J0` and `--J1` change the
source strengths; their defaults are 1 and 0.2.

For this program, `T` is the final time and `s` is the largest time interval.
If `T/s` is not an integer, the last interval is shortened to reach `T`
exactly. Outputs go into `results/schwinger/`. The
[Schwinger page](physics/schwinger.md) gives the coupling conventions and
the conversion from the saved field to electric flux.

## Lattice ``\phi^4`` theory

The following command keeps the default couplings and packet parameters,
and evolves from ``t=0`` to ``t=1`` in steps of ``0.05``.

```bash
julia --startup-file=no --project=simulations \
  simulations/models/phi4/scripts/collide_wavepackets.jl \
  --local_dim 5 --bond_dimension 10 --evolution_bond_dimension 20 \
  --mu_sq 0.5 --lambda 2.0 --momentum 0.3 --delta_p 0.1 --sigma 0.1 \
  --total_time 21 --time_step 0.05 \
  --output_dir results/phi4/mu0p5_lambda2_k0p3
```

The argument `--local_dim` sets the number of oscillator states retained at
each site. The couplings are `--mu_sq` and `--lambda`; `--momentum` sets the
packet momentum, and `--delta_p` sets the requested spacing of the momentum
grid. As in the Ising momentum-grid program, `--total_time` counts saved
samples and the window has twice as many sites as momentum points. The
program saves the bond energy and ``\phi^2`` expectation value after
subtracting their values in the vacuum.

Here the grid has 63 momenta, so the window has 126 sites. The 21 saved rows
include the initial state. Increase `--total_time` to follow the packets for
longer, keeping the last time before an outgoing packet reaches a window
boundary. The [wave-packet page](physics/wave-packets.md) describes the
momentum weights and their effect on the initial packet shape.

## Workstation runs

You can keep each run's arrays, figures, and terminal output in one directory.
The following command saves them in `results/ift/gx1p06_gz0p006_k0p38/`.

```bash
mkdir -p results/ift/gx1p06_gz0p006_k0p38
JULIA_NUM_THREADS=auto julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl \
  -D 16 --evolution_bond_dimension 32 \
  -L 320 -n 80 -k 0.38 -s 30 -T 801 -t 0.05 -x 1.06 -z 0.006 \
  --output_dir results/ift/gx1p06_gz0p006_k0p38 \
  2>&1 | tee results/ift/gx1p06_gz0p006_k0p38/run.log
```

A `tmux` session keeps a remote calculation running after you disconnect.
The programs save the observables at the end of evolution, so an interrupted
process currently has to be restarted from the beginning.

The calculation time depends on the window length, oscillator cutoff, and
bond dimensions. Compilation adds to the first run. Timing a short evolution
with the intended parameters gives a more useful estimate than the machine's
core count alone. These programs set BLAS to one thread; `JULIA_NUM_THREADS`
sets Julia's thread count separately.

The [numerical comparisons](comparisons.md) page describes how to compare
the energy and field expectation values between runs with different time
steps, bond dimensions, basis sizes, and window lengths.
