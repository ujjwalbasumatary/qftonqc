# Tensor-network calculations

The package `QFTSimulations` contains the numerical constructions shared by
the Ising-field-theory, bosonized Schwinger, and lattice $\phi^4$
calculations. The model Hamiltonians and time evolutions are in `models/`.

## Installation

The environment supports Julia 1.12 and later 1.x releases. From the
repository root, run

```bash
julia --startup-file=no --project=simulations -e \
  'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.test()'
```

The first MPSKit and TensorKit compilation can take several minutes.

## Ising field theory

The spectrum calculation finds a one-site uMPS vacuum and evaluates the lowest
tangent-space excitation branch:

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/spectrum.jl \
  -D 6 -n 9 --gx 1.06 --gz 0.006 --k-max 0.7 --target-cm-ratio 6
```

It prints the mass, the sampled dispersion, and the momentum for which a
symmetric collision has energy closest to $6m_1$.

The fixed-momentum collision is

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl \
  -D 6 --evolution_bond_dimension 12 \
  -L 120 -n 30 -k 0.38 -s 10 -T 251 -t 0.1 -x 1.06 -z 0.006
```

Here `T` is the number of saved time slices, including $t=0$; the final time
is `(T-1)dt`. The program saves the vacuum-subtracted bond-energy density and
$\langle\sigma^z_n\rangle-\langle\sigma^z\rangle_{\rm vac}$ under
`results/ift/`. A different directory can
be supplied with `--output_dir`.

The momentum-grid construction is

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_momentum_grid.jl \
  -D 10 -p 0.2 -m 0.3 -s 0.1 -T 20 -t 0.05 -x 1.06 -z 0.01
```

This version Fourier sums excitation tensors from a commensurate momentum
grid. The phase of each tensor is currently fixed independently. Neighboring
momenta need compatible phases; otherwise their Fourier sum can shift,
distort, or delocalize the packet.

## Schwinger source quench

The current Schwinger calculation prepares a finite-chain ground state with a
source on the central sites and changes the source strength before TDVP
evolution:

```bash
julia --startup-file=no --project=simulations \
  simulations/models/schwinger/scripts/source_quench.jl \
  -L 40 -r 8 -D 12 -d 120 -b 3.5449077018 -u 0.7071067812 \
  -k 1.0 -m 0.3162277660 -t 3.1415926536 -T 1.0 -s 0.05
```

`d` is the oscillator cutoff and `r` is the number of onsite eigenstates kept
after diagonalization. The calculation in arXiv:2307.02522 used an oscillator
cutoff near 2000 and kept 12 onsite states. Increase `d` at fixed `r` and
compare the retained energies and matrix elements; then repeat while
increasing `r`.

The parameter names in this program differ from those in the paper. They are
related by

| Hamiltonian parameter | Command argument |
| --- | --- |
| $\mu^2=0.1$ | `--m sqrt(0.1)` |
| $\lambda=0.5$ | `--mu sqrt(0.5)` |
| $\kappa$ in $\frac{\kappa}{2}(\phi_n-\phi_{n-1})^2$ | `--kappa 1` |
| $\beta=\sqrt{4\pi}$ | `--beta 3.5449077018` |
| $\theta$ | `--theta` |

The program saves $\langle\phi_n(t)\rangle$ as `field`. The `flux` key is an
alias for the same array. The convention in arXiv:2307.02522 is
$E_T/e=\phi/\sqrt{\pi}$.

## Longer calculations

A longer Ising run with `L=320`, packet width `30`, evolution bond dimension
`32`, and `dt=0.05` can be collected in its own directory. The command below
saves the terminal output there as well:

```bash
mkdir -p results/ift/gx1p06_gz0p006_k0p38
JULIA_NUM_THREADS=auto julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl \
  -D 16 --evolution_bond_dimension 32 \
  -L 320 -n 80 -k 0.38 -s 30 -T 801 -t 0.05 -x 1.06 -z 0.006 \
  --output_dir results/ift/gx1p06_gz0p006_k0p38 \
  2>&1 | tee results/ift/gx1p06_gz0p006_k0p38/run.log
```

For a remote run, a persistent shell such as `tmux` keeps the calculation
running after you disconnect. The programs save observables at the end of the
run; saving intermediate MPS states for restart is still to be added.

The remaining calculations are listed in [`STATUS.md`](STATUS.md).
