# Tensor-network calculations

The Julia programs in `models/` calculate ground states and time evolution
for Ising field theory, the bosonized Schwinger model, and lattice $\phi^4$
theory. They share the package `QFTSimulations`, which contains oscillator
matrices, formulas for the free dispersions, and functions that assemble
wave packets as matrix product states (MPS).

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

The spectrum program represents the vacuum by a single MPS tensor repeated
along the infinite chain. It then calculates the lowest excitation energy
returned by the tangent-space calculation at each momentum. The following
command samples nine momenta between zero and `0.7`.

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/spectrum.jl \
  -D 6 -n 9 --gx 1.06 --gz 0.006 --k-max 0.7 --target-cm-ratio 6
```

It prints the mass, the sampled dispersion, and the momentum for which a
symmetric collision has energy closest to $6m_1$.

To prepare two packets from excitation tensors at opposite momenta and
evolve their collision, run

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_fixed_momentum.jl \
  -D 6 --evolution_bond_dimension 12 \
  -L 120 -n 30 -k 0.38 -s 10 -T 251 -t 0.1 -x 1.06 -z 0.006
```

Here `T` is the number of saved times, including $t=0$, so the final time
is `(T-1)dt`. At each time the program measures the energy density on the
bonds and subtracts its value in the vacuum. It saves this array together with
$\langle\sigma^z_n\rangle-\langle\sigma^z\rangle_{\rm vac}$ under
`results/ift/`. You can choose a different directory with `--output_dir`.

The second collision program constructs each packet by summing excitation
tensors calculated at a set of equally spaced momenta.

```bash
julia --startup-file=no --project=simulations \
  simulations/models/ift/scripts/collide_momentum_grid.jl \
  -D 10 -p 0.2 -m 0.3 -s 0.1 -T 20 -t 0.05 -x 1.06 -z 0.01
```

The phase of each excitation tensor enters this Fourier sum. The program
chooses those phases independently, so they need not vary continuously
between neighbouring momenta. Abrupt phase changes can shift or distort the
packet, or spread it across the window.

## Schwinger source quench

The Schwinger program finds the ground state of a finite chain with a source
on the central sites. It then changes the source strength and evolves the
state using the time-dependent variational principle (TDVP).

```bash
julia --startup-file=no --project=simulations \
  simulations/models/schwinger/scripts/source_quench.jl \
  -L 40 -r 8 -D 12 -d 120 -b 3.5449077018 -u 0.7071067812 \
  -k 1.0 -m 0.3162277660 -t 3.1415926536 -T 1.0 -s 0.05
```

The program diagonalizes the onsite Hamiltonian in `d` oscillator states,
then keeps its `r` lowest eigenstates as the local basis for the chain.
The calculation in arXiv:2307.02522 used an oscillator
cutoff near 2000 and kept 12 onsite states. Increase `d` at fixed `r` and
compare the retained energies and matrix elements; then repeat while
increasing `r`.

The program and the paper use different names for the mass and cosine
couplings. The arguments below reproduce the listed Hamiltonian parameters.

| Hamiltonian parameter | Command argument |
| --- | --- |
| $\mu^2=0.1$ | `--m sqrt(0.1)` |
| $\lambda=0.5$ | `--mu sqrt(0.5)` |
| $\kappa$ in $\frac{\kappa}{2}(\phi_n-\phi_{n-1})^2$ | `--kappa 1` |
| $\beta=\sqrt{4\pi}$ | `--beta 3.5449077018` |
| $\theta$ | `--theta` |

The square roots in the table specify the values to pass to Julia; the
command above uses their decimal values.

The program saves $\langle\phi_n(t)\rangle$ as `field`. The `flux` key
contains the same array. For the Schwinger coupling in the table, the
electric-field convention in arXiv:2307.02522 is
$E_T/e=\phi/\sqrt{\pi}$.

## Longer calculations

The command below evolves an Ising window with `L=320`, packet width `30`,
evolution bond dimension `32`, and `dt=0.05`. It collects the arrays,
figures, and terminal output in one directory.

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
