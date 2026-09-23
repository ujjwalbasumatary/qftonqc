# Model programs

The five programs below contain the Hamiltonians, state preparation, and
measurements for the Julia calculations. Each has a `parse_cmdline()` function
that reads `ARGS`; `--help` prints its arguments. [Run commands](../running.md)
use each program in a separate Julia process. Several files define the same
function names, so their definitions are kept separate.

The source docstrings give argument conventions, return values, and tensor
dimensions. The [shared-function reference](functions.md) covers the helpers
imported from `QFTSimulations`.

## Ising spectrum

[`spectrum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/spectrum.jl)
has two functions:

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the fields, vacuum bond dimension, momentum interval, VUMPS tolerance, and desired centre-of-mass energy ratio. |
| `main(args)` | Constructs the Ising MPO, finds its uniform vacuum, and evaluates one tangent-space excitation branch on the momentum grid. Prints ``E(k)``, ``E(k)/E(0)``, and the momentum nearest the requested ratio. |

For equal and opposite incoming momenta, `main` uses
``E_{\rm cm}/m_1=2E(k)/E(0)``. At ``g_z=0`` it compares the calculated branch
with `ift_free_fermion_dispersion` from the shared module,

```math
E_{\rm exact}(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
```

The program prints a table and does not save files. Ratios involving ``E(0)``
are undefined when the mass vanishes at the critical point.

## Ising packets at two fixed momenta

[`collide_fixed_momentum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_fixed_momentum.jl)
evaluates excitation tensors at ``+\kappa`` and ``-\kappa``.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the fields, dimensions, packet centres, position width, and saved time-sample count. |
| `get_ops()` | Returns the three Pauli matrices as `TensorMap`s. |
| `get_ham(h_x, h_z)` | Builds ``H=-\sum_n(\sigma_n^z\sigma_{n+1}^z+h_x\sigma_n^x+h_z\sigma_n^z)`` as an infinite MPO. |
| `prep_gs(D, ham)` | Finds a one-site uniform vacuum with VUMPS. |
| `get_QPstate(ψ_gs, ham, momenta)` | Returns the energies and tangent-space states from MPSKit's quasiparticle calculation. |
| `get_B_tensor_list(states)` | Extracts dense excitation tensors and makes their first component real by a separate phase choice for each tensor. |
| `create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)` | Multiplies the two excitation tensors by position-space Gaussians and joins their supports into an `L`-site window. |
| `main(parsed_args)` | Normalizes the window, evolves it with two-site TDVP, and saves vacuum-subtracted bond energy and ``\sigma^z``. |

The packet function uses

```math
B_L(n)=B(+\kappa)e^{+i\kappa(n-n_L)-(n-n_L)^2/\sigma^2},\qquad
B_R(n)=B(-\kappa)e^{-i\kappa(n-n_R)-(n-n_R)^2/\sigma^2}.
```

`two_particle_packet_tensors` in the shared module closes one excitation
insertion within each half-window and joins the halves using the inverse
vacuum bond matrix. [Vacua and wave packets](../physics/wave-packets.md)
describes this tensor construction.

## Ising packets on a momentum grid

[`collide_momentum_grid.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_momentum_grid.jl)
calculates excitation tensors throughout a commensurate Brillouin-zone grid.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the requested momentum spacing, packet momentum and width, fields, coupling, and time arguments. |
| `get_ops()` | Returns the Pauli matrices. |
| `get_ham(J, h_x, h_z)` | Builds the Ising MPO with nearest-neighbour coupling `J`. |
| `prep_gs(D, ham)` | Finds the uniform vacuum with VUMPS. |
| `get_QPstate(ψ_gs, ham, momenta)` | Calculates the default tangent-space branch at every supplied momentum. |
| `get_B_tensor_list(states)` | Extracts the dense tensors and chooses each tensor's phase independently. |
| `nearest_momentum_index(momentum, Δp, n_momenta)` | Wraps momentum into ``[-\pi,\pi)`` and selects the nearest grid point. |
| `create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)` | Fourier-sums the excitation tensors with periodic Gaussian momentum weights. |
| `create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)` | Joins two `L`-site supports into a `2L`-site window. |
| `main()` | Chooses the commensurate grid, prepares and evolves the two-packet state, and saves the observables. |

The site tensor is

```math
B_n=\sum_j e^{-\delta p_j^2/\sigma^2}e^{ip_j(n-n_0)}B(p_j),
\qquad p_j=-\pi+(j-1)\Delta p.
```

Here ``\delta p_j`` is the shortest periodic displacement from the packet's
central grid point. The phases chosen by `get_B_tensor_list` enter this sum;
the function does not enforce continuity between neighbouring momenta.

## Bosonized Schwinger source quench

[`source_quench.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/schwinger/scripts/source_quench.jl)
prepares a finite-chain ground state with source `J0` and evolves it with
source `J1`.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the oscillator and retained-basis dimensions, couplings, source strengths, chain length, and final time. |
| `validate_parameters(args)` | Checks dimensions, signs, and finite values before constructing operators. |
| `matrix_elems(d)` | Returns dense matrices for ``\phi``, ``\phi^2``, and ``\pi^2`` in `d` oscillator states. |
| `get_elems(d_trunc; d, beta, mu, m, theta)` | Diagonalizes the onsite Hamiltonian and projects the operators into its lowest `d_trunc` eigenstates. Returns projected operators, retained energies, and the projection residual. |
| `build_hamiltonian(L, d_trunc, phi, phi_sq, onsite; source_strength, kappa)` | Adds open-chain gradient couplings and a source on up to five central sites. |
| `simulation_times(total_time, dt)` | Produces times ending exactly at `total_time`, with a shorter final interval if needed. |
| `main()` | Finds the preparation ground state with DMRG, evolves with TDVP, and saves the field, times, parameters, and residuals. Returns `(field, times, plot_path, data_path)` as a named tuple. |

The local Hamiltonian diagonalized by `get_elems` is

```math
h=\frac{\pi^2+m^2\phi^2}{2}
  +\mu^2[1-\cos(\beta\phi-\theta)].
```

`build_hamiltonian` adds
``\frac{\kappa}{2}\sum_{n=1}^{L-1}(\phi_n-\phi_{n+1})^2
+J\sum_{n\in S}\phi_n``, where
`S = max(1,L÷2-2):min(L,L÷2+2)`. The [Schwinger page](../physics/schwinger.md)
relates these parameter names to the reference Hamiltonian.

## Scalar-field packets

[`collide_wavepackets.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/phi4/scripts/collide_wavepackets.jl)
uses a truncated oscillator basis and two packets assembled on a momentum grid.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the bare scalar couplings, oscillator dimension, packet grid, bond dimensions, and time arguments. |
| `matrix_elems(d)` | Returns ``\phi``, ``P_d\phi^2P_d``, ``P_d\pi^2P_d``, and ``P_d\phi^4P_d`` as `TensorMap`s. Powers are projected from the oscillator basis rather than formed from the truncated ``\phi`` matrix. |
| `get_ham(d, μ0_sq, λ0)` | Builds the infinite scalar MPO with quadratic, quartic, and nearest-neighbour gradient terms. |
| `prep_gs(d, D, ham)` | Finds the uniform vacuum with VUMPS at tolerance `1e-12`. |
| `get_QPstate(ψ_gs, ham, momenta)` | Returns tangent-space energies and states on the supplied grid. |
| `get_B_tensor_list(states)` | Extracts dense tensors with independent phase choices at each momentum. |
| `nearest_momentum_index(momentum, Δp, n_momenta)` | Selects the nearest periodic momentum-grid point. |
| `create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)` | Uses the Fourier sum written above for the Ising momentum-grid program. |
| `create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)` | Joins the two packet supports using the shared two-particle tensor construction. |
| `main()` | Evolves the normalized window with two-site TDVP and saves vacuum-subtracted energy and ``\phi^2``. |

The [scalar-field page](../physics/phi4.md) gives the Hamiltonian and basis
conventions. Array dimensions, time indexing, and local energy assignments
for all the programs are in [Data and figures](../data.md).
