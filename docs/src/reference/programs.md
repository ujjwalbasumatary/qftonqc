# Model programs

The Julia programs below construct the model Hamiltonians, prepare states,
and calculate spectra or time evolution. Their command-line entry points have
a `parse_cmdline()` function that reads `ARGS`; `--help` prints the available
arguments. The overlap modules provide functions for analysing saved states.
Several files define functions with the same names,
so run each program in a separate Julia process using the
[commands given here](../running.md).

The docstrings in the source files describe the arguments, return values,
and tensor dimensions. Functions imported from `QFTSimulations` are
documented in the [shared-function reference](functions.md).

## Ising spectrum

[`spectrum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/spectrum.jl)
contains the functions `parse_cmdline()` and `main(args)`.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the fields, vacuum bond dimension, momentum interval, VUMPS tolerance, and desired centre-of-mass energy ratio. |
| `main(args)` | Constructs the Ising MPO, finds its uniform vacuum, and evaluates one tangent-space excitation branch on the momentum grid. Prints ``E(k)``, ``E(k)/E(0)``, and the momentum nearest the requested ratio. |

For equal and opposite incoming momenta, `main` uses
``E_{\rm cm}/m_1=2E(k)/E(0)``. At ``g_z=0`` and ``g_x>1``, it compares the
calculated branch with `ift_free_fermion_dispersion` from the shared module,

```math
E_{\rm exact}(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
```

The program prints a table and does not save files. It rejects ``g_z=0``
with ``g_x<1`` because a single kink needs different ordered vacua on its
left and right, whereas the program uses the same vacuum on both sides.
It also rejects ``(g_x,g_z)=(1,0)``, where the mass vanishes and the ratios
involving ``E(0)`` are undefined.

## Ising bound-state spectrum

[`bound_state_spectrum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/bound_state_spectrum.jl)
has the implementation of the vacuum bond-dimension comparison. It saves
the rest energies before scanning nonzero momenta, so the rest-mass table is
available while the rest of the calculation runs.

| Function | Calculation |
| --- | --- |
| `ising_hamiltonian(gx, gz)` | Constructs the uniform Ising MPO in units ``J=a=\hbar=1``. |
| `excitation_solutions(vacuum, hamiltonian, momentum, environments; num=3)` | Calculates the low excitation eigenvalues, both gauges of their tensors, and the residual ``\|H_{\rm eff}B-EB\|/\|B\|``. |
| `sampled_threshold(total_momentum, samples, lowest_energies)` | Minimizes the sum of two light-particle energies over the supplied momentum pairs. |
| `main(args)` | Finds independent vacua at the requested dimensions and saves the spectra, residuals, settings, and threshold comparisons. |

The sampled threshold is an upper bound on the minimum over the entire
Brillouin zone. The excitation residual measures the eigensolver's error
within its chosen variational space; comparing energies at different bond
dimensions measures a separate source of error.

## Ising packets at two fixed momenta

[`collide_fixed_momentum.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/collide_fixed_momentum.jl)
evaluates excitation tensors at ``+\kappa`` and ``-\kappa``.

| Function | Calculation |
| --- | --- |
| `parse_cmdline()` | Reads the fields, dimensions, packet centres, width in position space, number of saved times, and MPS saving interval. |
| `get_ops()` | Returns the three Pauli matrices as `TensorMap`s. |
| `get_ham(h_x, h_z)` | Builds ``H=-\sum_n(\sigma_n^z\sigma_{n+1}^z+h_x\sigma_n^x+h_z\sigma_n^z)`` as an infinite MPO. |
| `prep_gs(D, ham)` | Finds a one-site uniform vacuum with VUMPS. |
| `get_QPstate(ψ_gs, ham, momenta)` | Returns the energies and tangent-space states from MPSKit's quasiparticle calculation. |
| `get_B_tensor_list(states)` | Extracts dense excitation tensors and makes their first component real by a separate phase choice for each tensor. |
| `create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)` | Weights the two excitation tensors with position-space Gaussians and places the resulting packets in an `L`-site window. |
| `main(parsed_args)` | Normalizes the two-packet state, evolves it with two-site TDVP, saves MPS states, and saves the vacuum-subtracted bond energy and ``\sigma^z`` expectation values. |

[`state_io.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/state_io.jl)
has the implementation of `save_ift_state`, `load_ift_state`, and
`should_save_state`. The initial and final states are always saved;
`--save_every` sets the interval between intermediate states. Each file keeps
the vacuum, Hamiltonian, and incoming excitation tensors alongside the MPS.
The file contents and loading example are in [Data and figures](../data.md).

The packet function uses

```math
B_L(n)=B(+\kappa)e^{+i\kappa(n-n_L)-(n-n_L)^2/\sigma^2},\qquad
B_R(n)=B(-\kappa)e^{-i\kappa(n-n_R)-(n-n_R)^2/\sigma^2}.
```

The shared function `two_particle_packet_tensors` constructs each
half-window with exactly one excitation insertion. It joins the two halves
using the inverse vacuum bond matrix. This tensor construction is described
in [Vacua and wave packets](../physics/wave-packets.md).

## Ising excitation tensors and two-particle overlaps

These two files contain functions to include in an analysis of saved MPS
states. They do not start a new time evolution.

[`particle_basis.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/particle_basis.jl)
defines the module `IFTParticleBasis`.

| Function | Calculation |
| --- | --- |
| `vacuum_arrays(vac)` | Returns dense left- and right-canonical vacuum tensors, the centre matrix, and its inverse. |
| `right_gauge_tensor(vac, B, k)` | Converts a left-gauged excitation to the right gauge at momentum `k`, preserving the momentum eigenstate and its phase. |
| `excitation_tensors(vac, ham, momenta; num=1)` | Calculates tangent-space energies and excitation tensors in both gauges on the supplied vacuum. |
| `gauge_residuals(vac, BL, BR)` | Measures the canonical and excitation gauge residuals and the excitation tensor norms. |
| `orthonormal_tensor_basis(tensors; rtol=1e-10)` | Uses an SVD to obtain an orthonormal reference basis and complex expansion coefficients. |

[`two_particle_overlap.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/two_particle_overlap.jl)
defines the module `IFTTwoParticleOverlap` and uses arrays with indices
`(left bond, physical spin, right bond)`.

| Function | Calculation |
| --- | --- |
| `two_particle_overlaps(tensors, AL, AR, Cinv, BL, BR; minimum_separation=1)` | Contracts the finite MPS with localized excitation pairs at every allowed ordered pair of positions. |
| `localized_pair_norms(AL, Cinv, BL, BR, max_separation)` | Calculates the squared norm of each reference pair as a function of separation. |
| `pair_basis_grams(AL, Cinv, left_basis, right_basis, max_separation)` | Calculates the Gram matrices between reference pairs at the same positions. |
| `two_particle_weight(overlaps, norms, state_norm2; minimum_separation=1)` | Sums the normalized squared overlaps for an orthogonal localized-pair basis. |

Supply the evolved state as `AC[1], AR[2], ..., AR[L]`, with the same vacuum
gauges at its boundaries as in the reference tensors. Use a left-gauged
excitation for `BL` and a right-gauged excitation for `BR`. The
[particle-production page](../physics/particle-production.md) explains the
normalization and the reconstruction of momentum-dependent particle states.

## Ising three-particle overlaps

[`three_particle_overlap.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/models/ift/scripts/three_particle_overlap.jl)
defines `IFTThreeParticleOverlap`. Its arrays have the same index order as
the two-particle module. Use left-gauged tensors for the first and middle
insertions and a right-gauged tensor for the last.

| Function | Calculation |
| --- | --- |
| `three_particle_overlaps(tensors, AL, AR, Cinv, BL, BM, BR; minimum_separation=1)` | Contracts every allowed ordered triple and groups the amplitudes by the outer sites. |
| `middle_position_gram(AL, Cinv, BL, BM, BR, span; middle_offsets)` | Calculates overlaps between different middle positions at fixed outer separation. |
| `localized_triple_norms(AL, Cinv, BL, BM, BR, span; middle_offsets)` | Returns the diagonal entries of that Gram matrix. |
| `pair_triple_cross_gram(AL, Cinv, BLpair, BRpair, BLtriple, BMtriple, BRtriple, span; middle_offsets)` | Calculates ``\langle\mathrm{triple}|\mathrm{pair}\rangle`` for a specified reference pair. |
| `three_particle_weight(blocks, grams, state_norm2; minimum_separation=1, gram_minimum_separation=1)` | Combines the amplitudes using the full middle-position Gram matrix and reports its eigenvalue diagnostics. |

The [320-site collision](../demonstrations/ising-collision.md) includes an
example using these functions. Their source docstrings specify the stored
position indices, eigenvalue cutoff, and boundary gauge assumptions.

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
| `create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)` | Combines two packets, each supported on `L` sites, in a `2L`-site window. |
| `main()` | Chooses the commensurate grid, prepares and evolves the two-packet state, and saves the observables. |

The site tensor is

```math
B_n=\sum_j e^{-\delta p_j^2/\sigma^2}e^{ip_j(n-n_0)}B(p_j),
\qquad p_j=-\pi+(j-1)\Delta p.
```

Here ``\delta p_j`` is the shortest periodic displacement from the packet's
central grid point. The phases chosen by `get_B_tensor_list` enter this sum.
Each phase is chosen independently, so continuity between neighbouring
momenta is not imposed.

With ``N`` momenta and ``\Delta p=2\pi/N``, the Fourier sum obeys
``B_{n+N}=(-1)^N B_n``. It repeats for even ``N`` and changes sign for odd
``N``, while its norm repeats in both cases. Each packet is cut to an
``N``-site support inside the infinite chain, so the construction accepts
either parity of ``N``. See [Wave packets](../physics/wave-packets.md).

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
| `main()` | Finds the ground state with source `J0` using DMRG, evolves it with source `J1` using TDVP, and saves the field, times, parameters, and residuals. Returns `(field, times, plot_path, data_path)` as a named tuple. |

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
