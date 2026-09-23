# Data and figures

You can collect quench and collision outputs in `results/`. Each evolution
program creates `data/` and `plots/` beneath its `--output_dir`, writes the
expectation values to JLD2 files, and saves PNG heatmaps. Rows are times and
columns are lattice sites or bonds. The PNGs use a resolution of 600 dpi.

| Program | Default output directory | JLD2 files |
| --- | --- | --- |
| Ising `collide_fixed_momentum.jl` | `results/ift/` | `data/energy_<name>.jld2`, `data/sz_value_<name>.jld2` |
| Ising `collide_momentum_grid.jl` | `results/ift/` | `data/<name>_energy.jld2`, `data/<name>_sz.jld2` |
| Schwinger `source_quench.jl` | `results/schwinger/` | `data/source_quench_<parameters>.jld2` |
| Scalar `collide_wavepackets.jl` | `results/phi4/` | `data/energy_<name>.jld2`, `data/phi_sq_<name>.jld2` |

The Ising spectrum program prints its results to the terminal and writes no
data files. The source code for each program is linked in the
[program reference](reference/programs.md).

## Ising collision arrays

Both collision programs write the energy and spin expectation values to
separate files. Each file also contains the times at which the expectation
values were saved.

| Key | Shape | Quantity |
| --- | --- | --- |
| `energy_exp` | `(T, L)` | Vacuum-subtracted bond-energy expectation value; columns `1:L-1` are filled. |
| `s_z_exp` | `(T, L)` | Vacuum-subtracted ``\sigma_n^z`` expectation value at all `L` sites. |
| `times` | `(T,)` | `0, dt, …, (T-1)dt`; included in both files. |

Here `T` is the integer `--total_time` argument. Row one contains the initial
state. For the fixed-momentum program, `L` is `--length`. For the momentum-grid
program, `L = 2N`, with `N = round(Int, 2π/delta_p)` and `delta_p` the requested
spacing. The actual momentum spacing is ``2\pi/N``.

The bond operator is

```math
h_{n,n+1}=-J\sigma_n^z\sigma_{n+1}^z
 -\frac12(h_x\sigma_n^x+h_z\sigma_n^z)
 -\frac12(h_x\sigma_{n+1}^x+h_z\sigma_{n+1}^z).
```

The fixed-momentum program sets ``J=1``; the momentum-grid program takes
`--J`. With ``|\Omega\rangle`` the uniform MPS returned by `prep_gs`, the arrays
contain

```math
\begin{aligned}
\mathtt{energy\_exp}[r,n]
 &=\operatorname{Re}\langle h_{n,n+1}\rangle_{\psi(t_r)}
   -\operatorname{Re}\langle h_{n,n+1}\rangle_\Omega,\\
\mathtt{s\_z\_exp}[r,n]
 &=\operatorname{Re}\langle\sigma_n^z\rangle_{\psi(t_r)}
   -\operatorname{Re}\langle\sigma_n^z\rangle_\Omega.
\end{aligned}
```

The vacuum expectation values are evaluated with the same operators used in
the evolving window. The final column of `energy_exp` remains zero because
there are only `L-1` internal bonds. Use `energy_exp[:, 1:end-1]` for plotting
or summing those bonds. The energy PNG already omits that column.

## Scalar-field collision arrays

The scalar program uses the same definitions of `T`, `times`, and `L = 2N` as
the Ising momentum-grid program. Its energy file contains `energy_exp` and
`times`; its other file contains `phi_sq_exp` and `times`. Both observable
arrays have shape `(T, L)`.

The measured bond operator includes the onsite terms at the left site,

```math
h_{n,n+1}=\frac{\pi_n^2}{2}+\frac{\mu_0^2\phi_n^2}{2}
 +\frac{\lambda_0\phi_n^4}{24}+\phi_n^2-\phi_n\phi_{n+1}.
```

`energy_exp[r,n]` subtracts the expectation value of this operator in the
uniform vacuum. Its final column is again unused. All columns of `phi_sq_exp`
contain ``\operatorname{Re}\langle\phi_n^2\rangle_{\psi(t_r)}
-\operatorname{Re}\langle\phi_n^2\rangle_\Omega``. These local expectation
values follow the disturbance through the lattice. Outgoing particle
probabilities require the [state projections](physics/particle-production.md)
described separately.

## Schwinger source-quench arrays

The quench file contains the field expectation values at all
`L = parameters["lattice"]` sites, along with the times, parameters, and
quantities returned during state preparation.

| Key | Contents |
| --- | --- |
| `field` | Array of shape `(length(times), L)` containing ``\operatorname{Re}\langle\phi_n(t)\rangle`` without subtracting the vacuum expectation value. |
| `flux` | Contains the same values as `field`, with no change of normalization. |
| `times` | Saved times from zero through the requested final `--total_time`. |
| `parameters` | Dictionary returned by the argument parser, including couplings, cutoffs, source strengths, and time arguments. |
| `onsite_energies` | The `d_trunc` lowest eigenvalues of the onsite Hamiltonian in `d` oscillator states. |
| `onsite_projection_residual` | ``\|W^\dagger hW-\operatorname{diag}(\epsilon_1,\ldots,\epsilon_{d_{\rm trunc}})\|_\infty``, with `W` the retained eigenvectors. |
| `dmrg_residual` | Residual returned by MPSKit when finding the preparation ground state. |

Here `--total_time` is a physical duration. The final interval is shorter than
`--time_step` when necessary to reach it exactly. The onsite residual compares
the projected Hamiltonian with the diagonal matrix of retained eigenvalues
at the chosen cutoff. The dependence on `d` and `d_trunc` can be examined by
repeating the calculation with different numbers of oscillator states and
retained onsite eigenstates.

At ``\beta=\sqrt{4\pi}``, the Schwinger normalization in
[the reference](https://arxiv.org/abs/2307.02522) gives
``E_T/e=\phi/\sqrt{\pi}``. For that comparison, divide `field` by `sqrt(pi)`.
The program also accepts other values of `beta`, so the saved name `flux`
alone does not specify an electric-field normalization. The PNG displays at
most 21 sites around `L÷2`; the JLD2 array retains the full chain.

## Loading files and keeping run parameters

You can read a saved energy file with the following command from the
repository root. Replace the example filename with the path to your file.

```bash
julia --startup-file=no --project=simulations -e '
    using JLD2
    saved = load(ARGS[1])
    @show keys(saved) size(saved["energy_exp"]) saved["times"]
' "results/ift/data/energy_<name>.jld2"
```

This reads the arrays without starting an evolution. In Julia, `load(path,
"energy_exp")` reads that key alone.

The Ising and scalar files contain the observable and `times`, with run
parameters encoded in the filename rather than a saved dictionary. Their
filenames replace decimal points by `p` and minus signs by `m`. Keep the
command, Julia version, and repository commit with the data. The scalar
filenames omit `--momentum`, so use a different `--output_dir` when changing
only that argument to avoid overwriting another run. Git ignores the
contents of `results/`; copy data that you want to retain to storage where
you keep backups.
