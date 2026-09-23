using MPSKit
using TensorKit
using Plots
using LaTeXStrings
using LinearAlgebra
using Plots.PlotMeasures
using JLD2
using ArgParse

"""
Directory used for Schwinger output when `--output_dir` is not given.

The path resolves to `results/schwinger/` at the repository root. `main` creates
`plots/` and `data/` below it when those directories do not already exist.
"""
const DEFAULT_OUTPUT_DIRECTORY = normpath(
    joinpath(@__DIR__, "..", "..", "..", "..", "results", "schwinger")
)

"""
    parse_cmdline()

Read the command-line arguments for the finite-chain source quench.

The code evolves the open-chain Hamiltonian

    H(J) = sum_i [(pi_i^2 + m^2 phi_i^2)/2
                  + mu^2 (1 - cos(beta phi_i - theta))]
           + (kappa/2) sum_i (phi_i - phi_(i+1))^2
           + J sum_(i in S) phi_i,

where `S = max(1,L÷2-2):min(L,L÷2+2)`. The ground state is found with
`J = J0` and the real-time evolution uses `J = J1`.

The names `m` and `mu` belong to this program. At `chi = 1`, comparison with
the Hamiltonian in arXiv:2307.02522 requires

  * code `m^2` = paper `mu^2`, the coefficient of `phi^2`;
  * code `mu^2` = paper `lambda`, the cosine strength;
  * code `kappa = 1`; and
  * code `beta = sqrt(4pi)`.

There is no overall `chi` parameter in this Hamiltonian. `--d` is the number
of oscillator states used to diagonalize the onsite term, whereas `--d_trunc`
is the number of its lowest eigenstates kept on each lattice site. `--D` is the
finite-MPS bond dimension. `--total_time` is the requested final time and
`--time_step` is the largest interval passed to TDVP.

Returns the dictionary produced by `ArgParse.parse_args`.
"""
function parse_cmdline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--lattice", "-L"
        help = "Number of lattice sites"
        arg_type = Int
        default = 100

        "--J0", "-j"
        help = "Five-site source strength used to prepare the ground state"
        arg_type = Float64
        default = 1.0

        "--J1", "-J"
        help = "Five-site source strength after the quench"
        arg_type = Float64
        default = 0.2

        "--d_trunc", "-r"
        help = "Number of low-energy onsite eigenstates retained"
        arg_type = Int
        default = 12

        "--D", "-D"
        help = "MPS bond dimension"
        arg_type = Int
        default = 20

        "--d", "-d"
        help = "Harmonic-oscillator cutoff before onsite diagonalization"
        arg_type = Int
        default = 2000

        "--beta", "-b"
        help = "beta in cos(beta*phi - theta); the paper uses sqrt(4*pi)"
        arg_type = Float64
        default = 1.0

        "--mu", "-u"
        help = "mu in the onsite coefficient mu^2*(1-cos(beta*phi-theta))"
        arg_type = Float64
        default = 0.5

        "--kappa", "-k"
        help = "Nearest-neighbour gradient coefficient kappa"
        arg_type = Float64
        default = 0.1

        "--m", "-m"
        help = "Bare onsite mass m"
        arg_type = Float64
        default = 1.0

        "--theta", "-t"
        help = "theta in cos(beta*phi - theta)"
        arg_type = Float64
        default = pi

        "--total_time", "-T"
        help = "Final lattice time; a shorter final step is used when needed"
        arg_type = Float64
        default = 100.0

        "--time_step", "-s"
        help = "Maximum real-time evolution step"
        arg_type = Float64
        default = 0.05

        "--progress_every", "-p"
        help = "Print progress every this many steps (0 disables progress output)"
        arg_type = Int
        default = 10

        "--output_dir", "-o"
        help = "Root directory in which plots/ and data/ are created"
        arg_type = String
        default = DEFAULT_OUTPUT_DIRECTORY
    end

    return parse_args(settings)
end

"""
    validate_parameters(args)

Check the command-line values before any tensor-network calculation begins.

The lattice must contain at least two sites, `1 <= d_trunc <= d`, the MPS bond
dimension must be positive, and the mass, cosine strength, and gradient
coefficient must be nonnegative. Times and couplings must be finite;
`time_step` must be positive. The source strengths may have either sign.

Returns `nothing`. Invalid values raise `ArgumentError`.
"""
function validate_parameters(args)
    args["lattice"] >= 2 || throw(ArgumentError("--lattice must be at least 2"))
    args["d_trunc"] >= 1 || throw(ArgumentError("--d_trunc must be positive"))
    args["d"] >= args["d_trunc"] ||
        throw(ArgumentError("--d must be at least --d_trunc"))
    args["D"] >= 1 || throw(ArgumentError("--D must be positive"))
    args["m"] >= 0 || throw(ArgumentError("--m must be nonnegative"))
    args["mu"] >= 0 || throw(ArgumentError("--mu must be nonnegative"))
    args["kappa"] >= 0 || throw(ArgumentError("--kappa must be nonnegative"))
    args["total_time"] >= 0 || throw(ArgumentError("--total_time must be nonnegative"))
    args["time_step"] > 0 || throw(ArgumentError("--time_step must be positive"))
    args["progress_every"] >= 0 ||
        throw(ArgumentError("--progress_every must be nonnegative"))

    finite_keys = ("J0", "J1", "beta", "mu", "kappa", "m", "theta",
        "total_time", "time_step")
    all(key -> isfinite(args[key]), finite_keys) ||
        throw(ArgumentError("all real-valued simulation parameters must be finite"))
    return nothing
end

"""
    matrix_elems(d)

Construct `phi`, `phi^2`, and `pi^2` in the first `d` harmonic-oscillator
states `|n>`, with `n = 0,...,d-1` and

    phi = (a + a†)/sqrt(2),    pi = (a - a†)/(i sqrt(2)).

Each returned object is a dense `d × d` `ComplexF64` matrix. The matrix
elements of the quadratic operators are filled directly, so they represent
`P_d phi^2 P_d` and `P_d pi^2 P_d`; they are not obtained by squaring the
truncated `phi` matrix.

`d` must be positive. The return value is `(phi, phi_sq, pi_sq)`.
"""
function matrix_elems(d::Int)
    d >= 1 || throw(ArgumentError("oscillator cutoff d must be positive"))

    phi = zeros(ComplexF64, d, d)
    phi_sq = zeros(ComplexF64, d, d)
    pi_sq = zeros(ComplexF64, d, d)

    # phi = (a + a^dagger)/sqrt(2)
    @inbounds for i in 2:d
        value = sqrt((i - 1) / 2)
        phi[i, i - 1] = value
        phi[i - 1, i] = value
    end

    # Fill P_d phi^2 P_d for sign=+1 or P_d pi^2 P_d for sign=-1.
    function fill_quadratic!(matrix, sign)
        @inbounds for i in 1:d
            if i + 2 <= d
                value = sign * sqrt(i * (i + 1)) / 2
                matrix[i, i + 2] = value
                matrix[i + 2, i] = value
            end
            matrix[i, i] = (2 * i - 1) / 2
        end
        return matrix
    end

    fill_quadratic!(phi_sq, +1)
    fill_quadratic!(pi_sq, -1)
    return phi, phi_sq, pi_sq
end

"""
    get_elems(d_trunc; d=2000, beta=1.0, mu=0.5, m=1.0, theta=0.0)

Diagonalize the single-site Hamiltonian

    h = (pi^2 + m^2 phi^2)/2 + mu^2 [1 - cos(beta phi - theta)]

in `d` oscillator states and retain its `d_trunc` lowest eigenvectors. The
columns of `W` are the retained eigenvectors, and the return value is a named
tuple with

  * `phi`: `W† phi W` as a map on `ℂ^d_trunc`;
  * `phi_sq`: `W† phi^2 W`, projected before truncation and therefore generally
    different from `phi * phi` in the retained space;
  * `onsite`: the diagonal retained onsite Hamiltonian;
  * `energies`: all `d_trunc` retained eigenvalues; and
  * `residual`: the infinity norm of
    `W† h W - Diagonal(energies)`.

The residual checks the eigendecomposition inside the chosen `d`-state
oscillator space. It does not measure the change in the retained eigenvalues
or operators when `d` or `d_trunc` is increased.
"""
function get_elems(d_trunc::Int; d::Int=2000, beta::Float64=1.0,
        mu::Float64=0.5, m::Float64=1.0, theta::Float64=0.0)
    1 <= d_trunc <= d || throw(ArgumentError("require 1 <= d_trunc <= d"))

    phi, phi_sq, pi_sq = matrix_elems(d)
    identity_matrix = Matrix{ComplexF64}(I, d, d)
    onsite_matrix = (m^2 * phi_sq + pi_sq) / 2 +
        mu^2 * (identity_matrix - cos(beta * phi - theta * identity_matrix))

    eigensystem = eigen(Hermitian(onsite_matrix))
    retained_vectors = eigensystem.vectors[:, 1:d_trunc]
    retained_energies = eigensystem.values[1:d_trunc]

    projected_hamiltonian = retained_vectors' * onsite_matrix * retained_vectors
    residual = opnorm(projected_hamiltonian - Diagonal(retained_energies), Inf)
    isfinite(residual) || error("onsite eigensolve produced a non-finite residual")

    phi_projected = retained_vectors' * phi * retained_vectors
    phi_sq_projected = retained_vectors' * phi_sq * retained_vectors
    local_space = ℂ^d_trunc

    phi_tensor = TensorMap(phi_projected, local_space ← local_space)
    phi_sq_tensor = TensorMap(phi_sq_projected, local_space ← local_space)
    onsite_tensor = TensorMap(Matrix(Diagonal(retained_energies)),
        local_space ← local_space)

    return (phi=phi_tensor, phi_sq=phi_sq_tensor, onsite=onsite_tensor,
        energies=retained_energies, residual=residual)
end

"""
    build_hamiltonian(L, d_trunc, phi, phi_sq, onsite;
                      source_strength=1.0, kappa=0.1)

Construct the open-chain MPO

    H = sum_i onsite_i
        + (kappa/2) sum_(i=1)^(L-1) (phi_i - phi_(i+1))^2
        + source_strength sum_(i in S) phi_i

on `L` copies of `ℂ^d_trunc`. `phi`, `phi_sq`, and `onsite` must already be
expressed in that retained onsite basis.

The source region is
`S = max(1,L÷2-2):min(L,L÷2+2)`. It contains five sites when the chain is
long enough, is clipped at an end for short chains, and is one site to the left
of the geometric centre when `L` is odd because Julia integer division is used.
No overall factor `chi` multiplies the MPO.

Returns a `FiniteMPOHamiltonian`. `L < 2` raises `ArgumentError`.
"""
function build_hamiltonian(L::Int, d_trunc::Int, phi, phi_sq, onsite;
        source_strength::Float64=1.0, kappa::Float64=0.1)
    L >= 2 || throw(ArgumentError("L must be at least 2"))

    onsite_terms = [i => onsite for i in 1:L]
    gradient_left = [i => kappa * phi_sq / 2 for i in 1:(L - 1)]
    gradient_right = [i => kappa * phi_sq / 2 for i in 2:L]
    gradient_bonds = [(i, i + 1) => -kappa * phi ⊗ phi for i in 1:(L - 1)]

    center = L ÷ 2
    source_sites = max(1, center - 2):min(L, center + 2)
    source_terms = [i => source_strength * phi for i in source_sites]

    chain = fill(ℂ^d_trunc, L)
    return FiniteMPOHamiltonian(chain, gradient_left..., gradient_right...,
        onsite_terms..., gradient_bonds..., source_terms...)
end

"""
    simulation_times(total_time, dt)

Return the saved times from zero through exactly `total_time`. Intervals have
length `dt`, except for a shorter final interval when `total_time/dt` is not an
integer. The result is `[0.0]` when `total_time == 0`.

Both arguments are `Float64`; `total_time` must be nonnegative and `dt` must be
positive.
"""
function simulation_times(total_time::Float64, dt::Float64)
    total_time >= 0 || throw(ArgumentError("total_time must be nonnegative"))
    dt > 0 || throw(ArgumentError("dt must be positive"))

    times = collect(0.0:dt:total_time)
    tolerance = 16 * eps(max(total_time, dt, 1.0))
    if isapprox(times[end], total_time; atol=tolerance, rtol=1.0e-12)
        times[end] = total_time
    elseif times[end] < total_time
        push!(times, total_time)
    end
    return times
end

"""
    main()

Prepare the DMRG ground state of `H(J0)`, change the central source to `J1`,
and evolve the finite MPS with TDVP. The same TDVP environments are passed from
one time interval to the next. BLAS is restricted to one thread before the
onsite diagonalization and MPS calculations.

At every saved time the program measures the raw field expectation value
`field[t_index, site] = real(<phi_site>)`. Thus `field` has shape
`(length(times), L)`. The PNG heatmap shows at most 21 sites, from ten sites
left of `L÷2` through ten sites right of it; the JLD2 file contains all `L`
sites.

The output directory receives

  * `plots/<parameter-dependent name>.png`; and
  * `data/<parameter-dependent name>.jld2`.

The JLD2 file stores `field`, `times`, the parsed parameters, the retained
onsite energies, the onsite eigensolver residual, and the DMRG residual. It
also stores `flux` as another name for the unrescaled `field` array. In the
shifted-field convention of arXiv:2307.02522, `E_T/e = beta*phi/(2pi)`.
At the Schwinger coupling `beta = sqrt(4pi)`, this becomes
`E_T/e = phi/sqrt(pi)`. The saved `flux` values have not been rescaled.

This calculation changes a source on a finite open chain. It does not build
incoming quark or meson wave packets and does not project the late-time state
onto outgoing particle states.

Returns `(field, times, plot_path, data_path)` as a named tuple. It also prints
progress, creates the output directories, and writes the PNG and JLD2 files.
"""
function main()
    args = parse_cmdline()
    validate_parameters(args)
    BLAS.set_num_threads(1)

    L = args["lattice"]
    J0 = args["J0"]
    J1 = args["J1"]
    d_trunc = args["d_trunc"]
    bond_dimension = args["D"]
    oscillator_cutoff = args["d"]
    beta = args["beta"]
    mu = args["mu"]
    kappa = args["kappa"]
    mass = args["m"]
    theta = args["theta"]
    total_time = args["total_time"]
    dt = args["time_step"]
    progress_every = args["progress_every"]

    @info "Building the shared truncated onsite basis" oscillator_cutoff d_trunc
    basis = get_elems(d_trunc; d=oscillator_cutoff, beta=beta, mu=mu,
        m=mass, theta=theta)
    @info "Onsite basis ready" projection_residual=basis.residual

    preparation_hamiltonian = build_hamiltonian(
        L, d_trunc, basis.phi, basis.phi_sq, basis.onsite;
        source_strength=J0, kappa=kappa)
    quench_hamiltonian = build_hamiltonian(
        L, d_trunc, basis.phi, basis.phi_sq, basis.onsite;
        source_strength=J1, kappa=kappa)

    state = FiniteMPS(L, ℂ^d_trunc, ℂ^bond_dimension)
    state, _, dmrg_residual = find_groundstate(state, preparation_hamiltonian, DMRG())
    isfinite(dmrg_residual) || error("DMRG produced a non-finite residual")
    @info "Ground-state preparation finished" dmrg_residual

    times = simulation_times(total_time, dt)
    field = zeros(Float64, length(times), L)
    for site in 1:L
        field[1, site] = real(expectation_value(state, site => basis.phi))
    end

    # Reuse the quench environments across TDVP steps instead of rebuilding them.
    tdvp = TDVP()
    quench_environments = environments(state, quench_hamiltonian)
    for step in 2:length(times)
        start_time = times[step - 1]
        step_size = times[step] - start_time
        state, quench_environments = timestep(state, quench_hamiltonian,
            start_time, step_size, tdvp, quench_environments)

        for site in 1:L
            field[step, site] = real(expectation_value(state, site => basis.phi))
        end
        if progress_every > 0 && (step == 2 || step == length(times) ||
                (step - 1) % progress_every == 0)
            println("Evolution step $(step - 1)/$(length(times) - 1), t=$(times[step])")
        end
    end

    center = L ÷ 2
    displayed_sites = max(1, center - 10):min(L, center + 10)
    plot = heatmap(displayed_sites, times, field[:, displayed_sites];
        dpi=600, ylabel=L"Lattice time $t$", xlabel=L"Lattice site $n$",
        colorbar_title=L"\langle\phi_n\rangle", right_margin=15mm)

    filename = "source_quench_L_$(L)_d_$(oscillator_cutoff)_dtrunc_$(d_trunc)" *
        "_D_$(bond_dimension)_tmax_$(total_time)_dt_$(dt)_J0_$(J0)_J1_$(J1)" *
        "_beta_$(beta)_kappa_$(kappa)_m_$(mass)_theta_$(theta)_mu_$(mu)"
    plot_directory = joinpath(args["output_dir"], "plots")
    data_directory = joinpath(args["output_dir"], "data")
    mkpath(plot_directory)
    mkpath(data_directory)
    plot_path = joinpath(plot_directory, filename * ".png")
    data_path = joinpath(data_directory, filename * ".jld2")

    savefig(plot, plot_path)
    jldsave(data_path; field, flux=field, times, parameters=args,
        onsite_energies=basis.energies, onsite_projection_residual=basis.residual,
        dmrg_residual)
    @info "Saved exploratory source-quench outputs" plot_path data_path

    return (field=field, times=times, plot_path=plot_path, data_path=data_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
