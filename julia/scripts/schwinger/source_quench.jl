using MPSKit
using TensorKit
using Plots
using LaTeXStrings
using LinearAlgebra
using Plots.PlotMeasures
using JLD2
using ArgParse

"""
Exploratory finite-chain source-quench simulation for a truncated bosonic lattice.

This script is a useful numerical baseline, but it is not the asymptotic wave-packet
scattering calculation of arXiv:2307.02522. In particular, it uses finite-chain DMRG
and a local five-site source quench rather than uniform-MPS quasiparticles, glued
wave packets, and late-time particle projections.

Run with the defaults using

    julia --project=julia julia/scripts/schwinger/source_quench.jl

or inspect all parameters with `--help`.
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
        default = normpath(joinpath(@__DIR__, "..", "..", "results", "schwinger"))
    end

    return parse_args(settings)
end

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

"""Return phi, phi^2, and pi^2 in a `d`-state oscillator basis."""
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

    # phi^2 and pi^2 have the same diagonal and opposite n <-> n+2 entries.
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
Diagonalize the onsite Hamiltonian once and project the operators needed below.

The returned residual checks that the retained vectors diagonalize the projected
onsite Hamiltonian. It does not by itself estimate oscillator-cutoff convergence.
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

"""Construct a finite-chain Hamiltonian from an already projected onsite basis."""
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

"""Return a grid from zero through exactly `total_time` with steps no larger than `dt`."""
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
