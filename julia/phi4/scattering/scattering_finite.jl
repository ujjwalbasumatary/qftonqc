using MPSKit, TensorKit, LaTeXStrings, SpecialFunctions, Plots, ArgParse, JLD2

"""
Note that the above packages are not in the base Julia distribution.
Install these by going to the REPL and typing
using Pkg [enter]
Pkg.add(package_name_as_a_string) [enter]
"""


function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--lat_size", "-L"
        help = "Lattice size"
        arg_type = Int
        default = 300
        "--local_dim", "-d"
        help = "Local Hilbert space dimension"
        arg_type = Int
        default = 16
        "--bond_dimension", "-D"
        help = "Maximum bond dimension"
        arg_type = Int
        default = 24
        "--offset", "-o"
        help = "Location of the initial disturbances from the center L ÷ 2"
        arg_type = Int
        default = 20
        "--lambda", "-l"
        help = "Bare lattice coupling lambda"
        arg_type = Float64
        default = 0.2
        "--mu_sq", "-m"
        help = "Bare lattice coupling squared"
        arg_type = Float64
        default = -0.07
        "--total_time", "-T"
        help = "Maximum time step for the evolution"
        arg_type = Int
        default = 1000
        "--time_step", "-t"
        help = "Step size for time evolution"
        arg_type = Float64
        default = 0.02
    end

    return parse_args(ARGS, s)
end

function matrix_elems(d)
    phi_sq = zeros(ComplexF64, (d, d))
    pi_sq = zeros(ComplexF64, (d, d))
    phi_4 = zeros(ComplexF64, (d, d))
    phi = zeros(ComplexF64, (d, d))
    for i in 2:d
        phi[i, i-1] = sqrt((i - 1) / 2)
        phi[i-1, i] = sqrt((i - 1) / 2)
    end
    for i in 1:d
        if i < d - 1
            phi_sq[i, i+2] = sqrt(i * (i + 1)) / 2
            phi_sq[i+2, i] = sqrt(i * (i + 1)) / 2
        end
        phi_sq[i, i] = (2 * i - 1) / 2
    end
    for i in 1:d
        if i < d - 1
            pi_sq[i, i+2] = -sqrt(i * (i + 1)) / 2
            pi_sq[i+2, i] = -sqrt(i * (i + 1)) / 2
        end
        pi_sq[i, i] = (2 * i - 1) / 2
    end
    for i in 1:d
        n = i - 1  # occupation number

        # diagonal
        phi_4[i, i] = (6 * n^2 + 6 * n + 3) / 4

        # connect |n> <-> |n+2>
        if i + 2 <= d
            val = (4 * n + 6) * sqrt((n + 1) * (n + 2)) / 4
            j = i + 2
            phi_4[i, j] = val
            phi_4[j, i] = val   # Hermitian
        end

        # connect |n> <-> |n+4>
        if i + 4 <= d
            val = sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4
            j = i + 4
            phi_4[i, j] = val
            phi_4[j, i] = val   # Hermitian
        end
    end
    ϕ2 = TensorMap(phi_sq, ℂ^d ← ℂ^d)
    π2 = TensorMap(pi_sq, ℂ^d ← ℂ^d)
    ϕ4 = TensorMap(phi_4, ℂ^d ← ℂ^d)
    ϕ = TensorMap(phi, ℂ^d ← ℂ^d)
    return ϕ, ϕ2, π2, ϕ4
end

function get_ham(d, L, μ0_sq, λ0)
    """
    Prepares the infinte MPO Hamiltonian from μ0, λ0 and
    the local Hilbert space dimension d as inputs
    """
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)
    chain = fill(ℂ^d, L)
    single_site_term = (μ0_sq * ϕ2 + π2) / 2 + λ0 * ϕ4 / 24 + ϕ2
    two_site_term = -ϕ ⊗ ϕ
    single_site_terms = [i => single_site_term for i in 1:L]
    two_site_terms = [(i, i + 1) => two_site_term for i in 1:L-1]
    ham = FiniteMPOHamiltonian(chain, single_site_terms..., two_site_terms...)
    return ham
end

function prep_gs(d, L, D, ham)
    """
    rand: random matrices
    ComplexF64: dataype of the matrices. Here we have complex numbers with 64-bit floats
    ℂ^d: local Hilbert space dimension
    ℂ^D: bond dimension
    Returns the converged ground state of the uMPS
    """
    ψ0 = FiniteMPS(rand, ComplexF64, L, ℂ^d, ℂ^D)
    ψ, _, _ = find_groundstate(ψ0, ham, DMRG(; tol=1e-7, maxiter=5))
    return ψ
end

function main()
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"] # max bond dimension
    L = parsed_args["lat_size"] # number of sites
    d = parsed_args["local_dim"] # local Hilbert space dim
    λ0 = parsed_args["lambda"]
    μ0_sq = parsed_args["mu_sq"]
    offset = parsed_args["offset"]
    T = parsed_args["total_time"]
    L = parsed_args["lat_size"]
    dt = parsed_args["time_step"]

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end

    ham = get_ham(d, L, μ0_sq, λ0)

    ψ = prep_gs(d, L, D, ham)

    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)

    # apply the field operator
    mid = L ÷ 2
    @tensor excite_left[a, b; c] := ϕ[b, l] * ψ.AC[mid-offset][a, l; c]
    ψ.AC[mid-offset] = excite_left
    @tensor excite_right[a, b; c] := ϕ[b, l] * ψ.AC[mid+offset][a, l; c]
    ψ.AC[mid+offset] = excite_right

    # normalize after applying the excitations
    normalize!(ψ)

    field_exp = zeros(Float64, T, L)

    x = 1:L
    y = 0:dt:(dt*(T-1))
    plt = heatmap(x, y, field_exp, dpi=600, xlabel=L"Lattice position $n$", ylabel=L"Lattice time $t$", title=L"$\langle\phi\rangle$ at $\tilde{\lambda} = $λ0, $\tilde{\mu}_0^2 = $μ0_sq")

    for i in 1:L
        field_exp[1, i] = real(expectation_value(ψ, i => ϕ))
    end
    init_exp = plot(field_exp[1, :])
    display(init_exp)

    for t_step in 1:T-1
        println("Currently at step $t_step")
        ψ, _ = timestep(ψ, ham, t_step - 1, dt, TDVP())
        for i in 1:L
            field_exp[t_step+1, i] = real(expectation_value(ψ, i => ϕ))
        end
        heatmap!(plt, x, y, field_exp, overwrite=true)
        display(plt)
    end

    filename = replace("scattering_finite_L_$(L)_T_$(T)_musq_$(μ0_sq)_d_$(d)_D_$(D)_dt_$(dt)", "." => "p", "-" => "m")

    # using mkpath since its idempotent
    mkpath("./plots_finite")
    mkpath("./data_finite")

    savefig(plt, "./plots_finite/" * filename * ".png")
    println("Plot saved.")

    @save "./data_finite/" * filename * ".jld2" field_exp
    println("Data saved.")
end

main()