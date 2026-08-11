using MPSKit, TensorKit, LaTeXStrings, SpecialFunctions, Plots, ArgParse, JLD2

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--win_size", "-L"
            help = "Window size"
            arg_type = Int
            default = 100
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
            default = 15
        "--lambda", "-l"
            help = "Bare lattice coupling lambda"
            arg_type = Float64
            default = 0.2
        "--mu_sq", "-k"
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
    phi    = zeros(ComplexF64, (d, d))
    phi_sq = zeros(ComplexF64, (d, d))
    pi_sq  = zeros(ComplexF64, (d, d))
    phi_4  = zeros(ComplexF64, (d, d))

    # phi = (a + a^\dagger)/sqrt(2)
    for i in 2:d
    	val = sqrt((i - 1)/ 2)
        phi[i, i - 1] = val
        phi[i - 1, i] = val
    end

    # helper to fill phi_sq and pi_sq (same structure, sign flip on off-diagonals)
    function fill_quadratic!(M, sign)
        @inbounds for i in 1:d
            if i < d - 1
                val = sign * sqrt(i * (i + 1)) / 2
                M[i, i + 2] = val
                M[i + 2, i] = val  # Hermitian
            end
            M[i, i] = (2 * i - 1) / 2
        end
    end

    fill_quadratic!(phi_sq, +1)
    fill_quadratic!(pi_sq,  -1)

    # phi^4 in harmonic oscillator basis
    @inbounds for i in 1:d
        n = i - 1  # occupation number

        # diagonal
        phi_4[i, i] = (6 * n^2 + 6 * n + 3) / 4

        # connect |n> <-> |n+2>
        if i + 2 <= d
            val = (4 * n + 6) * sqrt((n + 1) * (n + 2)) / 4
            j = i + 2
            phi_4[i, j] = val
            phi_4[j, i] = val  # Hermitian
        end

        # connect |n> <-> |n+4>
        if i + 4 <= d
            val = sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4
            j = i + 4
            phi_4[i, j] = val
            phi_4[j, i] = val  # Hermitian
        end
    end

    ϕ  = TensorMap(phi,    ℂ^d ← ℂ^d)
    ϕ2 = TensorMap(phi_sq, ℂ^d ← ℂ^d)
    π2 = TensorMap(pi_sq,  ℂ^d ← ℂ^d)
    ϕ4 = TensorMap(phi_4,  ℂ^d ← ℂ^d)

    return ϕ, ϕ2, π2, ϕ4
end

function get_ham(d, μ0_sq, λ0)
    """
    Prepares the infinite MPO Hamiltonian from μ0, λ0 and
    the local Hilbert space dimension d as inputs
    """
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)
    chain = PeriodicVector([ℂ^d])
    single_site_term = (μ0_sq * ϕ2 + π2) / 2 + λ0 * ϕ4 / 24 + ϕ2
    two_site_term = - ϕ ⊗ ϕ
    ham = InfiniteMPOHamiltonian(chain, 1 => single_site_term, (1, 2) => two_site_term)
    return ham
end

function prep_gs(d, D, ham)
    """
    Returns the converged ground state of the uMPS using the VUMPS algorithm
    """
    ψ0 = InfiniteMPS(ℂ^d, ℂ^D)
    ψ, _, _ = find_groundstate(ψ0, ham, VUMPS())
    return ψ
end

function main()
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"] # max bond dimension
    L = parsed_args["win_size"] # number of sites
    d = parsed_args["local_dim"] # local Hilbert space dim
    λ0 = parsed_args["lambda"]
    μ0_sq = parsed_args["mu_sq"]
    offset = parsed_args["offset"]
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
      println("$arg = $val")
    end


    ham = get_ham(d, μ0_sq, λ0)

    ψ_gs = prep_gs(d, D, ham)

    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)

    ξ = correlation_length(ψ_gs) # ground state lattice correlation length
    
    # create a WindowMPS object
    ψ_window = WindowMPS(ψ_gs, L)

    # apply the field operator
    mid = L ÷ 2
    @tensor excite_left[a, b; c] := ϕ[b, l] * ψ_window.AC[mid - offset][a, l; c]
    @tensor excite_right[a, b; c] := ϕ[b, l] * ψ_window.AC[mid + offset][a, l; c]
    ψ_window.AC[mid - offset] = excite_left
    ψ_window.AC[mid + offset] = excite_right

    # normalize after applying the excitations
    normalize!(ψ_window)

    field_exp = zeros(Float64, T, L)

    for i in 1:L
        field_exp[1, i] = real(expectation_value(ψ_window, i => ϕ)) # get rid of small imaginary part
    end

    for t_step in 2:T
        println("Currently at step $t_step")
        ψ_window, _ = timestep(ψ_window, ham, t_step - 1, dt, TDVP())
        for i in 1:L
            field_exp[t_step, i] = real(expectation_value(ψ_window, i => ϕ))
        end
    end

    x = 1:L
    y = 0:dt:(dt * (T - 1))

    heatmap(x, y, field_exp, dpi=600, xlabel=L"$n$", ylabel=L"$t$", title=L"$\langle\phi\rangle$ at $\tilde{\lambda} = %$λ0, \tilde{\mu}_0^2 = %$(μ0_sq)$")

    # make the folder plots if it does not exist
    if !isdir("./plots")
        mkdir("./plots")
    end

    filename = replace("scattering_infinite_L_$(L)_T_$(T)_mu0sq_$(μ0_sq)_d_$(d)_D_$(D)_dt_$(dt)", '.' => 'p', '-' => 'm')
    full_path_image = "./plots/" * filename * ".png"

    savefig(full_path_image)
    println("Plot saved.")

    if !isdir("./data")
        mkdir("./data")
    end
    full_path_data = "./data/" * filename * ".jld2"
    @save full_path_data field_exp
    println("Data saved.")
end

main()
