using MPSKit, TensorKit, Plots, LaTeXStrings, LinearAlgebra, JLD2, ArgParse

BLAS.set_num_threads(1)

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--delta_p", "-p"
        help = "Momentum spacing"
        arg_type = Float64
        default = 0.1
        "--local_dim", "-d"
        help = "Local Hilbert space dimension"
        arg_type = Int
        default = 5
        "--bond_dimension", "-D"
        help = "Maximum bond dimension"
        arg_type = Int
        default = 10
        "--lambda", "-l"
        help = "Bare lattice coupling lambda"
        arg_type = Float64
        default = 2.0
        "--mu_sq", "-m"
        help = "Bare lattice coupling squared"
        arg_type = Float64
        default = 0.5
        "--total_time", "-T"
        help = "Maximum time step for the evolution"
        arg_type = Int
        default = 800
        "--time_step", "-t"
        help = "Step size for time evolution"
        arg_type = Float64
        default = 0.1
        "--sigma", "-s"
        help = "Spread of the momentum space wavepacket"
        arg_type = Float64
        default = 0.1
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

function get_ham(d, μ0, λ0)
    """
    Prepares the infinte MPO Hamiltonian from μ0, λ0 and
    the local Hilbert space dimension d as inputs
    """
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)
    chain = PeriodicVector([ℂ^d])
    single_site_term = (μ0 * ϕ2 + π2) / 2 + λ0 * ϕ4 / 24 + ϕ2
    two_site_term = -ϕ ⊗ ϕ
    ham = InfiniteMPOHamiltonian(chain, 1 => single_site_term, (1, 2) => two_site_term)
    return ham
end

function prep_gs(d, D, ham)
    """
    Returns the converged ground state of the uMPS
    """
    ψ0 = InfiniteMPS(ℂ^d, ℂ^D)
    ψ, _, _ = find_groundstate(ψ0, ham, VUMPS(; tol=1e-12))
    return ψ
end

function get_QPstate(ψ_gs, ham, Δp)
    energies, states = excitations(ham, QuasiparticleAnsatz(), -π:Δp:π-Δp, ψ_gs)
    return energies, states
end

function get_B_tensor_list(states)
    """
    Returns a Vector of gauge-fixed B tensors (as dense arrays)
    for each state in `states`.
    """
    n = length(states)
    B_list = Vector{Array{ComplexF64,3}}(undef, n)

    for i in 1:n
        VLs = states[i].VLs[1]
        Xs = states[i].Xs[1]

        B_tensor = VLs * Xs
        B_array = convert(Array, B_tensor)

        # take the window / chosen sector
        B_array_new = B_array[:, :, 1, :]   # 3D array now

        θ = angle(B_array_new[1, 1, 1])
        B_array_new .*= exp(-im * θ)
        B_list[i] = B_array_new
    end

    return B_list
end

function create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)
    r"""
    Returns the wavepacket at n
    \sum_p c_p e^{i p n} B(p)
    """
    p_max = length(B_tensor_list)
    shape_tensor = size(B_tensor_list[1])
    B_packet = zeros(ComplexF64, shape_tensor)
    for i in 1:p_max
        B_packet += exp(-((i - mom_idx) * Δp)^2 / sigma^2) * exp(im * (-π + (i - 1) * Δp) * (n - offset)) * B_tensor_list[i]
    end
    return B_packet
end

function block_upper(AL, AR, B)
    D, d, D2 = size(AL)
    @assert size(B) == (D, d, D2)
    @assert D == D2 "A must be D×d×D"

    T = zeros(eltype(AL), 2D, d, 2D)

    # top-left block = AL
    T[1:D, :, 1:D] .= AL

    # top-right block = B
    T[1:D, :, D+1:2D] .= B

    # bottom-left block = 0  (already zero)

    # bottom-right block = AR
    T[D+1:2D, :, D+1:2D] .= AR

    return T
end


function create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)
    """
    Takes the ground state left and right environments and constructs
    the stacked tensors.
    """
    AL = ψ_gs.AL[]
    AR = ψ_gs.AR[]
    AL_array = convert(Array, AL)
    AR_array = convert(Array, AR)
    D, d, _ = size(AL_array)

    # create a window of the same type as AL i.e. TensorMap
    window = eltype(ψ_gs.AL)[]
    """
    B_1 is left moving and B_2 is right moving.
    So B_2 spans 1:mid and B_1 spans mid + 1 : L
    """
    mat = cat(AL_array, B_packet_list_left[1]; dims=3)
    tensor = TensorMap(mat, ℂ^(D) ⊗ ℂ^d ← ℂ^(2D))
    push!(window, tensor)
    for i in 2:L
        mat = block_upper(AL_array, AR_array, B_packet_list_left[i])
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    ### Start next window ###
    for i in 1:L-1
        mat = block_upper(AL_array, AR_array, B_packet_list_right[i])
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    mat = cat(B_packet_list_right[L], AR_array; dims=1)
    tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(D))
    push!(window, tensor)
    return window
end

function main()
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"] # max bond dimension
    d = parsed_args["local_dim"] # local Hilbert space dim
    λ0 = parsed_args["lambda"]
    μ0_sq = parsed_args["mu_sq"]
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]
    Δp = parsed_args["delta_p"]
    σ = parsed_args["sigma"]

    L_sites = floor(Int, 2 * π / Δp) # number of sites spanning the states   

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end

    ham = get_ham(d, μ0_sq, λ0)
    ψ_gs = prep_gs(d, D, ham)
    energies, states = get_QPstate(ψ_gs, ham, Δp)
    offset_left = L_sites ÷ 2 # where in the window I want the packet to appear?
    offset_right = L_sites ÷ 2

    # now I also need the momentum about which the packets are centered, index to be specific
    # this is the momentum about which the packets will be centered
    mom = π / 2
    mom_idx_left = 1 + floor(Int, (mom + π) / Δp)
    mom_idx_right = 1 + floor(Int, (-mom + π) / Δp)
    B_tensor_list = get_B_tensor_list(states)
    B_packet_list_left = [create_B_packet(B_tensor_list, n, offset_left, mom_idx_left, Δp, σ) for n in 1:L_sites]
    B_packet_list_right = [create_B_packet(B_tensor_list, n, offset_right, mom_idx_right, Δp, σ) for n in 1:L_sites]
    wavepacket_window = create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L_sites)
    L, = size(wavepacket_window)
    Iden = TensorMap(Matrix{ComplexF64}(I, d, d), ℂ^d ← ℂ^d)
    # define the Hamiltonian density to measure as time evolves
    ϕ, ϕ2, π2, ϕ4 = matrix_elems(d)
    ham_density = ((μ0_sq * ϕ2 + π2) / 2 + λ0 * ϕ4 / 24 + ϕ2) ⊗ Iden - ϕ ⊗ ϕ
    ψ_window = WindowMPS(ψ_gs, wavepacket_window)

    # normalize the states
    normalize!(ψ_window)

    energy_exp = zeros(Float64, T, L)
    phi_sq_exp = zeros(Float64, T, L)

    # need to have the ground state values for comparison
    gs_value_energy = [real(expectation_value(ψ_gs, (i, i + 1) => ham_density)) for i in 1:L-1]
    gs_value_phi_sq = [real(expectation_value(ψ_gs, i => ϕ2)) for i in 1:L]

    # we should also fill the first row with the values at t = 0.
    for i in 1:L-1
        energy_exp[1, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
        phi_sq_exp[1, i] = real(expectation_value(ψ_window, i => ϕ2)) - gs_value_phi_sq[i]
    end
    phi_sq_exp[1, L] = real(expectation_value(ψ_window, L => ϕ2)) - gs_value_phi_sq[L]
    dt = 0.1 # time step for evolution
    # time evolution loop starts here
    for t_step in 2:T
        println("\rCurrently at step $t_step")
        ψ_window, _ = timestep(ψ_window, ham, t_step - 1, dt, TDVP())
        for i in 1:L-1
            energy_exp[t_step, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
            phi_sq_exp[t_step, i] = real(expectation_value(ψ_window, i => ϕ2)) - gs_value_phi_sq[i]
        end
        phi_sq_exp[t_step, L] = real(expectation_value(ψ_window, L => ϕ2)) - gs_value_phi_sq[L]
    end
    heatmap(energy_exp, dpi=600)

    # make the folder plots if it does not exist
    if !isdir("./plots")
        mkdir("./plots")
    end

    filename = replace("scattering_infinite_dp_$(Δp)_sigma_$(σ)_T_$(T)_mu0sq_$(μ0_sq)_lam_$(λ0)_d_$(d)_D_$(D)_dt_$(dt)", '.' => 'p', '-' => 'm')
    full_path_image = "./plots/" * filename * ".png"

    savefig(full_path_image)
    println("Plot saved.")

    if !isdir("./data")
        mkdir("./data")
    end
    @save "./data/energy_" * filename * ".jld2" energy_exp
    @save "./data/phi_sq_" * filename * ".jld2" phi_sq_exp
    println("Data saved.")

end

main()
