using MPSKit, TensorKit, Plots, JLD2, LaTeXStrings, ArgParse, LinearAlgebra
BLAS.set_num_threads(1)

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--delta_p", "-p"
        help = "Momentum spacing"
        arg_type = Float64
        default = 0.1
        "--bond_dimension", "-D"
        help = "Maximum bond dimension"
        arg_type = Int
        default = 10
        "--total_time", "-T"
        help = "Maximum time step for the evolution"
        arg_type = Int
        default = 800
        "--mom", "-m"
        help = "Momentum of the wavepacket"
        arg_type = Float64
        default = 0.3
        "--time_step", "-t"
        help = "Step size for time evolution"
        arg_type = Float64
        default = 0.1
        "--J"
        help = "Coupling strength"
        arg_type = Float64
        default = 1.0
        "--h_x", "-x"
        help = "X field strength"
        arg_type = Float64
        default = 0.0
        "--h_z", "-z"
        help = "Z field strength"
        arg_type = Float64
        default = 0.0
        "--sigma", "-s"
        help = "Spread of the momentum space wavepacket"
        arg_type = Float64
        default = 0.1
    end

    return parse_args(ARGS, s)
end

function get_ops()
    """
    This function returns the Pauli matrices as TensorMaps.
    """
    σ_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    σ_y = TensorMap(ComplexF64[0 -im; im 0], ℂ^2 ← ℂ^2)
    σ_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    return σ_x, σ_y, σ_z
end

function get_ham(J, h_x, h_z)
    """
    This function builds the Hamiltonian as an InfiniteMPOHamiltonian object.
    """
    σ_x, σ_y, σ_z = get_ops()
    ham = InfiniteMPOHamiltonian(PeriodicVector([ℂ^2]), 1 => -h_x * σ_x - h_z * σ_z, (1, 2) => -J * σ_z ⊗ σ_z)
    return ham
end

function prep_gs(D, ham)
    ψ0 = InfiniteMPS(ℂ^2, ℂ^D)
    ψ, env, eps = find_groundstate(ψ0, ham, VUMPS())
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
        # ------------------------------

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
    @assert D == D2 "A must be D x d x D"

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

    # the piece with C^{-1} 'glued in'
    # c = ψ_gs.C[]
    # c_inv = c \ id(domain(c))
    # @tensor AR_glue_TM[a, b; c] := AR[a, b; d] * c_inv[d; c]
    # AR_glue = convert(Array, AR_glue_TM)

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

    # start next window
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
    # Parse arguments for the simulation.
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"] # max bond dimension
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]
    Δp = parsed_args["delta_p"]
    σ = parsed_args["sigma"]
    h_z = parsed_args["h_z"]
    h_x = parsed_args["h_x"]
    J = parsed_args["J"]
    mom = parsed_args["mom"]

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end

    L_sites = floor(Int, 2 * π / Δp) # number of sites spanning the states   

    ham = get_ham(J, h_x, h_z)
    ψ_gs = prep_gs(D, ham)
    ξ = correlation_length(ψ_gs)
    println("Correlation length: $ξ")
    println("Ground state energy: $(expectation_value(ψ_gs, ham))")

    energies, states = get_QPstate(ψ_gs, ham, Δp)
    offset_left = L_sites ÷ 2 # where in the window I want the packet to appear?
    offset_right = L_sites ÷ 2

    mom_idx_left = 1 + floor(Int, (mom + π) / Δp)
    mom_idx_right = 1 + floor(Int, (-mom + π) / Δp)
    B_tensor_list = get_B_tensor_list(states)
    B_packet_list_left = [create_B_packet(B_tensor_list, n, offset_left, mom_idx_left, Δp, σ) for n in 1:L_sites]
    B_packet_list_right = [create_B_packet(B_tensor_list, n, offset_right, mom_idx_right, Δp, σ) for n in 1:L_sites]
    wavepacket_window = create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L_sites)
    L, = size(wavepacket_window)
    Iden = TensorMap(Matrix{ComplexF64}(I, 2, 2), ℂ^2 ← ℂ^2)
    # define the Hamiltonian density to measure as time evolves
    ψ_window = WindowMPS(ψ_gs, wavepacket_window)
    σ_x, σ_y, σ_z = get_ops()
    ham_density = -(J * σ_z ⊗ σ_z) - 0.5 * (h_x * σ_x + h_z * σ_z) ⊗ Iden - 0.5 * Iden ⊗ (h_x * σ_x + h_z * σ_z)

    # normalize the states
    normalize!(ψ_window)

    energy_exp = zeros(Float64, T, L)
    s_z_exp = zeros(Float64, T, L)

    # need to have the ground state values for comparison
    gs_value_energy = [real(expectation_value(ψ_gs, (i, i + 1) => ham_density)) for i in 1:L-1]
    gs_value_s_z = [real(expectation_value(ψ_gs, i => σ_z)) for i in 1:L]

    # we should also fill the first row with the values at t = 0.
    for i in 1:L-1
        energy_exp[1, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
        s_z_exp[1, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
    end
    s_z_exp[1, L] = real(expectation_value(ψ_window, L => σ_z)) - gs_value_s_z[L]

    # create the plots for energy and s_z
    energy_plot = heatmap(1:L, dt * [1:T], energy_exp, dpi=600, title=L"$E - E_{vac}$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$", xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$")
    sz_plot = heatmap(1:L, dt * [1:T], s_z_exp, dpi=600, title=L"$S_z$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$", xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$")

    # time evolution loop starts here
    for t_step in 2:T
        println("\rCurrently at step $t_step")
        ψ_window, _ = timestep(ψ_window, ham, t_step - 2, dt, TDVP())
        for i in 1:L-1
            energy_exp[t_step, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
            s_z_exp[t_step, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
        end
        s_z_exp[t_step, L] = real(expectation_value(ψ_window, L => σ_z)) - gs_value_s_z[L]

        heatmap!(energy_plot, 1:L, dt * [1:T], energy_exp, dpi=600, xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$", overwrite=true)
        heatmap!(sz_plot, 1:L, dt * [1:T], s_z_exp, dpi=600, xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$", overwrite=true)
        display(energy_plot)
    end

    # make the folder plots if it does not exist
    mkpath("./plots")
    mkpath("./data")

    filename = replace("scattering_infinite_J_mom_$(mom)_$(J)_hx_$(h_x)_hz_$(h_z)_dp_$(Δp)_sigma_$(σ)_T_$(T)_D_$(D)_dt_$(dt)", '.' => 'p', '-' => 'm')
    full_path_sz_image = "./plots/$(filename)_sz.png"
    full_path_energy_image = "./plots/$(filename)_energy.png"

    savefig(full_path_sz_image)
    savefig(full_path_energy_image)
    println("Plot saved.")

    @save "./data/$(filename)_energy.jld2" energy_exp
    @save "./data/$(filename)_sz.jld2" s_z_exp
    println("Data saved.")
end

main()