using MPSKit, TensorKit, Plots, JLD2, LaTeXStrings, ArgParse, LinearAlgebra

BLAS.set_num_threads(1)

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--bond_dimension", "-D"
        help = "Maximum bond dimension"
        arg_type = Int
        default = 10
        "--length", "-L"
        help = "Length of the window"
        arg_type = Int
        default = 200
        "--n_center", "-n"
        help = "Center of the wavepacket"
        arg_type = Int
        default = 50
        "--kappa", "-k"
        help = "Momentum about which the wavepacket is centered"
        arg_type = Float64
        default = 0.3
        "--total_time", "-T"
        help = "Total time for the evolution"
        arg_type = Int
        default = 800
        "--time_step", "-t"
        help = "Step size for time evolution"
        arg_type = Float64
        default = 0.1
        "--h_x", "-x"
        help = "X field strength"
        arg_type = Float64
        default = 1.06
        "--h_z", "-z"
        help = "Z field strength"
        arg_type = Float64
        default = 0.01
        "--sigma", "-s"
        help = "Spread of the position space wavepacket"
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

function get_ham(h_x, h_z)
    """
    This function builds the Hamiltonian as an InfiniteMPOHamiltonian object.
    """
    σ_x, σ_y, σ_z = get_ops()
    ham = InfiniteMPOHamiltonian(PeriodicVector([ℂ^2]), 1 => -h_x * σ_x - h_z * σ_z, (1, 2) => -1 * σ_z ⊗ σ_z)
    return ham
end

function prep_gs(D, ham)
    ψ0 = InfiniteMPS(ℂ^2, ℂ^D)
    ψ, env, eps = find_groundstate(ψ0, ham, VUMPS())
    return ψ
end

function get_QPstate(ψ_gs, ham, momenta)
    energies, states = excitations(ham, QuasiparticleAnsatz(), momenta, ψ_gs)
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

function create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)
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
    B_list[1] is right moving and
    B_list[2] is left moving
    """
    mat = cat(AL_array, B_list[1] * exp(im * (1 - n_center) * κ - (1 - n_center)^2 / σ^2); dims=3)
    tensor = TensorMap(mat, ℂ^(D) ⊗ ℂ^d ← ℂ^(2D))
    push!(window, tensor)
    Lhalf = L ÷ 2
    for i in 2:Lhalf
        mat = block_upper(AL_array, AR_array, B_list[1] * exp(im * (i - n_center) * κ - (i - n_center)^2 / σ^2))
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    for i in Lhalf+1:L-1
        mat = block_upper(AL_array, AR_array, B_list[2] * exp(-im * (i - L + n_center) * κ - (i - (L - n_center))^2 / σ^2))
        tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(2D))
        push!(window, tensor)
    end
    mat = cat(B_list[2] * exp(-im * (L - (L - n_center)) * κ - (L - (L - n_center))^2 / σ^2), AR_array; dims=1)
    tensor = TensorMap(mat, ℂ^(2D) ⊗ ℂ^d ← ℂ^(D))
    push!(window, tensor)
    return window
end

function main(parsed_args)
    # Parse arguments for the simulation.
    D = parsed_args["bond_dimension"] # max bond dimension
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]
    σ = parsed_args["sigma"] # position space spread
    L = parsed_args["length"]
    n_center = parsed_args["n_center"]
    κ = parsed_args["kappa"] # momentum about which packet is centered
    h_z = parsed_args["h_z"]
    h_x = parsed_args["h_x"]

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end

    ham = get_ham(h_x, h_z)
    ψ_gs = prep_gs(D, ham)
    ξ = correlation_length(ψ_gs)
    println("Correlation length: $ξ")
    println("Ground state energy: $(expectation_value(ψ_gs, ham))")

    # this is the momentum about which the packets will be centered
    energies, states = get_QPstate(ψ_gs, ham, [κ, -κ])

    # now I also need the momentum about which the packets are centered, index to be specific
    B_list = get_B_tensor_list(states)
    wavepacket_window = create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)
    # need the identity matrix
    Iden = TensorMap(Matrix{ComplexF64}(I, 2, 2), ℂ^2 ← ℂ^2)
    # define the Hamiltonian density to measure as time evolves
    ψ_window = WindowMPS(ψ_gs, wavepacket_window)
    σ_x, σ_y, σ_z = get_ops()
    ham_density = -(σ_z ⊗ σ_z) - 0.5 * (h_x * σ_x + h_z * σ_z) ⊗ Iden - 0.5 * Iden ⊗ (h_x * σ_x + h_z * σ_z)

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
    # time evolution loop starts here
    for t_step in 2:T
        println("\rCurrently at step $t_step")
        ψ_window, _ = timestep(ψ_window, ham, t_step - 1, dt, TDVP())
        for i in 1:L-1
            energy_exp[t_step, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
            s_z_exp[t_step, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
        end
        s_z_exp[t_step, L] = real(expectation_value(ψ_window, L => σ_z)) - gs_value_s_z[L]
    end
    heatmap(energy_exp, dpi=600)

    # make the folder plots if it does not exist
    mkpath("./fixed_mom/plots")

    filename = replace("scattering_infinite_mom_$(κ)_hx_$(h_x)_hz_$(h_z)_sigma_$(σ)_L_$(L)_n_$(n_center)_T_$(T)_D_$(D)_dt_$(dt)", '.' => 'p', '-' => 'm')
    full_path_image = "./fixed_mom/plots/" * filename * ".png"

    savefig(full_path_image)
    println("Plot saved.")

    mkpath("./fixed_mom/data")

    @save "./fixed_mom/data/energy_" * filename * ".jld2" energy_exp
    @save "./fixed_mom/data/sz_value_" * filename * ".jld2" s_z_exp
    println("Data saved.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_cmdline()
    main(parsed_args)
end
