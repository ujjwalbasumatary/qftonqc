using MPSKit, TensorKit, Plots, JLD2, LaTeXStrings, ArgParse, LinearAlgebra
using QFTSimulations: commensurate_momentum_grid, two_particle_packet_tensors
BLAS.set_num_threads(1)

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--delta_p", "-p"
        help = "Momentum spacing"
        arg_type = Float64
        default = 0.1
        "--bond_dimension", "-D"
        help = "Ground-state bond dimension"
        arg_type = Int
        default = 10
        "--evolution_bond_dimension"
        help = "Maximum bond dimension during two-site TDVP (0 uses 2D)"
        arg_type = Int
        default = 0
        "--total_time", "-T"
        help = "Number of saved time samples, including t = 0"
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
        default = 1.06
        "--h_z", "-z"
        help = "Z field strength"
        arg_type = Float64
        default = 0.01
        "--sigma", "-s"
        help = "Spread of the momentum space wavepacket"
        arg_type = Float64
        default = 0.1
        "--output_dir", "-o"
        help = "Root directory in which plots/ and data/ are created"
        arg_type = String
        default = normpath(joinpath(@__DIR__, "..", "..", "results", "ift"))
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

function create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)
    r"""
    Returns the wavepacket at n
    \sum_p c_p e^{i p n} B(p)
    """
    p_max = length(B_tensor_list)
    shape_tensor = size(B_tensor_list[1])
    B_packet = zeros(ComplexF64, shape_tensor)
    for i in 1:p_max
        δp = mod((i - mom_idx) * Δp + π, 2π) - π
        p = -π + (i - 1) * Δp
        B_packet += exp(-δp^2 / sigma^2) * exp(im * p * (n - offset)) * B_tensor_list[i]
    end
    return B_packet
end

function nearest_momentum_index(momentum, Δp, n_momenta)
    wrapped_momentum = mod(momentum + π, 2π) - π
    return mod(round(Int, (wrapped_momentum + π) / Δp), n_momenta) + 1
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

    length(B_packet_list_left) == L == length(B_packet_list_right) ||
        throw(DimensionMismatch("both packet supports must have length L"))

    C = ψ_gs.C[]
    Cinv = convert(Array, C \ id(domain(C)))
    dense_window = two_particle_packet_tensors(
        AL_array, AR_array, Cinv, B_packet_list_left, B_packet_list_right
    )

    # Each packet is closed before the next is reopened. This guarantees one B
    # insertion in each region; one block-upper chain across both regions would
    # instead encode a single particle in a superposition of two locations.
    return [
        TensorMap(mat, ℂ^(size(mat, 1)) ⊗ ℂ^d ← ℂ^(size(mat, 3)))
        for mat in dense_window
    ]
end

function main()
    # Parse arguments for the simulation.
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"] # max bond dimension
    D_evolution_arg = parsed_args["evolution_bond_dimension"]
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]
    requested_Δp = parsed_args["delta_p"]
    σ = parsed_args["sigma"]
    h_z = parsed_args["h_z"]
    h_x = parsed_args["h_x"]
    J = parsed_args["J"]
    mom = parsed_args["mom"]
    output_root = abspath(parsed_args["output_dir"])

    D > 0 || throw(ArgumentError("bond_dimension must be positive"))
    T > 0 || throw(ArgumentError("total_time must be positive"))
    dt > 0 || throw(ArgumentError("time_step must be positive"))
    requested_Δp > 0 || throw(ArgumentError("delta_p must be positive"))
    isfinite(mom) || throw(ArgumentError("mom must be finite"))
    D_evolution = iszero(D_evolution_arg) ? 2D : D_evolution_arg
    D_evolution >= 2D || throw(ArgumentError(
        "evolution_bond_dimension must be at least 2 * bond_dimension = $(2D)"
    ))

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end
    println("effective evolution_bond_dimension = $D_evolution")

    n_momenta = round(Int, 2π / requested_Δp)
    n_momenta >= 2 || throw(ArgumentError("delta_p must produce at least two momentum points"))
    momenta = commensurate_momentum_grid(n_momenta)
    Δp = step(momenta)
    L_sites = length(momenta)
    println("commensurate momentum grid: n = $n_momenta, delta_p = $Δp")

    ham = get_ham(J, h_x, h_z)
    ψ_gs = prep_gs(D, ham)
    ξ = correlation_length(ψ_gs)
    println("Correlation length: $ξ")
    println("Ground state energy: $(expectation_value(ψ_gs, ham))")

    energies, states = get_QPstate(ψ_gs, ham, momenta)
    offset_left = L_sites ÷ 2 # where in the window I want the packet to appear?
    offset_right = L_sites ÷ 2

    mom_idx_left = nearest_momentum_index(mom, Δp, n_momenta)
    mom_idx_right = nearest_momentum_index(-mom, Δp, n_momenta)
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

    # time evolution loop starts here
    evolution_alg = TDVP2(; trscheme=truncrank(D_evolution))
    for t_step in 2:T
        println("\rCurrently at step $t_step")
        t = (t_step - 2) * dt
        ψ_window, _ = timestep(ψ_window, ham, t, dt, evolution_alg)
        for i in 1:L-1
            energy_exp[t_step, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
            s_z_exp[t_step, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
        end
        s_z_exp[t_step, L] = real(expectation_value(ψ_window, L => σ_z)) - gs_value_s_z[L]

    end

    times = (0:(T - 1)) .* dt
    energy_plot = heatmap(
        1:(L - 1), times, energy_exp[:, 1:(L - 1)]; dpi=600,
        title=L"$E - E_{vac}$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$",
        xlabel=L"Lattice bond $n$", ylabel=L"Lattice time $t$"
    )
    sz_plot = heatmap(
        1:L, times, s_z_exp; dpi=600,
        title=L"$S_z - (S_z)_{vac}$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$",
        xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$"
    )

    plot_dir = joinpath(output_root, "plots")
    data_dir = joinpath(output_root, "data")
    mkpath(plot_dir)
    mkpath(data_dir)

    filename = replace("scattering_infinite_J_mom_$(mom)_$(J)_hx_$(h_x)_hz_$(h_z)_dp_$(Δp)_sigma_$(σ)_T_$(T)_D_$(D)_Dmax_$(D_evolution)_dt_$(dt)", '.' => 'p', '-' => 'm')
    full_path_sz_image = joinpath(plot_dir, "$(filename)_sz.png")
    full_path_energy_image = joinpath(plot_dir, "$(filename)_energy.png")

    savefig(sz_plot, full_path_sz_image)
    savefig(energy_plot, full_path_energy_image)
    println("Plot saved.")

    energy_path = joinpath(data_dir, "$(filename)_energy.jld2")
    sz_path = joinpath(data_dir, "$(filename)_sz.jld2")
    @save energy_path energy_exp times
    @save sz_path s_z_exp times
    println("Data saved.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
