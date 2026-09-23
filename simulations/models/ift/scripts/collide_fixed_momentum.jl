using MPSKit, TensorKit, Plots, JLD2, LaTeXStrings, ArgParse, LinearAlgebra
using QFTSimulations: two_particle_packet_tensors

BLAS.set_num_threads(1)

"""
    DEFAULT_OUTPUT_DIRECTORY

Directory used when `--output_dir` is omitted. Figures are saved in
`results/ift/plots/`, and the arrays used to make them are saved in
`results/ift/data/`.
"""
const DEFAULT_OUTPUT_DIRECTORY = normpath(
    joinpath(@__DIR__, "..", "..", "..", "..", "results", "ift")
)

"""
    parse_cmdline()

Read the collision parameters from `ARGS` and return the dictionary produced by
`ArgParse.parse_args`.

`total_time` is the number of stored rows, including the initial state, rather
than a physical duration. With time step `dt`, the last stored time is
`(total_time - 1)dt`. `length` is the number of sites in the finite window and
must be even when passed to `main`. `n_center` locates the left packet; the
right packet is centred at `length - n_center`. The position-space amplitude
is proportional to `exp(-x^2/sigma^2)`. `evolution_bond_dimension = 0` asks
`main` to use twice the vacuum bond dimension.

Calling this function with `--help` prints the option list and exits through
ArgParse.
"""
function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--bond_dimension", "-D"
        help = "Ground-state bond dimension"
        arg_type = Int
        default = 10
        "--evolution_bond_dimension"
        help = "Maximum bond dimension during two-site TDVP (0 uses 2D)"
        arg_type = Int
        default = 0
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
        help = "Number of saved time samples, including t = 0"
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
        default = 10.0
        "--output_dir", "-o"
        help = "Root directory in which plots/ and data/ are created"
        arg_type = String
        default = DEFAULT_OUTPUT_DIRECTORY
    end

    return parse_args(ARGS, s)
end

@doc raw"""
    get_ops()

Return ``(\sigma^x,\sigma^y,\sigma^z)`` as complex `TensorMap`s on a
two-dimensional local Hilbert space. The matrices use the ``\sigma^z`` basis,

```math
\sigma^x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
\sigma^y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\qquad
\sigma^z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
```
"""
function get_ops()
    σ_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    σ_y = TensorMap(ComplexF64[0 -im; im 0], ℂ^2 ← ℂ^2)
    σ_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    return σ_x, σ_y, σ_z
end

@doc raw"""
    get_ham(h_x, h_z)

Construct the Hamiltonian as a one-site uniform matrix product operator (MPO),

```math
H=-\sum_j\left(\sigma_j^z\sigma_{j+1}^z
  +h_x\sigma_j^x+h_z\sigma_j^z\right).
```

where the nearest-neighbour coupling and lattice spacing are one. In the
command-line calculation, `h_x` and `h_z` are real dimensionless couplings.
"""
function get_ham(h_x, h_z)
    σ_x, σ_y, σ_z = get_ops()
    ham = InfiniteMPOHamiltonian(PeriodicVector([ℂ^2]), 1 => -h_x * σ_x - h_z * σ_z, (1, 2) => -1 * σ_z ⊗ σ_z)
    return ham
end

"""
    prep_gs(D, ham)

Find a one-site uniform matrix product state for the ground state of `ham`. The
initial state has local space `ℂ^2` and virtual space `ℂ^D`; the variational
uniform matrix product state (VUMPS) calculation uses its default stopping
settings. Only the state returned by `find_groundstate` is kept.
"""
function prep_gs(D, ham)
    ψ0 = InfiniteMPS(ℂ^2, ℂ^D)
    ψ, env, eps = find_groundstate(ψ0, ham, VUMPS())
    return ψ
end

"""
    get_QPstate(ψ_gs, ham, momenta)

Return the excitation energies and tangent-space states obtained from
`MPSKit.excitations(ham, QuasiparticleAnsatz(), momenta, ψ_gs)`. Entries follow
the order of `momenta`. This is MPSKit's default quasiparticle branch; the
function does not identify or track a particle species across different
momenta.
"""
function get_QPstate(ψ_gs, ham, momenta)
    energies, states = excitations(ham, QuasiparticleAnsatz(), momenta, ψ_gs)
    return energies, states
end

"""
    get_B_tensor_list(states)

Extract one dense excitation tensor from every tangent-space state. For state
`i`, the tensor is formed as `states[i].VLs[1] * states[i].Xs[1]`; after
conversion to an array, the slice `[:, :, 1, :]` is retained. The result is a
`Vector{Array{ComplexF64,3}}` in left-bond, physical, right-bond order.

Each tensor is multiplied by `exp(-im * angle(B[1,1,1]))`, making that component
real and nonnegative when it is nonzero. This choice is made separately at each
momentum. It therefore supplies no phase continuity between neighbouring
momenta, and a zero `B[1,1,1]` leaves the phase undetermined.
"""
function get_B_tensor_list(states)
    n = length(states)
    B_list = Vector{Array{ComplexF64,3}}(undef, n)

    for i in 1:n
        VLs = states[i].VLs[1]
        Xs = states[i].Xs[1]

        B_tensor = VLs * Xs
        B_array = convert(Array, B_tensor)

        B_array_new = B_array[:, :, 1, :]

        θ = angle(B_array_new[1, 1, 1])
        B_array_new .*= exp(-im * θ)

        B_list[i] = B_array_new
    end

    return B_list
end

@doc raw"""
    create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)

Construct the finite MPS window for two incoming packets with central momenta
``+\kappa`` and ``-\kappa``. The first packet occupies sites `1:L÷2` and is
centred at `n_center`; the second occupies sites `L÷2+1:L` and is centred at
`L - n_center`. Their site-dependent excitation tensors are

```math
B_L(n)=B(+\kappa)e^{+i\kappa(n-n_L)-(n-n_L)^2/\sigma^2},
\qquad
B_R(n)=B(-\kappa)e^{-i\kappa(n-n_R)-(n-n_R)^2/\sigma^2}.
```

For the corresponding continuous Gaussian, `σ/√2` is the standard deviation
of the amplitude envelope and `σ/2` is that of its squared magnitude. The
sampled packet is also affected by lattice spacing and finite support. The
weights receive no separate normalization here; `main` normalizes the
complete `WindowMPS`.

The left- and right-canonical vacuum tensors are converted to dense arrays with
shape `(D, d, D)`. `two_particle_packet_tensors` closes the first excitation
before opening the second and joins them with the vacuum bond matrix `C⁻¹`, so
there is one excitation insertion in each half-window. The returned
`Vector{TensorMap}` has physical dimension `d` and virtual dimensions `D` or
`2D`, according to its position in the two packet constructions.

`n_center` must lie strictly inside the left half-window and `σ` must be
positive. `B_list[1]` and `B_list[2]` are assumed to correspond to `+κ` and
`-κ`, respectively.
"""
function create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)
    AL = ψ_gs.AL[]
    AR = ψ_gs.AR[]
    AL_array = convert(Array, AL)
    AR_array = convert(Array, AR)
    D, d, _ = size(AL_array)

    Lhalf = L ÷ 2
    1 < n_center < Lhalf ||
        throw(ArgumentError("n_center must lie strictly inside the left half-window"))
    σ > 0 || throw(ArgumentError("sigma must be positive"))

    left_packet = [
        B_list[1] * exp(im * (i - n_center) * κ - (i - n_center)^2 / σ^2)
        for i in 1:Lhalf
    ]
    right_center = L - n_center
    right_packet = [
        B_list[2] * exp(-im * (i - right_center) * κ - (i - right_center)^2 / σ^2)
        for i in (Lhalf + 1):L
    ]

    C = ψ_gs.C[]
    Cinv = convert(Array, C \ id(domain(C)))
    dense_window = two_particle_packet_tensors(
        AL_array, AR_array, Cinv, left_packet, right_packet
    )
    return [
        TensorMap(mat, ℂ^(size(mat, 1)) ⊗ ℂ^d ← ℂ^(size(mat, 3)))
        for mat in dense_window
    ]
end

@doc raw"""
    main(parsed_args)

Prepare two Gaussian quasiparticle packets and evolve their finite MPS window
with two-site TDVP. The vacuum is the one-site uniform MPS returned by
`prep_gs`. Tangent tensors are evaluated only at ``+\kappa`` and ``-\kappa``;
the same tensor at each central momentum is multiplied by the corresponding
position-space Gaussian across its half-window.

The window state is normalized once before evolution. `TDVP2` uses
`truncrank(D_evolution)`, where `D_evolution` is the supplied maximum rank or
`2D` when the option is zero. This is a rank cutoff rather than a discarded-
weight threshold.

`D`, `T`, and `dt` must be positive, `L` must be even, and an explicitly
supplied `D_evolution` must be at least `2D`. `create_stacked_tensor` also
checks the packet centre and Gaussian width. The two excitation energies
calculated at ``+\kappa`` and ``-\kappa`` are not saved.

For every stored time, the program evaluates the vacuum-subtracted symmetric
bond-energy density

```math
h_{n,n+1}=-\sigma_n^z\sigma_{n+1}^z
-\frac12(h_x\sigma_n^x+h_z\sigma_n^z)
-\frac12(h_x\sigma_{n+1}^x+h_z\sigma_{n+1}^z)
```

and ``\langle\sigma_n^z\rangle-\langle\sigma^z\rangle_{\rm vac}``. If `T` is
`total_time` and `L` is `length`, `energy_exp` and `s_z_exp` both have shape
`(T, L)`. Energy values occupy columns `1:L-1`; column `L` remains zero and is
omitted from the heatmap. All `L` columns of `s_z_exp` are filled. Row one is
the initial state at `t = 0`. Before row `r = 2, …, T` is measured, `timestep`
advances the state from `(r-2)dt` by `dt`, so `times[r] = (r-1)dt`.

The program saves two 600-dpi PNG heatmaps in `plots/` and two JLD2 files in
`data/`. The energy file stores `energy_exp` and `times`; the spin file stores
`s_z_exp` and `times`. File names contain the fields, packet parameters, window
length, bond dimensions, sample count, and time step. The program also prints
the input parameters, vacuum correlation length, vacuum energy, and the current
time-step index.

The saved heatmaps are local expectation values. They contain no projection
onto outgoing particle sectors. The program writes no separate series for the
state norm, summed-energy drift, discarded weight, or estimated arrival at a
window boundary.
"""
function main(parsed_args)
    D = parsed_args["bond_dimension"]
    D_evolution_arg = parsed_args["evolution_bond_dimension"]
    T = parsed_args["total_time"]
    dt = parsed_args["time_step"]
    σ = parsed_args["sigma"]
    L = parsed_args["length"]
    n_center = parsed_args["n_center"]
    κ = parsed_args["kappa"]
    h_z = parsed_args["h_z"]
    h_x = parsed_args["h_x"]
    output_root = abspath(parsed_args["output_dir"])

    D > 0 || throw(ArgumentError("bond_dimension must be positive"))
    T > 0 || throw(ArgumentError("total_time must be positive"))
    dt > 0 || throw(ArgumentError("time_step must be positive"))
    iseven(L) || throw(ArgumentError("length must be even"))
    D_evolution = iszero(D_evolution_arg) ? 2D : D_evolution_arg
    D_evolution >= 2D || throw(ArgumentError(
        "evolution_bond_dimension must be at least 2 * bond_dimension = $(2D)"
    ))

    println("Running simulation for the following set of parameters: ")
    for (arg, val) in parsed_args
        println("$arg = $val")
    end
    println("effective evolution_bond_dimension = $D_evolution")

    ham = get_ham(h_x, h_z)
    ψ_gs = prep_gs(D, ham)
    ξ = correlation_length(ψ_gs)
    println("Correlation length: $ξ")
    println("Ground state energy: $(expectation_value(ψ_gs, ham))")

    energies, states = get_QPstate(ψ_gs, ham, [κ, -κ])

    B_list = get_B_tensor_list(states)
    wavepacket_window = create_stacked_tensor(ψ_gs, B_list, L, n_center, κ, σ)
    Iden = TensorMap(Matrix{ComplexF64}(I, 2, 2), ℂ^2 ← ℂ^2)
    ψ_window = WindowMPS(ψ_gs, wavepacket_window)
    σ_x, σ_y, σ_z = get_ops()
    ham_density = -(σ_z ⊗ σ_z) - 0.5 * (h_x * σ_x + h_z * σ_z) ⊗ Iden - 0.5 * Iden ⊗ (h_x * σ_x + h_z * σ_z)

    normalize!(ψ_window)

    energy_exp = zeros(Float64, T, L)
    s_z_exp = zeros(Float64, T, L)

    gs_value_energy = [real(expectation_value(ψ_gs, (i, i + 1) => ham_density)) for i in 1:L-1]
    gs_value_s_z = [real(expectation_value(ψ_gs, i => σ_z)) for i in 1:L]

    for i in 1:L-1
        energy_exp[1, i] = real(expectation_value(ψ_window, (i, i + 1) => ham_density)) - gs_value_energy[i]
        s_z_exp[1, i] = real(expectation_value(ψ_window, i => σ_z)) - gs_value_s_z[i]
    end
    s_z_exp[1, L] = real(expectation_value(ψ_window, L => σ_z)) - gs_value_s_z[L]
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
        xlabel=L"Lattice bond $n$", ylabel=L"Lattice time $t$",
        title=L"$E - E_{vac}$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$"
    )
    sz_plot = heatmap(
        1:L, times, s_z_exp; dpi=600,
        xlabel=L"Lattice site $n$", ylabel=L"Lattice time $t$",
        title=L"$S_z - (S_z)_{vac}$ for $h_x$ = %$(h_x)$, $h_z$ = %$(h_z)$"
    )

    plot_dir = joinpath(output_root, "plots")
    data_dir = joinpath(output_root, "data")
    mkpath(plot_dir)
    mkpath(data_dir)

    filename = replace("scattering_infinite_mom_$(κ)_hx_$(h_x)_hz_$(h_z)_sigma_$(σ)_L_$(L)_n_$(n_center)_T_$(T)_D_$(D)_Dmax_$(D_evolution)_dt_$(dt)", '.' => 'p', '-' => 'm')
    energy_image_path = joinpath(plot_dir, "$(filename)_energy.png")
    sz_image_path = joinpath(plot_dir, "$(filename)_sz.png")

    savefig(energy_plot, energy_image_path)
    savefig(sz_plot, sz_image_path)
    println("Plot saved.")

    energy_path = joinpath(data_dir, "energy_$(filename).jld2")
    sz_path = joinpath(data_dir, "sz_value_$(filename).jld2")
    @save energy_path energy_exp times
    @save sz_path s_z_exp times
    println("Data saved.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_cmdline()
    main(parsed_args)
end
