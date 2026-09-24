using MPSKit, TensorKit, Plots, JLD2, LaTeXStrings, ArgParse, LinearAlgebra
using QFTSimulations: commensurate_momentum_grid, two_particle_packet_tensors
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

`delta_p` is a requested momentum spacing. `main` replaces it by the
commensurate value `2π/N`, with `N = round(Int, 2π/delta_p)`. `mom` gives the
central momentum of the left packet; the right packet uses `-mom`. The
momentum-space amplitude is proportional to `exp(-δp^2/sigma^2)`. `total_time`
is the number of stored rows, including the initial state, and the last stored
time is `(total_time - 1)time_step`. `evolution_bond_dimension = 0` asks `main`
to use twice the vacuum bond dimension.

Calling this function with `--help` prints the option list and exits through
ArgParse.
"""
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
    get_ham(J, h_x, h_z)

Construct the Hamiltonian as a one-site uniform matrix product operator (MPO),

```math
H=-\sum_j\left(J\sigma_j^z\sigma_{j+1}^z
  +h_x\sigma_j^x+h_z\sigma_j^z\right).
```

where the lattice spacing is one. In the command-line calculation, `J`, `h_x`,
and `h_z` are real dimensionless couplings.
"""
function get_ham(J, h_x, h_z)
    σ_x, σ_y, σ_z = get_ops()
    ham = InfiniteMPOHamiltonian(PeriodicVector([ℂ^2]), 1 => -h_x * σ_x - h_z * σ_z, (1, 2) => -J * σ_z ⊗ σ_z)
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
function does not identify or track a particle species across the Brillouin
zone.
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
momenta, and a zero `B[1,1,1]` leaves the phase undetermined. The Fourier sum in
`create_B_packet` inherits these independent phase choices.
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
    create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)

Fourier-sum the excitation tensors into the tensor inserted at site `n`:

```math
B_n=\sum_{i=1}^{N}
e^{-\delta p_i^2/\sigma^2}
e^{ip_i(n-n_0)}B(p_i),
\qquad
p_i=-\pi+(i-1)\Delta p.
```

Here `N = length(B_tensor_list)`, `n_0 = offset`, and `mom_idx` labels the
central grid point. The displacement
`δp_i = mod((i-mom_idx)Δp + π, 2π) - π` uses the shortest separation on the
Brillouin zone. The formula assumes the tensors follow the commensurate grid
with `N*Δp = 2π`.

For the corresponding continuous, unbounded Gaussian, `sigma/√2` is the
standard deviation of its amplitude envelope and `sigma/2` is that of its
squared magnitude. The variance on the sampled Brillouin zone also depends on
the spacing and periodic wrapping. The sum contains neither a factor of `Δp`
nor a separate normalization. `main`
normalizes the complete `WindowMPS` after both packets have been assembled.
The return value is a dense complex array with the same three dimensions as
one entry of `B_tensor_list`. A positive, nonzero `sigma` is expected; this
function does not check it.
"""
function create_B_packet(B_tensor_list, n, offset, mom_idx, Δp, sigma)
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

"""
    nearest_momentum_index(momentum, Δp, n_momenta)

Return the one-based index of the commensurate grid point nearest `momentum`.
The momentum is first wrapped to `[-π, π)`, and index one denotes `-π`.
Rounding uses Julia's `round(Int, ...)`; the result is wrapped modulo
`n_momenta`, so `π` and `-π` select the same point.

The caller is responsible for supplying `Δp = 2π/n_momenta` and a positive
number of grid points.
"""
function nearest_momentum_index(momentum, Δp, n_momenta)
    wrapped_momentum = mod(momentum + π, 2π) - π
    return mod(round(Int, (wrapped_momentum + π) / Δp), n_momenta) + 1
end

"""
    create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)

Construct an ordered two-particle window from two packet supports of length
`L`. The left- and right-canonical vacuum tensors are converted to dense arrays
with shape `(D, d, D)`. Each packet entry must have the same shape, and both
lists must contain exactly `L` entries.

`two_particle_packet_tensors` closes the first excitation before opening the
second and joins them with the vacuum bond matrix `C⁻¹`. The returned
`Vector{TensorMap}` therefore contains `2L` sites, with one excitation insertion
in each `L`-site support. Its physical dimension is `d`; its virtual dimensions
are `D` or `2D`, according to position within the two packet constructions.
"""
function create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L)
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

    return [
        TensorMap(mat, ℂ^(size(mat, 1)) ⊗ ℂ^d ← ℂ^(size(mat, 3)))
        for mat in dense_window
    ]
end

@doc raw"""
    main()

Build two momentum-space Gaussian packets and evolve their finite MPS window
with two-site TDVP. The requested spacing `delta_p` first determines
``N=\operatorname{round}(2\pi/\mathtt{delta\_p})``. The actual momentum grid is
the half-open interval ``[-\pi,\pi)`` with spacing ``\Delta p=2\pi/N``. Each
packet has an `N`-site support, and the concatenated window has `2N` sites. The
two central momenta are the grid points nearest `mom` and `-mom`; both packets
are centred at site `N÷2` within their respective supports.

The vacuum is the one-site uniform MPS returned by `prep_gs`. A tangent tensor
is calculated at every grid momentum and Fourier-summed by `create_B_packet`.
Since `get_B_tensor_list` fixes each tensor's phase independently, neighbouring
momenta need not have a continuous phase convention. Such phase changes enter
the Fourier sum and can move or deform the position-space packet. The
excitation energies returned with the tensors are not used or written.

The assembled window state is normalized once before evolution. `TDVP2` uses
`truncrank(D_evolution)`, where `D_evolution` is the supplied maximum rank or
`2D` when the option is zero. This is a rank cutoff rather than a discarded-
weight threshold.

`D`, `T`, `dt`, and the requested momentum spacing must be positive; `mom` must
be finite; the rounded grid must contain at least two momenta; and an explicitly
supplied `D_evolution` must be at least `2D`. The program does not check
`sigma`, although `create_B_packet` expects a positive, nonzero width.

The symmetric bond-energy operator is

```math
h_{n,n+1}=-J\sigma_n^z\sigma_{n+1}^z
-\frac12(h_x\sigma_n^x+h_z\sigma_n^z)
-\frac12(h_x\sigma_{n+1}^x+h_z\sigma_{n+1}^z)
```

For every stored time, the program measures its expectation value, subtracts
the expectation value in the vacuum, and saves the difference. It also saves
``\langle\sigma_n^z\rangle-\langle\sigma^z\rangle_{\rm vac}``. If `T` is
`total_time` and `L = 2N`, `energy_exp` and `s_z_exp` both have shape `(T, L)`.
Energy values occupy columns `1:L-1`; column `L` remains zero and is omitted
from the heatmap. All `L` columns of `s_z_exp` are filled. Row one is the
initial state at `t = 0`. Before row `r = 2, …, T` is measured, `timestep`
advances the state from `(r-2)dt` by `dt`, so `times[r] = (r-1)dt`.

The program saves two 600-dpi PNG heatmaps in `plots/` and two JLD2 files in
`data/`. The energy file stores `energy_exp` and `times`; the spin file stores
`s_z_exp` and `times`. File names contain the coupling, fields, central
momentum, actual grid spacing, Gaussian width, bond dimensions, sample count,
and time step. The program also prints the input parameters, actual momentum
spacing, vacuum correlation length, vacuum energy, and current time-step index.

The saved heatmaps are local expectation values. They contain no projection
onto outgoing particle sectors. The program writes no separate series for the
state norm, summed-energy drift, discarded weight, or estimated arrival at a
window boundary.
"""
function main()
    parsed_args = parse_cmdline()
    D = parsed_args["bond_dimension"]
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
    offset_left = L_sites ÷ 2
    offset_right = L_sites ÷ 2

    mom_idx_left = nearest_momentum_index(mom, Δp, n_momenta)
    mom_idx_right = nearest_momentum_index(-mom, Δp, n_momenta)
    B_tensor_list = get_B_tensor_list(states)
    B_packet_list_left = [create_B_packet(B_tensor_list, n, offset_left, mom_idx_left, Δp, σ) for n in 1:L_sites]
    B_packet_list_right = [create_B_packet(B_tensor_list, n, offset_right, mom_idx_right, Δp, σ) for n in 1:L_sites]
    wavepacket_window = create_stacked_tensor(ψ_gs, B_packet_list_left, B_packet_list_right, L_sites)
    L, = size(wavepacket_window)
    Iden = TensorMap(Matrix{ComplexF64}(I, 2, 2), ℂ^2 ← ℂ^2)
    ψ_window = WindowMPS(ψ_gs, wavepacket_window)
    σ_x, σ_y, σ_z = get_ops()
    ham_density = -(J * σ_z ⊗ σ_z) - 0.5 * (h_x * σ_x + h_z * σ_z) ⊗ Iden - 0.5 * Iden ⊗ (h_x * σ_x + h_z * σ_z)

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
