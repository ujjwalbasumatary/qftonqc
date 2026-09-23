"""
    QFTSimulations

The Ising, Schwinger, and scalar-field programs use this module to construct
onsite operators, evaluate lattice dispersion relations, and assemble MPS
wave packets.

The oscillator functions give operator matrices in a finite set of number
states. The lattice formulas give the Ising scaling variable and the free
Ising-fermion and bosonized-Schwinger dispersions. For wave packets, the
module supplies momentum grids, Gaussian amplitudes, and MPS tensors with
one excitation insertion in each specified spatial region.

Momenta are dimensionless lattice momenta. Energies are expressed in the units
used by the Hamiltonian accompanying each function. MPS arrays are ordered as
`(left virtual index, physical index, right virtual index)`.
"""
module QFTSimulations

export harmonic_oscillator_matrices,
       ift_eta_latt,
       ift_free_fermion_dispersion,
       schwinger_free_lattice_dispersion,
       commensurate_momentum_grid,
       gaussian_weights,
       single_particle_packet_tensors,
       two_particle_packet_tensors

"""
    _ISING_MAGNETIC_EXPONENT

Magnetic scaling exponent `8/15` in
`eta_latt = (g_x - 1) / abs(g_z)^(8/15)` for the Ising field theory.
"""
const _ISING_MAGNETIC_EXPONENT = 8 / 15

"""
    _require_finite(value, name)

Return the real number `value` when it is finite. Throw
`DomainError(value, "\$name must be finite")` when `value` is `NaN`, `Inf`, or
`-Inf`. The dispersion and scaling-variable functions apply this check before
testing inequalities such as `gx >= 0` or `chi > 0`.
"""
function _require_finite(value::Real, name::AbstractString)
    isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

"""
    harmonic_oscillator_matrices(d)

Construct the onsite field operators in the truncated oscillator basis
`|0⟩, |1⟩, ..., |d-1⟩`. Before projection,

```math
\\phi = \\frac{a+a^\\dagger}{\\sqrt{2}}, \\qquad
\\pi = \\frac{i(a^\\dagger-a)}{\\sqrt{2}}, \\qquad [\\phi,\\pi]=i.
```

The result is the named tuple `(; phi, phi2, pi2, phi4)`. Each entry is a
`d × d` `Matrix{ComplexF64}`. With
`P_d = sum_{n=0}^{d-1} |n⟩⟨n|`, the four entries contain `P_d phi P_d`,
`P_d phi^2 P_d`, `P_d pi^2 P_d`, and `P_d phi^4 P_d`, respectively.

The powers of the field are formed in the infinite oscillator space before
projection. An intermediate application of the field can reach a state above
the cutoff and then return to a retained state. Taking powers of the
truncated `phi` matrix would omit that contribution, which affects matrix
elements near `|d-1⟩`.

The nonzero matrix elements have `Δn = ±1` for `phi`, `Δn = 0, ±2` for
`phi2` and `pi2`, and `Δn = 0, ±2, ±4` for `phi4`.

Throw `DomainError` when `d < 1`. Since `Bool <: Integer` in Julia, `false`
also reaches this error; `true` throws `ArgumentError` because a Boolean is not
accepted as a Hilbert-space dimension.
"""
function harmonic_oscillator_matrices(d::Integer)
    d >= 1 || throw(DomainError(d, "the oscillator cutoff d must be positive"))
    d isa Bool && throw(ArgumentError("the oscillator cutoff d must be an integer dimension, not Bool"))

    phi = zeros(ComplexF64, d, d)
    phi2 = zeros(ComplexF64, d, d)
    pi2 = zeros(ComplexF64, d, d)
    phi4 = zeros(ComplexF64, d, d)

    # phi = (a + a†) / sqrt(2).
    for i in 2:d
        value = sqrt((i - 1) / 2)
        phi[i, i - 1] = value
        phi[i - 1, i] = value
    end

    # Exact projected matrix elements of phi² and pi².
    for i in 1:d
        n = i - 1
        diagonal = n + 1 / 2
        phi2[i, i] = diagonal
        pi2[i, i] = diagonal

        if i + 2 <= d
            value = sqrt((n + 1) * (n + 2)) / 2
            phi2[i, i + 2] = value
            phi2[i + 2, i] = value
            pi2[i, i + 2] = -value
            pi2[i + 2, i] = -value
        end
    end

    # Exact projected matrix elements of phi⁴.  Only Δn = 0, ±2, ±4 occur.
    for i in 1:d
        n = i - 1
        phi4[i, i] = (6n^2 + 6n + 3) / 4

        if i + 2 <= d
            value = (4n + 6) * sqrt((n + 1) * (n + 2)) / 4
            phi4[i, i + 2] = value
            phi4[i + 2, i] = value
        end

        if i + 4 <= d
            value = sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4
            phi4[i, i + 4] = value
            phi4[i + 4, i] = value
        end
    end

    return (; phi, phi2, pi2, phi4)
end

"""
    ift_eta_latt(gx, gz)

Return the Ising-field-theory lattice scaling variable

```math
\\eta_{\\mathrm{latt}} = \\frac{g_x-1}{|g_z|^{8/15}},
```

using Eq. (4) of arXiv:2411.13645. Here `gx` is the transverse coupling and
`gz` is the longitudinal coupling, with the nearest-neighbour Ising coupling
set to one. The sign of `gz` does not enter this variable.

The inputs are promoted to a common floating type. For nonzero `gz`, the
`Float64` exponent can further promote the result; for example, `Float32`
inputs give a `Float64` result. For `gz == 0` and `gx != 1`, return `-Inf`
below the critical coupling and `Inf` above it in the promoted input type.
The point `(gx, gz) == (1, 0)` has the indeterminate form `0/0` and throws
`DomainError`.

Throw `DomainError` if either argument is not finite or if `gx < 0`.
"""
function ift_eta_latt(gx::Real, gz::Real)
    gx_float, gz_float = promote(float(gx), float(gz))
    _require_finite(gx_float, "gx")
    _require_finite(gz_float, "gz")
    gx_float >= 0 || throw(DomainError(gx, "gx = T/Tc must be nonnegative"))

    thermal_deformation = gx_float - one(gx_float)
    if iszero(gz_float)
        iszero(thermal_deformation) &&
            throw(DomainError((gx, gz), "eta_latt is undefined at the critical point (1, 0)"))
        infinity = typeof(gx_float)(Inf)
        return copysign(infinity, thermal_deformation)
    end

    return thermal_deformation / abs(gz_float)^_ISING_MAGNETIC_EXPONENT
end

"""
    ift_free_fermion_dispersion(k, gx)

Return the exact one-fermion energy at `gz = 0` for the Ising lattice
Hamiltonian used in arXiv:2411.13645. With the nearest-neighbour coupling set
to one, the dispersion is

```math
\\epsilon(k)=2\\sqrt{1+g_x^2-2g_x\\cos k}.
```

`k` is a dimensionless lattice momentum in radians and the result is periodic
under `k -> k + 2pi`. The implementation evaluates the identical expression

```math
2\\,\\operatorname{hypot}\\!\\left(1-g_x,\\,2\\sqrt{g_x}\\sin(k/2)\\right),
```

which does not form the subtraction `1 + gx^2 - 2gx*cos(k)` near
`gx = 1, k = 0`. The result has the floating type obtained by promoting `k`
and `gx`. Apply broadcasting, `ift_free_fermion_dispersion.(momenta, gx)`, to
evaluate an array of momenta.

Throw `DomainError` if either argument is not finite or if `gx < 0`.
"""
function ift_free_fermion_dispersion(k::Real, gx::Real)
    k_float, gx_float = promote(float(k), float(gx))
    _require_finite(k_float, "k")
    _require_finite(gx_float, "gx")
    gx_float >= 0 || throw(DomainError(gx, "gx must be nonnegative"))

    return 2 * hypot(one(gx_float) - gx_float,
                     2 * sqrt(gx_float) * sin(k_float / 2))
end

"""
    schwinger_free_lattice_dispersion(p, mu; chi=1, kappa=1)

Return the normal-mode energy of the quadratic (`lambda = 0`) bosonized
Schwinger lattice Hamiltonian

```math
H=\\frac{\\chi}{2}\\sum_x\\left[\\pi_x^2
 +\\kappa(\\phi_x-\\phi_{x-1})^2+\\mu^2\\phi_x^2\\right],
\\qquad
\\omega(p)=\\chi\\sqrt{\\mu^2+4\\kappa\\sin^2(p/2)}.
```

Equation (3) of arXiv:2307.02522 has `kappa = 1`. `mu` is the dimensionless
mass in this lattice Hamiltonian, `kappa` multiplies the nearest-neighbour
gradient term, and the positive factor `chi` fixes the energy scale. `p` is a
dimensionless lattice momentum in radians. The result has the floating type
obtained by promoting all four arguments. Apply broadcasting to evaluate an
array of momenta.

Throw `DomainError` when an argument is not finite, when `mu < 0`, when
`chi <= 0`, or when `kappa < 0`.
"""
function schwinger_free_lattice_dispersion(p::Real, mu::Real;
                                           chi::Real=1, kappa::Real=1)
    p_float, mu_float, chi_float, kappa_float =
        promote(float(p), float(mu), float(chi), float(kappa))
    _require_finite(p_float, "p")
    _require_finite(mu_float, "mu")
    _require_finite(chi_float, "chi")
    _require_finite(kappa_float, "kappa")
    mu_float >= 0 || throw(DomainError(mu, "mu must be nonnegative"))
    chi_float > 0 || throw(DomainError(chi, "chi must be positive"))
    kappa_float >= 0 || throw(DomainError(kappa, "kappa must be nonnegative"))

    return chi_float * hypot(mu_float,
                             2 * sqrt(kappa_float) * sin(p_float / 2))
end

"""
    commensurate_momentum_grid(n; period=2π)

Return the range of `n` momenta

```math
p_j=-\\frac{P}{2}+(j-1)\\frac{P}{n}, \\qquad j=1,\\ldots,n,
```

where `P = period`. The grid covers the half-open interval
`[-period/2, period/2)`, has spacing `period/n`, and satisfies
`p_n + period/n = p_1 + period`. For the default `period = 2π`,
`exp(im * p_j * n) = (-1)^n`. A Fourier sum over this grid repeats after
`n` sites for even `n` and changes sign for odd `n`; its squared magnitude
repeats in both cases. The collision programs use this sum on a finite
support inside an infinite vacuum, without imposing a periodic boundary
condition on the MPS window.

The return value is a one-dimensional Julia range of length `n`, so Julia does
not store all `n` entries separately. Throw `DomainError` if `n < 2`, if
`period <= 0`, or if `period` is not finite. Both Boolean values also throw
`DomainError`, since neither represents an allowed number of grid points.
"""
function commensurate_momentum_grid(n::Integer; period::Real=2π)
    n >= 2 || throw(DomainError(n, "a momentum grid needs at least two points"))
    n isa Bool && throw(ArgumentError("the number of momentum points must not be Bool"))
    period_float = float(period)
    _require_finite(period_float, "period")
    period_float > 0 || throw(DomainError(period, "period must be positive"))

    spacing = period_float / n
    return range(-period_float / 2; step=spacing, length=Int(n))
end

"""
    gaussian_weights(grid, center, sigma; period=2π, normalization=:l2)

Evaluate the real Gaussian amplitudes

```math
w_j=\\exp\\!\\left[-\\frac{\\delta(p_j,p_0)^2}{2\\sigma^2}\\right]
```

at the points `p_j = grid[j]`, with `p_0 = center`. For a numeric `period = P`,
momentum differences are wrapped into `[-P/2, P/2)`. The displacement used
in the Gaussian is therefore

```math
\\delta(p,p_0)=\\operatorname{mod}(p-p_0+P/2,P)-P/2.
```

Thus a packet centred at one edge of a Brillouin zone continues across the
other edge. Set `period=nothing` to use `delta(p,p_0) = p-p_0`. The parameter
`sigma` is the standard deviation of the amplitude Gaussian used in Eq. (S40)
of arXiv:2307.02522; the squared amplitudes have standard deviation
`sigma/sqrt(2)` in the continuous, unbounded Gaussian. The variance of the
sampled weights also depends on the grid spacing, range, and centre.

The returned real vector has one entry for each point in `grid`. The
`normalization` keyword determines how these weights are rescaled.

- `normalization=:l2` divides by `sqrt(sum(abs2, w))`, so
  `sum(abs2, weights) == 1` up to floating-point rounding;
- `normalization=:l1` divides by `sum(w)`, so `sum(weights) == 1` up to
  floating-point rounding;
- `normalization=:none` returns the formula above without rescaling.

For `:l1` and `:l2`, the function first subtracts the largest exponent from
every exponent, then evaluates the exponentials. The common rescaling
cancels during normalization. This keeps at least one sampled weight
nonzero for a narrow packet.

Throw `ArgumentError` if `grid` is empty or if `normalization` is not `:none`,
`:l1`, or `:l2`. Throw `DomainError` if `center`, `sigma`, a grid point, or a
numeric `period` is not finite; also throw `DomainError` when `sigma <= 0` or a
numeric `period <= 0`.
"""
function gaussian_weights(grid::AbstractVector{<:Real}, center::Real, sigma::Real;
                          period::Union{Nothing,Real}=2π,
                          normalization::Symbol=:l2)
    isempty(grid) && throw(ArgumentError("grid must contain at least one point"))
    center_float = float(center)
    sigma_float = float(sigma)
    _require_finite(center_float, "center")
    _require_finite(sigma_float, "sigma")
    sigma_float > 0 || throw(DomainError(sigma, "sigma must be positive"))
    normalization in (:none, :l1, :l2) ||
        throw(ArgumentError("normalization must be :none, :l1, or :l2"))

    period_float = if isnothing(period)
        nothing
    else
        value = float(period)
        _require_finite(value, "period")
        value > 0 || throw(DomainError(period, "period must be positive"))
        value
    end

    log_weights = map(grid) do point
        point_float = float(point)
        _require_finite(point_float, "each grid point")
        displacement = point_float - center_float
        if !isnothing(period_float)
            displacement = mod(displacement + period_float / 2, period_float) -
                           period_float / 2
        end
        -0.5 * (displacement / sigma_float)^2
    end

    normalization === :none && return exp.(log_weights)

    # Subtracting the largest exponent prevents an otherwise well-defined very
    # narrow packet from underflowing before it is normalized.
    weights = exp.(log_weights .- maximum(log_weights))
    scale = normalization === :l1 ? sum(weights) : sqrt(sum(abs2, weights))
    return weights ./ scale
end

"""
    _check_packet_inputs(AL, AR, packet)

Check that the vacuum and excitation tensors can be combined into a packet.
`AL` and `AR` must have the same shape `(D, d, D)`, and every tensor in
`packet` must have that shape as well. Here `D` is the vacuum MPS bond
dimension and `d` is the local Hilbert-space dimension. The three array
indices specify the left bond, physical state, and right bond, in that order.

Return `nothing` when the dimensions agree. Throw `ArgumentError` when
`packet` is empty. Throw `DimensionMismatch` when `AL` and `AR` have different
dimensions, when the left and right vacuum bond dimensions differ, or when any
packet tensor has dimensions different from `AL`.
"""
function _check_packet_inputs(AL::AbstractArray{<:Number,3},
                              AR::AbstractArray{<:Number,3},
                              packet::AbstractVector{<:AbstractArray{<:Number,3}})
    isempty(packet) && throw(ArgumentError("a packet must contain at least one tensor"))
    size(AL) == size(AR) ||
        throw(DimensionMismatch("AL and AR must have identical dimensions"))
    Dleft, _, Dright = size(AL)
    Dleft == Dright ||
        throw(DimensionMismatch("vacuum tensors must have equal left and right bond dimensions"))
    all(B -> size(B) == size(AL), packet) ||
        throw(DimensionMismatch("every excitation tensor must have the same dimensions as AL and AR"))
    return nothing
end

"""
    _block_upper(AL, AR, B)

Form the upper-triangular MPS tensor

```math
M^s=\\begin{pmatrix}A_L^s & B^s\\\\ 0 & A_R^s\\end{pmatrix}
```

from arrays `AL`, `AR`, and `B` of shape `(D, d, D)`. The upper and lower
blocks keep track of whether the `B` tensor has been inserted. A product can
pass from the upper block to the lower block through `B`; the zero block
prevents it from returning. The returned dense array has shape `(2D, d, 2D)`
and element type
`promote_type(eltype(AL), eltype(AR), eltype(B))`. None of the inputs is
modified.

The block formula assumes that `_check_packet_inputs` has already checked the
three shapes.
"""
function _block_upper(AL::AbstractArray{<:Number,3},
                      AR::AbstractArray{<:Number,3},
                      B::AbstractArray{<:Number,3})
    D, d, _ = size(AL)
    T = promote_type(eltype(AL), eltype(AR), eltype(B))
    block = zeros(T, 2D, d, 2D)
    block[1:D, :, 1:D] .= AL
    block[1:D, :, (D + 1):(2D)] .= B
    block[(D + 1):(2D), :, (D + 1):(2D)] .= AR
    return block
end

"""
    single_particle_packet_tensors(AL, AR, packet)

Construct the dense MPS tensors for a single tangent-space excitation that
can be inserted on any of `N = length(packet)` sites. `AL` and `AR` are the
left- and right-canonical vacuum tensors of shape `(D, d, D)`. At site `n`,
`packet[n]` gives the excitation tensor with its envelope and phase already
included. It has the same shape as the vacuum tensors.

Contracting the returned tensors gives the sum

```math
\\sum_{n=1}^{N} A_L^{[1]}\\cdots A_L^{[n-1]}
B_n^{[n]} A_R^{[n+1]}\\cdots A_R^{[N]}.
```

There is one `B_n` insertion in every term. No normalization is applied to
this state.

For `N > 1`, the first tensor has shape `(D, d, 2D)`, every interior tensor has
shape `(2D, d, 2D)`, and the last has shape `(2D, d, D)`. For `N == 1`, return
a one-element vector containing a copy of `packet[1]`, with shape `(D, d, D)`.
The array order is `(left bond, physical index, right bond)`.

For `N > 1`, the vector element type is `Array{T,3}`, where
`T = promote_type(eltype(AL), eltype(AR), eltype(packet[1]))`. For `N == 1`,
the copied packet tensor determines the element type. Throw `ArgumentError`
when `packet` is empty. Throw `DimensionMismatch` when `AL` and `AR` have
different dimensions, when their left and right bond dimensions differ, or
when a packet tensor does not have the vacuum-tensor dimensions.
"""
function single_particle_packet_tensors(
        AL::AbstractArray{<:Number,3}, AR::AbstractArray{<:Number,3},
        packet::AbstractVector{<:AbstractArray{<:Number,3}})
    _check_packet_inputs(AL, AR, packet)
    length(packet) == 1 && return [copy(only(packet))]

    T = promote_type(eltype(AL), eltype(AR), eltype(first(packet)))
    tensors = Vector{Array{T,3}}(undef, length(packet))
    tensors[1] = cat(AL, packet[1]; dims=3)
    for site in 2:(length(packet) - 1)
        tensors[site] = _block_upper(AL, AR, packet[site])
    end
    tensors[end] = cat(packet[end], AR; dims=1)
    return tensors
end

"""
    _right_multiply_bond(tensor, matrix)

Contract `matrix` into the right virtual index of `tensor`. For a tensor with
shape `(Dleft, d, Dright)`, the returned array has the same shape and entries

```math
T'_{a s b}=\\sum_{c=1}^{D_{\\mathrm{right}}}T_{a s c}M_{c b}.
```

The input arrays are not modified. Throw `DimensionMismatch` unless `matrix`
has shape `(Dright, Dright)`.
"""
function _right_multiply_bond(tensor::AbstractArray{<:Number,3},
                              matrix::AbstractMatrix{<:Number})
    Dleft, d, Dright = size(tensor)
    size(matrix) == (Dright, Dright) ||
        throw(DimensionMismatch("bond-gauge matrix must be $Dright × $Dright"))
    product = reshape(tensor, Dleft * d, Dright) * matrix
    return reshape(product, Dleft, d, Dright)
end

"""
    two_particle_packet_tensors(AL, AR, Cinv, left_packet, right_packet)

Construct dense MPS tensors with one tangent-space excitation in
`left_packet` and one in `right_packet`. The two vectors describe consecutive,
non-overlapping spatial regions. Their entries already contain the desired
envelopes and phases.

Each region is formed with [`single_particle_packet_tensors`](@ref), so every
term in the contracted state contains one tensor from `left_packet` and one
from `right_packet`. No normalization or orthogonalization is applied.

`AL` and `AR` are the left- and right-canonical vacuum tensors with shape
`(D, d, D)`. `Cinv` has shape `(D, D)` and represents the inverse centre matrix
`C^{-1}` of the canonical uniform MPS. It is contracted into the right bond of
the final tensor in the left region before the right region is appended. This
joins the right-canonical vacuum segment of the first packet to the
left-canonical vacuum segment of the second packet, as in Sec. III B of the
supplement to arXiv:2307.02522.

The result contains `length(left_packet) + length(right_packet)` arrays in
`(left bond, physical index, right bond)` order. Its outer bonds have dimension
`D`; the bond joining the two regions also has dimension `D`; and a packet with
more than one site has internal bond dimension `2D`.

Throw `ArgumentError` if either packet is empty. Throw `DimensionMismatch` if
`AL` and `AR` have different dimensions, if their left and right bond
dimensions differ, if a packet tensor does not have shape `(D, d, D)`, or if
`Cinv` does not have shape `(D, D)`.
"""
function two_particle_packet_tensors(
        AL::AbstractArray{<:Number,3}, AR::AbstractArray{<:Number,3},
        Cinv::AbstractMatrix{<:Number},
        left_packet::AbstractVector{<:AbstractArray{<:Number,3}},
        right_packet::AbstractVector{<:AbstractArray{<:Number,3}})
    _check_packet_inputs(AL, AR, left_packet)
    _check_packet_inputs(AL, AR, right_packet)
    D = size(AL, 1)
    size(Cinv) == (D, D) ||
        throw(DimensionMismatch("Cinv must have the vacuum bond dimensions $D × $D"))

    left_tensors = single_particle_packet_tensors(AL, AR, left_packet)
    right_tensors = single_particle_packet_tensors(AL, AR, right_packet)
    left_tensors[end] = _right_multiply_bond(left_tensors[end], Cinv)
    return vcat(left_tensors, right_tensors)
end

end # module QFTSimulations
