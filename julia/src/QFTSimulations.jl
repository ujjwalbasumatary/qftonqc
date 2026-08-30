module QFTSimulations

export harmonic_oscillator_matrices,
       ift_eta_latt,
       ift_free_fermion_dispersion,
       schwinger_free_lattice_dispersion,
       commensurate_momentum_grid,
       gaussian_weights,
       single_particle_packet_tensors,
       two_particle_packet_tensors

const _ISING_MAGNETIC_EXPONENT = 8 / 15

function _require_finite(value::Real, name::AbstractString)
    isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

"""
    harmonic_oscillator_matrices(d)

Return `(phi, phi2, pi2, phi4)` as `d × d` `ComplexF64` matrices in the
number basis `|0⟩, …, |d-1⟩`, with `[phi, pi] = i` before truncation.

`phi2`, `pi2`, and `phi4` are the matrix elements of the corresponding
infinite-dimensional operators projected into the truncated space.  In
particular, they are intentionally not formed by multiplying the truncated
`phi` matrix: such a product loses virtual transitions through states above
the cutoff.
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

Compute the Ising-field-theory lattice scaling variable

```
η_latt = (gx - 1) / |gz|^(8/15)
```

from Eq. (4) of arXiv:2411.13645.  At `gz == 0`, the signed free-fermion
limit is returned as an infinity.  The critical point `(gx, gz) == (1, 0)`
is undefined and raises `DomainError`.
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

Exact one-particle lattice dispersion at `gz = 0` for the Ising Hamiltonian
used in arXiv:2411.13645 (nearest-neighbour coupling set to one):

```
ϵ(k) = 2 sqrt(1 + gx^2 - 2 gx cos(k)).
```

The algebraically equivalent implementation below is stable close to the
critical point `gx = 1`, `k = 0`.  Use broadcasting for a grid of momenta.
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

Dispersion of the quadratic (`lambda = 0`) bosonized Schwinger lattice
Hamiltonian

```
H = chi/2 ∑x [pi_x^2 + kappa (phi_x - phi_(x-1))^2 + mu^2 phi_x^2],
ω(p) = chi sqrt(mu^2 + 4 kappa sin(p/2)^2).
```

Equation (3) of arXiv:2307.02522 has `kappa = 1`; the keyword also covers the
gradient convention in the repository's exploratory script.  Use broadcasting
for a grid of momenta.
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

Return `n` equally spaced momenta on the half-open Brillouin zone
`[-period/2, period/2)`.  The spacing is exactly `period/n`, so the Fourier
transform is periodic after `n` lattice sites.  This avoids the unmatched wrap
spacing produced by a range such as `-π:0.1:π-0.1`.
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

Construct Gaussian wave-packet amplitudes
`exp(-(p-center)^2 / (2 sigma^2))` on `grid`.  With a numeric `period`,
distances use the shortest periodic separation; pass `period=nothing` for an
ordinary nonperiodic Gaussian.

`normalization=:l2` (the default) makes `sum(abs2, weights) == 1`, appropriate
for quantum amplitudes.  `:l1` makes `sum(weights) == 1`, while `:none` returns
the unnormalized Gaussian.  `sigma` is the standard deviation of the amplitude
Gaussian, matching Eq. (S40) of arXiv:2307.02522.
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

Construct the dense site tensors for exactly one quasiparticle wave packet.
`packet[n]` is the already weighted excitation tensor at site `n`. The virtual
automaton opens from bond dimension `D` to `2D`, permits one and only one
off-diagonal `B` insertion, and closes back to `D`.

The returned arrays use MPSKit's `(left bond, physical, right bond)` ordering.
They can be converted to `TensorMap`s by the simulation layer.
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

Construct an ordered two-particle MPS from disjoint packet supports. Each
packet is independently closed, guaranteeing one excitation in each region.
The first close tensor is right-multiplied by the vacuum gauge matrix `C⁻¹`
before the second packet is reopened; for a canonical uMPS this is the glue
described in Sec. III B of the supplement to arXiv:2307.02522.

A single upper-triangular block spanning both packet regions is not equivalent:
it permits only one `B` insertion and therefore represents a one-particle
superposition. This routine deliberately exposes a `D`-dimensional bond between
the independently closed packets. Grow that bond before one-site TDVP if the
collision calculation needs a larger evolution bond.
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
