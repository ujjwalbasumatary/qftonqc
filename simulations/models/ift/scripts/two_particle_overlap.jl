"""
    IFTTwoParticleOverlap

Contract a finite MPS with pairs of localized excitation tensors above a
uniform vacuum. The functions use ordinary arrays with indices ordered as
`(left bond, physical index, right bond)`; no tensor-network package is
needed here. The infinite vacuum to either side of the window supplies
identity boundary contractions in its left and right canonical gauges.
"""
module IFTTwoParticleOverlap

using LinearAlgebra

export two_particle_overlaps, localized_pair_norms, pair_basis_grams, two_particle_weight

function _finite_array(value, dimensions, name)
    value isa AbstractArray || throw(ArgumentError("$name must be an array"))
    ndims(value) == dimensions || throw(ArgumentError("$name must have $dimensions indices"))
    all(>(0), size(value)) || throw(ArgumentError("$name cannot have an empty index"))
    all(isfinite, value) || throw(ArgumentError("$name must contain only finite entries"))
    return Array{ComplexF64}(value)
end

function _reference_tensors(AL, Cinv, BL, BR; AR=nothing)
    left = _finite_array(AL, 3, "AL")
    inverse_center = _finite_array(Cinv, 2, "Cinv")
    first = _finite_array(BL, 3, "BL")
    second = _finite_array(BR, 3, "BR")
    D, d, right_dimension = size(left)
    D == right_dimension || throw(ArgumentError("AL must have equal left and right bond dimensions"))
    size(inverse_center) == (D, D) || throw(ArgumentError("Cinv must have size ($D, $D)"))
    size(first) == size(left) || throw(ArgumentError("BL must have the same dimensions as AL"))
    size(second) == size(left) || throw(ArgumentError("BR must have the same dimensions as AL"))
    right = AR === nothing ? nothing : _finite_array(AR, 3, "AR")
    right === nothing || size(right) == size(left) ||
        throw(ArgumentError("AR must have the same dimensions as AL"))
    inserted_first = similar(first)
    for s in 1:d
        mul!(@view(inserted_first[:, s, :]), @view(first[:, s, :]), inverse_center)
    end
    all(isfinite, inserted_first) || throw(ArgumentError("BL * Cinv has nonfinite entries"))
    return left, right, inserted_first, second
end

function _left_transfer(environment, reference, ket)
    result = zeros(ComplexF64, size(reference, 3), size(ket, 3))
    for s in axes(reference, 2)
        result .+= @view(reference[:, s, :])' * environment * @view(ket[:, s, :])
    end
    return result
end

function _right_transfer(environment, reference, ket)
    result = zeros(ComplexF64, size(ket, 1), size(reference, 1))
    for s in axes(reference, 2)
        result .+= @view(ket[:, s, :]) * environment * @view(reference[:, s, :])'
    end
    return result
end

# Both environments already include the complex conjugation of the bra.
# dot(left, transpose(right)) would conjugate the bra a second time.
function _close_environments(left, right)
    value = zero(ComplexF64)
    for j in axes(left, 2), i in axes(left, 1)
        value += left[i, j] * right[j, i]
    end
    return value
end

"""
    two_particle_overlaps(tensors, AL, AR, Cinv, BL, BR;
                          minimum_separation=1)

Return the complex matrix `O[n,m] = ⟨n,BL; m,BR | ψ⟩` for `m > n`.
`tensors` represents `ψ` as one finite MPS: for a mixed-canonical MPS with
its center at the first site, supply `AC[1], AR[2], …, AR[L]`. Its norm and
overall complex phase are retained. Every tensor has indices ordered as
`(left bond, physical index, right bond)`. Internal ket bonds may have
different dimensions; the two external bonds must match the vacuum bonds.

The reference state for each pair contains `AL` before site `n`,
`BL * Cinv` at `n`, `AL` between the insertions, `BR` at site `m`, and
`AR` after `m`. The multiplication by `Cinv` acts on the right bond of
`BL`. Here `Cinv` is the inverse vacuum center matrix, in the same gauges
as all four tensors. The exterior vacuum contractions are identities.
Thus the ket and the reference must have the same infinite boundaries
expressed in those gauges; this function does not align different vacua.

Only pairs with `m-n ≥ minimum_separation` are calculated. Other entries,
including the diagonal and lower triangle, are zero. The separation must
be a positive integer; a separation at least `L` gives a zero matrix.

Left and right contractions outside each pair are reused. The work grows
as `L²` at fixed bond dimensions, without constructing one MPS per pair.
The result is a set of overlaps, not by itself a probability. Summing their
squared magnitudes gives a projection only for an orthogonal reference
basis. In the construction above, left-canonical `AL`, right-canonical
`AR`, and the excitation gauge conditions `Σₛ ALₛ† BLₛ = 0` and
`Σₛ BRₛ ARₛ† = 0` make distinct insertion positions orthogonal, with
the same boundary states assumed throughout. These conditions must be
checked where the excitation tensors are prepared.

Keeping one excitation tensor at a fixed momentum approximates a branch
over a narrow range of momenta. This function neither supplies the missing
momentum dependence nor includes other particle species or particle counts.
"""
function two_particle_overlaps(tensors, AL, AR, Cinv, BL, BR;
                               minimum_separation::Integer=1)
    minimum_separation >= 1 || throw(ArgumentError("minimum_separation must be positive"))
    left, right, first, second = _reference_tensors(AL, Cinv, BL, BR; AR)
    L = length(tensors)
    L >= 1 || throw(ArgumentError("tensors must contain at least one site"))
    ket = [_finite_array(tensor, 3, "tensors[$i]") for (i, tensor) in enumerate(tensors)]
    D, d, _ = size(left)
    for i in 1:L
        size(ket[i], 2) == d || throw(ArgumentError("tensors[$i] has a different physical dimension from AL"))
        if i < L
            size(ket[i], 3) == size(ket[i+1], 1) ||
                throw(ArgumentError("ket bond dimensions disagree between sites $i and $(i+1)"))
        end
    end
    size(ket[1], 1) == D || throw(ArgumentError("the left external ket bond must match AL"))
    size(ket[end], 3) == D || throw(ArgumentError("the right external ket bond must match AR"))
    overlaps = zeros(ComplexF64, L, L)
    minimum_separation >= L && return overlaps

    left_before = Vector{Matrix{ComplexF64}}(undef, L)
    left_before[1] = Matrix{ComplexF64}(I, D, D)
    for n in 2:L
        left_before[n] = _left_transfer(left_before[n-1], left, ket[n-1])
    end

    # right_at[m] includes BR at m and AR on the sites to its right.
    right_at = Vector{Matrix{ComplexF64}}(undef, L)
    right_after = Matrix{ComplexF64}(I, D, D)
    for m in L:-1:1
        right_at[m] = _right_transfer(right_after, second, ket[m])
        m > 1 && (right_after = _right_transfer(right_after, right, ket[m]))
    end

    for n in 1:(L-minimum_separation)
        environment = _left_transfer(left_before[n], first, ket[n])
        for m in (n+1):L
            if m-n >= minimum_separation
                overlaps[n, m] = _close_environments(environment, right_at[m])
            end
            m < L && (environment = _left_transfer(environment, left, ket[m]))
        end
    end
    return overlaps
end

"""
    localized_pair_norms(AL, Cinv, BL, BR, max_separation)

Calculate the squared norms of the localized pair states used by
[`two_particle_overlaps`](@ref). Entry `norms[r]` is the squared norm for
insertions `r` lattice sites apart, with `r-1` copies of `AL` between them.
The left insertion is `BL * Cinv` and the right insertion is `BR`.

The contractions outside the pair are identities, as supplied by the
left- and right-canonical vacuum. Starting from
`E = Σₛ (BLₛ Cinv)† (BLₛ Cinv)` and `R = Σₛ BRₛ BRₛ†`, the first norm
is `tr(E R)`. Each increase of the separation replaces
`E` by `Σₛ ALₛ† E ALₛ`. This retains any dependence of the norm on the
distance between the insertions instead of assuming unit norm.

`max_separation` must be a positive integer. Every returned norm must be
finite and strictly positive; a zero excitation or a pair with zero norm
throws `ArgumentError`. No normalization is applied to the supplied tensors.
"""
function localized_pair_norms(AL, Cinv, BL, BR, max_separation::Integer)
    max_separation >= 1 || throw(ArgumentError("max_separation must be positive"))
    left, _, first, second = _reference_tensors(AL, Cinv, BL, BR)
    D, d, _ = size(left)
    environment = zeros(ComplexF64, D, D)
    right = zeros(ComplexF64, D, D)
    for s in 1:d
        environment .+= @view(first[:, s, :])' * @view(first[:, s, :])
        right .+= @view(second[:, s, :]) * @view(second[:, s, :])'
    end
    norms = Vector{Float64}(undef, max_separation)
    for r in 1:max_separation
        norms[r] = real(_close_environments(environment, right))
        isfinite(norms[r]) && norms[r] > 0 ||
            throw(ArgumentError("the pair at separation $r has a nonpositive or nonfinite squared norm"))
        r < max_separation && (environment = _left_transfer(environment, left, left))
    end
    return norms
end

"""
    pair_basis_grams(AL, Cinv, left_basis, right_basis, max_separation)

Calculate the Gram matrices of several excitation pairs at fixed insertion
positions. `left_basis[a]` and `right_basis[b]` are excitation tensors with
indices `(left bond, physical index, right bond)`, each with the same
dimensions as `AL`. The state labeled `(a,b)` has `left_basis[a] * Cinv`
at the first insertion, `right_basis[b]` at the second, and `AL` between
them. As in [`localized_pair_norms`](@ref), the contractions outside the
pair are identities.

The result is a vector of complex matrices. Entry `grams[r]` is the Gram
matrix when the insertions are `r` lattice sites apart. If the basis sizes
are `KL` and `KR`, its rows and columns use the ordering
`index(a,b) = a + (b-1)*KL`, so that
`grams[r][index(a,b), index(c,d)] = ⟨a,b; r | c,d; r⟩`.
In particular, its diagonal reproduces the norms calculated separately by
[`localized_pair_norms`](@ref).

The calculation starts from
`E[a,c] = Σₛ (BL[a]ₛ Cinv)† (BL[c]ₛ Cinv)` and
`Q[d,b] = Σₛ BR[d]ₛ BR[b]ₛ†`. For adjacent insertions the matrix entry
is `tr(E[a,c] Q[d,b])`. Each increase of the separation propagates every
`E[a,c]` through one copy of `AL`. This retains the finite-separation
overlaps, which need not factor into overlaps of isolated particles.

Both basis collections must be nonempty and `max_separation` must be
positive. Linearly dependent and zero basis tensors are allowed: a Gram
matrix is positive semidefinite, not necessarily invertible. The function
returns the direct contractions without clipping eigenvalues, imposing
unit diagonal entries, or orthogonalizing the supplied basis. Hermiticity
and positive semidefiniteness hold up to floating-point roundoff. A matrix
different from the identity must be retained when projecting onto this
basis; summing squared overlaps alone would then give the wrong weight.

These matrices compare different excitation tensors at the same two
positions. They do not test orthogonality between different positions;
that also requires the canonical and excitation gauge conditions stated
in [`two_particle_overlaps`](@ref).
"""
function pair_basis_grams(AL, Cinv, left_basis, right_basis, max_separation::Integer)
    max_separation >= 1 || throw(ArgumentError("max_separation must be positive"))
    isempty(left_basis) && throw(ArgumentError("left_basis must contain at least one tensor"))
    isempty(right_basis) && throw(ArgumentError("right_basis must contain at least one tensor"))
    left, _, first_tensor, second_tensor =
        _reference_tensors(AL, Cinv, first(left_basis), first(right_basis))
    first_tensors = [first_tensor]
    second_tensors = [second_tensor]
    for tensor in Iterators.drop(left_basis, 1)
        _, _, transformed, _ = _reference_tensors(left, Cinv, tensor, second_tensor)
        push!(first_tensors, transformed)
    end
    for tensor in Iterators.drop(right_basis, 1)
        _, _, _, checked = _reference_tensors(left, Cinv, first(left_basis), tensor)
        push!(second_tensors, checked)
    end

    KL, KR = length(first_tensors), length(second_tensors)
    D, d, _ = size(left)
    environments = Matrix{Matrix{ComplexF64}}(undef, KL, KL)
    for a in 1:KL, c in 1:KL
        environment = zeros(ComplexF64, D, D)
        for s in 1:d
            environment .+= @view(first_tensors[a][:, s, :])' * @view(first_tensors[c][:, s, :])
        end
        environments[a, c] = environment
    end
    right_contractions = Matrix{Matrix{ComplexF64}}(undef, KR, KR)
    for b in 1:KR, e in 1:KR
        contraction = zeros(ComplexF64, D, D)
        for s in 1:d
            contraction .+= @view(second_tensors[e][:, s, :]) * @view(second_tensors[b][:, s, :])'
        end
        right_contractions[e, b] = contraction
    end

    grams = Vector{Matrix{ComplexF64}}(undef, max_separation)
    for r in 1:max_separation
        gram = Matrix{ComplexF64}(undef, KL*KR, KL*KR)
        for a in 1:KL, b in 1:KR, c in 1:KL, e in 1:KR
            gram[a+(b-1)*KL, c+(e-1)*KL] =
                _close_environments(environments[a, c], right_contractions[e, b])
        end
        all(isfinite, gram) || throw(ArgumentError("the Gram matrix at separation $r has nonfinite entries"))
        grams[r] = gram
        if r < max_separation
            for a in 1:KL, c in 1:KL
                environments[a, c] = _left_transfer(environments[a, c], left, left)
            end
        end
    end
    return grams
end

"""
    two_particle_weight(overlaps, norms, state_norm2; minimum_separation=1)

Sum `abs2(overlaps[n,m]) / norms[m-n] / state_norm2` for ordered positions
`n < m` separated by at least `minimum_separation`. `state_norm2` is
`⟨ψ|ψ⟩`, not its square root. Dividing by it removes an overall change in
the saved state's norm; this division does not undo any change in the
state's direction caused by time evolution or truncation.

This sum is the probability of the selected subspace only when the
localized reference states are mutually orthogonal. Otherwise their Gram
matrix is needed and this formula must not be interpreted as a probability.
The routine cannot establish orthogonality from the overlap matrix alone.
In particular, weights calculated with different fixed-momentum tensors
must not be added unless their subspaces have also been shown orthogonal.

`overlaps` must be square and finite. `norms` must contain finite, positive
squared norms through separation `L-1` whenever a pair is selected, and
`state_norm2` must be finite and positive. Excluding every pair by setting
the minimum separation to at least `L` returns zero. The function does not
clip the sum to one: a value above one should prompt a check of the basis,
normalization, and contractions.
"""
function two_particle_weight(overlaps, norms, state_norm2::Real;
                             minimum_separation::Integer=1)
    minimum_separation >= 1 || throw(ArgumentError("minimum_separation must be positive"))
    overlaps isa AbstractMatrix || throw(ArgumentError("overlaps must be a matrix"))
    size(overlaps, 1) == size(overlaps, 2) || throw(ArgumentError("overlaps must be square"))
    all(isfinite, overlaps) || throw(ArgumentError("overlaps must contain only finite entries"))
    norms isa AbstractVector || throw(ArgumentError("norms must be a vector"))
    all(value -> isfinite(value) && value > 0, norms) ||
        throw(ArgumentError("norms must contain only finite, positive squared norms"))
    isfinite(state_norm2) && state_norm2 > 0 || throw(ArgumentError("state_norm2 must be finite and positive"))
    L = size(overlaps, 1)
    minimum_separation >= L && return 0.0
    length(norms) >= L-1 || throw(ArgumentError("norms must extend through separation L-1"))
    result = 0.0
    for n in 1:(L-minimum_separation), m in (n+minimum_separation):L
        result += abs2(overlaps[n, m]) / norms[m-n]
    end
    return result / state_norm2
end

end # module IFTTwoParticleOverlap
