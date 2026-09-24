"""
    IFTThreeParticleOverlap

Overlaps with three ordered localized insertions above an MPS vacuum, and
the Gram matrices needed to combine those overlaps into a subspace weight.
All site tensors use indices `(left bond, physical index, right bond)`.
"""
module IFTThreeParticleOverlap

using LinearAlgebra

include(joinpath(@__DIR__, "two_particle_overlap.jl"))
using .IFTTwoParticleOverlap: _finite_array, _reference_tensors,
                             _left_transfer, _right_transfer, _close_environments

export three_particle_overlaps, middle_position_gram, localized_triple_norms,
       pair_triple_cross_gram, three_particle_weight

function _triple_reference(AL, AR, Cinv, BL, BM, BR)
    left, right, first, last = _reference_tensors(AL, Cinv, BL, BR; AR)
    _, _, middle, _ = _reference_tensors(AL, Cinv, BM, BR)
    return left, right, first, middle, last
end

function _sites(sites, L, name)
    indices = collect(sites)
    all(site -> site isa Integer && 1 <= site <= L, indices) ||
        throw(ArgumentError("$name must contain integer sites in 1:$L"))
    return sort!(unique!(Int.(indices)))
end

"""
    three_particle_overlaps(tensors, AL, AR, Cinv, BL, BM, BR;
                            minimum_separation=1,
                            first_sites=1:length(tensors),
                            middle_sites=1:length(tensors),
                            last_sites=1:length(tensors))

Calculate `⟨n,BL; m,BM; r,BR | ψ⟩` for ordered sites `n < m < r`.
The reference chain is `AL` before `n`, `BL*Cinv` at `n`, `AL` up to
`m`, `BM*Cinv` at `m`, `AL` up to `r`, `BR` at `r`, and `AR` thereafter.
Both multiplications by `Cinv` act on right bonds. The ket is supplied as
one finite MPS, for example `AC[1], AR[2], …, AR[L]`; its overall norm and
complex phase are unchanged. Its external bonds must match the vacuum
bonds, and the infinite exterior contractions must be identities in the
same gauges for ket and reference.

The two adjacent distances must each be at least `minimum_separation`.
Optional site collections restrict any of the three positions. Their
ordering and duplicate entries do not matter; empty collections give no
overlaps. The result is a dictionary keyed by `(n,r)`. Each entry contains
`middle_sites`, an increasing vector of the selected `m`, and `overlaps`,
the corresponding complex amplitudes. Only selected triples are stored.

The contractions before the first and after the last insertion are reused.
The right contraction through the middle tensor is stored for each allowed
middle and last site. Only a scalar contraction is then needed for each
triple. There are quadratically many transfer operations and cubically
many scalar contractions for unrestricted sites at fixed bond dimensions.
The stored right contractions take quadratic space in the window length.
No dense many-body state or individual MPS for each triple is constructed.

With left-canonical `AL`, right-canonical `AR`, and
`Σₛ ALₛ† BLₛ = 0`, `Σₛ BRₛ ARₛ† = 0`, states with different outer
positions `(n,r)` are orthogonal. States with the same outer positions
but different middle positions need not be orthogonal, even when `BM`
obeys the left excitation gauge. Use [`middle_position_gram`](@ref) for
those overlaps; dividing each amplitude by its individual norm does not
replace the Gram matrix.

The tensors are fixed throughout the window. This gives a specified
three-insertion subspace, not the entire physical three-particle sector
with its momentum-dependent excitation tensors. Its complement must not
be interpreted as a sum of other particle-production probabilities.
"""
function three_particle_overlaps(tensors, AL, AR, Cinv, BL, BM, BR;
        minimum_separation::Integer=1, first_sites=1:length(tensors),
        middle_sites=1:length(tensors), last_sites=1:length(tensors))
    minimum_separation >= 1 || throw(ArgumentError("minimum_separation must be positive"))
    left, right, first, middle, last = _triple_reference(AL, AR, Cinv, BL, BM, BR)
    L = length(tensors)
    L >= 1 || throw(ArgumentError("tensors must contain at least one site"))
    ket = [_finite_array(tensor, 3, "tensors[$i]") for (i,tensor) in enumerate(tensors)]
    D, d, _ = size(left)
    for i in 1:L
        size(ket[i],2) == d || throw(ArgumentError("tensors[$i] has the wrong physical dimension"))
        i == L || size(ket[i],3) == size(ket[i+1],1) ||
            throw(ArgumentError("ket bonds disagree between sites $i and $(i+1)"))
    end
    size(ket[1],1) == D || throw(ArgumentError("the left external ket bond must match AL"))
    size(ket[end],3) == D || throw(ArgumentError("the right external ket bond must match AR"))
    first_indices = _sites(first_sites,L,"first_sites")
    middle_indices = _sites(middle_sites,L,"middle_sites")
    last_indices = _sites(last_sites,L,"last_sites")
    Block = NamedTuple{(:middle_sites,:overlaps),Tuple{Vector{Int},Vector{ComplexF64}}}
    blocks = Dict{Tuple{Int,Int},Block}()
    (isempty(first_indices) || isempty(middle_indices) || isempty(last_indices)) && return blocks
    minimum_separation > div(L-1,2) && return blocks
    middle_allowed = falses(L)
    middle_allowed[middle_indices] .= true
    last_end = last_indices[end]

    left_before = Vector{Matrix{ComplexF64}}(undef,L)
    left_before[1] = Matrix{ComplexF64}(I,D,D)
    for n in 2:L
        left_before[n] = _left_transfer(left_before[n-1],left,ket[n-1])
    end
    right_at = Vector{Matrix{ComplexF64}}(undef,L)
    right_after = Matrix{ComplexF64}(I,D,D)
    for r in L:-1:1
        right_at[r] = _right_transfer(right_after,last,ket[r])
        r > 1 && (right_after = _right_transfer(right_after,right,ket[r]))
    end

    # For each middle site, columns hold the right environment for each
    # allowed last insertion. Transposition here, rather than adjunction,
    # preserves the conjugation already applied to the reference tensors.
    possible_middle = filter(m -> m-first_indices[1] >= minimum_separation &&
                             last_end-m >= minimum_separation,middle_indices)
    isempty(possible_middle) && return blocks
    fill!(middle_allowed,false)
    middle_allowed[possible_middle] .= true
    right_positions = Vector{Vector{Int}}(undef,L)
    right_vectors = Vector{Matrix{ComplexF64}}(undef,L)
    for m in possible_middle
        start = searchsortedfirst(last_indices,m+minimum_separation)
        right_positions[m] = last_indices[start:end]
        right_vectors[m] = Matrix{ComplexF64}(undef,D*size(ket[m],1),length(right_positions[m]))
    end
    for r in last_indices
        environment = right_at[r]
        for m in r-1:-1:possible_middle[1]
            if middle_allowed[m] && r-m >= minimum_separation
                column = searchsortedfirst(right_positions[m],r)
                with_middle = _right_transfer(environment,middle,ket[m])
                right_vectors[m][:,column] = vec(transpose(with_middle))
            end
            m > possible_middle[1] && (environment = _right_transfer(environment,left,ket[m]))
        end
    end

    for n in first_indices
        n + 2minimum_separation <= last_end || continue
        first_environment = _left_transfer(left_before[n],first,ket[n])
        for m in n+1:last_end-minimum_separation
            if m-n >= minimum_separation && middle_allowed[m]
                amplitudes = transpose(right_vectors[m]) * vec(first_environment)
                for (r,amplitude) in zip(right_positions[m],amplitudes)
                    block = get!(blocks,(n,r)) do
                        (;middle_sites=Int[],overlaps=ComplexF64[])
                    end
                    push!(block.middle_sites,m)
                    push!(block.overlaps,amplitude)
                end
            end
            m < last_end-minimum_separation &&
                (first_environment = _left_transfer(first_environment,left,ket[m]))
        end
    end
    return blocks
end

function _gram_contractions(AL,Cinv,BL,BM,BR,span,middle_offsets)
    span >= 2 || throw(ArgumentError("span must be at least two lattice sites"))
    offsets = collect(middle_offsets)
    all(offset -> offset isa Integer && 1 <= offset < span,offsets) ||
        throw(ArgumentError("middle_offsets must contain integers strictly between zero and span"))
    length(unique(offsets)) == length(offsets) || throw(ArgumentError("middle_offsets must not repeat a position"))
    offsets = Int.(offsets)
    left, _, first, middle, last = _triple_reference(AL,nothing,Cinv,BL,BM,BR)
    D,d,_ = size(left)
    initial = zeros(ComplexF64,D,D)
    final = zeros(ComplexF64,D,D)
    for s in 1:d
        initial .+= @view(first[:,s,:])' * @view(first[:,s,:])
        final .+= @view(last[:,s,:]) * @view(last[:,s,:])'
    end
    left_before = Vector{Matrix{ComplexF64}}(undef,span-1)
    right_after = Vector{Matrix{ComplexF64}}(undef,span-1)
    left_before[1], right_after[span-1] = initial, final
    for a in 2:span-1
        left_before[a] = _left_transfer(left_before[a-1],left,left)
    end
    for b in span-2:-1:1
        right_after[b] = _right_transfer(right_after[b+1],left,left)
    end
    return (;left,middle,offsets,left_before,right_after)
end

"""
    middle_position_gram(AL, Cinv, BL, BM, BR, span;
                         middle_offsets=1:span-1)

Return the Gram matrix for triples whose first and last insertions are
`span` sites apart. Entry `[i,j]` is the overlap with the bra's middle
insertion at offset `middle_offsets[i]` and the ket's at
`middle_offsets[j]`, both measured from the first insertion. The supplied
offset order is retained. Offsets must be distinct integers strictly
between zero and `span`; an empty collection gives a `0×0` matrix.

The outer tensors are `BL*Cinv` and `BR`, and the middle tensor is
`BM*Cinv`. Identity contractions outside the pair of outer insertions
are assumed. Because the vacuum is uniform, this matrix depends on the
span and the relative middle positions, not on the absolute first site.
It may therefore be reused for translated triples. The diagonal contains
the squared norms; off-diagonal entries retain the overlap between
different middle positions.

For offsets `a<b`, propagate the contraction from `BL*Cinv` through
`a-1` vacuum sites, insert the middle tensor in the bra, propagate to
`b`, insert it in the ket, and close against the remaining vacuum sites
and `BR`. Shared left and right contractions make the work quadratic in
the span at fixed bond dimension. The matrix is filled with its Hermitian
conjugate entries; zero eigenvalues are allowed.
"""
function middle_position_gram(AL,Cinv,BL,BM,BR,span::Integer;
                             middle_offsets=1:span-1)
    data = _gram_contractions(AL,Cinv,BL,BM,BR,span,middle_offsets)
    (;left,middle,offsets,left_before,right_after) = data
    number = length(offsets)
    gram = zeros(ComplexF64,number,number)
    index = Dict(offset=>j for (j,offset) in enumerate(offsets))
    right_middle = Dict(b=>_right_transfer(right_after[b],left,middle) for b in offsets)
    for a in sort(offsets)
        i = index[a]
        diagonal = _close_environments(_left_transfer(left_before[a],middle,middle),right_after[a])
        gram[i,i] = real(diagonal)
        environment = _left_transfer(left_before[a],middle,left)
        for b in a+1:maximum(offsets)
            if haskey(index,b)
                j = index[b]
                value = _close_environments(environment,right_middle[b])
                gram[i,j],gram[j,i] = value,conj(value)
            end
            b < maximum(offsets) && (environment = _left_transfer(environment,left,left))
        end
    end
    all(isfinite,gram) || throw(ArgumentError("the middle-position Gram matrix has nonfinite entries"))
    return gram
end

"""
    localized_triple_norms(AL, Cinv, BL, BM, BR, span;
                           middle_offsets=1:span-1)

Calculate the squared norm for each middle offset, in the supplied order,
using the same triple states as [`middle_position_gram`](@ref). Only the
diagonal contractions are performed. No tensor is normalized and a zero
norm is allowed; nonfinite or negative norms beyond roundoff are rejected.
"""
function localized_triple_norms(AL,Cinv,BL,BM,BR,span::Integer;
                                middle_offsets=1:span-1)
    (;left,middle,offsets,left_before,right_after) =
        _gram_contractions(AL,Cinv,BL,BM,BR,span,middle_offsets)
    values = [real(_close_environments(_left_transfer(left_before[a],middle,middle),right_after[a]))
              for a in offsets]
    all(isfinite,values) || throw(ArgumentError("a triple norm is nonfinite"))
    scale = maximum(abs,values;init=0.0)
    all(value -> value >= -1e-12*max(scale,eps()),values) ||
        throw(ArgumentError("a triple has a negative squared norm"))
    return values
end

"""
    pair_triple_cross_gram(AL, Cinv, BLpair, BRpair,
                           BLtriple, BMtriple, BRtriple, span;
                           middle_offsets=1:span-1)

Calculate `c[a] = ⟨triple at middle_offsets[a] | pair⟩` for states whose
first and last insertions occupy the same sites, separated by `span`.
The pair has `BLpair*Cinv` at its first site, `BRpair` at its last, and
`AL` between them. The triple has `BLtriple*Cinv`, `BMtriple*Cinv`, and
`BRtriple` at its three insertion sites. Multiplication by `Cinv` acts on
the right bond. The supplied middle-offset order is retained. As with
[`middle_position_gram`](@ref), the exterior contractions are identities
and all tensors must use the same vacuum gauges.

The initial mixed contraction is
`E = Σₛ (BLtripleₛ Cinv)† (BLpairₛ Cinv)`, and the final one is
`Q = Σₛ BRpairₛ BRtripleₛ†`. At the middle site, the bra contributes
`BMtriple*Cinv` while the ket contributes `AL`. Vacuum transfer operations
connect these insertions. Shared contractions give work linear in `span`
at fixed bond dimension, and neither state is normalized by this function.

This vector measures overlap between the pair and the chosen triple
references. The overlap need not vanish at finite separation even if the
middle tensor satisfies the left excitation gauge. If `G` is their
middle-position Gram matrix and `Npair` the pair's squared norm, then
`c† G⁺ c / Npair` is the weight of this normalized pair state in that
triple subspace. It provides a check for two-particle contamination of a
fixed-tensor three-insertion calculation. If weights from different outer
positions are combined, their mutual orthogonality also requires the
canonical and outer excitation gauge conditions; this cross contraction
alone does not check them.

The orientation matters for phases: multiplying the pair by `z` multiplies
this vector by `z`; multiplying every triple reference by `w` multiplies
it by `conj(w)`. Empty offsets give an empty complex vector. Other offsets
must be distinct integers strictly between zero and `span`.
"""
function pair_triple_cross_gram(AL,Cinv,BLpair,BRpair,BLtriple,BMtriple,BRtriple,
                               span::Integer;middle_offsets=1:span-1)
    span >= 2 || throw(ArgumentError("span must be at least two lattice sites"))
    offsets = collect(middle_offsets)
    all(a -> a isa Integer && 1 <= a < span,offsets) ||
        throw(ArgumentError("middle_offsets must contain integers strictly between zero and span"))
    length(unique(offsets)) == length(offsets) || throw(ArgumentError("middle_offsets must not repeat a position"))
    offsets = Int.(offsets)
    left,_,first_triple,middle,last_triple =
        _triple_reference(AL,nothing,Cinv,BLtriple,BMtriple,BRtriple)
    _,_,first_pair,last_pair = _reference_tensors(AL,Cinv,BLpair,BRpair)
    D,d,_ = size(left)
    environment = zeros(ComplexF64,D,D)
    final = zeros(ComplexF64,D,D)
    for s in 1:d
        environment .+= @view(first_triple[:,s,:])' * @view(first_pair[:,s,:])
        final .+= @view(last_pair[:,s,:]) * @view(last_triple[:,s,:])'
    end
    right_after = Vector{Matrix{ComplexF64}}(undef,span-1)
    right_after[span-1] = final
    for a in span-2:-1:1
        right_after[a] = _right_transfer(right_after[a+1],left,left)
    end
    result = Vector{ComplexF64}(undef,length(offsets))
    index = Dict(a=>j for (j,a) in enumerate(offsets))
    for a in 1:span-1
        if haskey(index,a)
            result[index[a]] = _close_environments(
                _left_transfer(environment,middle,left),right_after[a])
        end
        a < span-1 && (environment = _left_transfer(environment,left,left))
    end
    all(isfinite,result) || throw(ArgumentError("the pair-triple overlaps have nonfinite entries"))
    return result
end

"""
    three_particle_weight(blocks, grams, state_norm2;
                          minimum_separation=1,
                          gram_minimum_separation=minimum_separation,
                          rtol=1e-10)

Calculate `Σ_(n,r) o† G⁺ o / state_norm2`, including the middle-position
Gram matrix separately for each pair of outer sites. `blocks` is the
dictionary returned by [`three_particle_overlaps`](@ref). `grams[span]`
must contain [`middle_position_gram`](@ref) evaluated on the consecutive
offsets `gram_minimum_separation:span-gram_minimum_separation`, in that
order. The selected middle sites are restricted further by
`minimum_separation`; any site restrictions already present in `blocks`
are retained. A submatrix is taken when only some middle sites are used.

`state_norm2` means `⟨ψ|ψ⟩`. An eigenvalue is retained in the pseudoinverse
when it exceeds `rtol` times the largest eigenvalue of its Gram submatrix.
The return value contains `weight`, `discarded_overlap_norm_squared`, the
smallest and largest encountered eigenvalues, the largest condition
number among retained modes, the smallest and largest retained ranks,
and the numbers of outer blocks and triples. Discarded overlap is reported
in the ordinary coefficient norm; it is a numerical diagnostic, not an
additional physical probability. Repeating the calculation at several
cutoffs is appropriate if small Gram eigenvalues carry appreciable overlap.

This is a projection onto the selected fixed-tensor subspace only when
different outer-position blocks are orthogonal. The canonical and outer
excitation gauge conditions in [`three_particle_overlaps`](@ref) establish
that property; this function cannot infer it from the supplied matrices.
It neither clips the result to one nor identifies the full momentum-
dependent three-particle sector. Significantly negative Gram eigenvalues
or a non-Hermitian input matrix throw `ArgumentError`.
"""
function three_particle_weight(blocks,grams,state_norm2::Real;
        minimum_separation::Integer=1,gram_minimum_separation::Integer=minimum_separation,
        rtol::Real=1e-10)
    minimum_separation >= gram_minimum_separation >= 1 ||
        throw(ArgumentError("separations must satisfy minimum_separation >= gram_minimum_separation >= 1"))
    isfinite(state_norm2) && state_norm2 > 0 || throw(ArgumentError("state_norm2 must be finite and positive"))
    isfinite(rtol) && 0 <= rtol < 1 || throw(ArgumentError("rtol must be finite and in [0,1)"))
    cache = Dict{Tuple{Int,Tuple{Vararg{Int}}},Any}()
    total,discarded = 0.0,0.0
    smallest,largest,maxcondition = Inf,-Inf,0.0
    minimum_rank,maximum_rank = typemax(Int),0
    block_count,triple_count = 0,0
    for ((n,r),block) in blocks
        n < r || throw(ArgumentError("outer positions must be ordered"))
        length(block.middle_sites) == length(block.overlaps) || throw(ArgumentError("middle sites and overlaps have different lengths"))
        all(isfinite,block.overlaps) || throw(ArgumentError("overlaps must be finite"))
        length(unique(block.middle_sites)) == length(block.middle_sites) || throw(ArgumentError("middle sites must not repeat"))
        all(m -> m isa Integer && n < m < r,block.middle_sites) || throw(ArgumentError("middle sites must lie between the outer positions"))
        selected = findall(m -> m-n >= minimum_separation && r-m >= minimum_separation,block.middle_sites)
        isempty(selected) && continue
        span = r-n
        indices = Tuple(block.middle_sites[j]-n-gram_minimum_separation+1 for j in selected)
        data = get!(cache,(span,indices)) do
            haskey(grams,span) || throw(ArgumentError("the Gram matrix for span $span is missing"))
            gram = grams[span]
            expected = span-2gram_minimum_separation+1
            gram isa AbstractMatrix && size(gram) == (expected,expected) ||
                throw(ArgumentError("the Gram matrix for span $span has the wrong size"))
            all(isfinite,gram) || throw(ArgumentError("the Gram matrix for span $span has nonfinite entries"))
            relative_error = norm(gram-gram') / max(norm(gram),eps())
            relative_error <= 1e-8 || throw(ArgumentError("the Gram matrix for span $span is not Hermitian"))
            submatrix = gram[collect(indices),collect(indices)]
            system = eigen(Hermitian((submatrix+submatrix')/2))
            values = system.values
            scale = maximum(abs,values;init=0.0)
            minimum(values) >= -1e-10*max(scale,eps()) ||
                throw(ArgumentError("the Gram matrix for span $span has a negative eigenvalue"))
            keep = values .> rtol*max(maximum(values),0.0)
            (;system,keep)
        end
        values,keep = data.system.values,data.keep
        rotated = data.system.vectors' * block.overlaps[selected]
        total += sum(abs2.(rotated[keep]) ./ values[keep])
        discarded += sum(abs2,rotated[.!keep])
        rank = count(keep)
        smallest,largest = min(smallest,minimum(values)),max(largest,maximum(values))
        rank > 0 && (maxcondition = max(maxcondition,maximum(values[keep])/minimum(values[keep])))
        minimum_rank,maximum_rank = min(minimum_rank,rank),max(maximum_rank,rank)
        block_count += 1
        triple_count += length(selected)
    end
    return (;weight=total/state_norm2,discarded_overlap_norm_squared=discarded/state_norm2,
            minimum_eigenvalue=block_count == 0 ? NaN : smallest,
            maximum_eigenvalue=block_count == 0 ? NaN : largest,
            maximum_retained_condition_number=maxcondition,
            minimum_retained_rank=block_count == 0 ? 0 : minimum_rank,
            maximum_retained_rank=maximum_rank,block_count,triple_count,rtol)
end

end # module IFTThreeParticleOverlap
