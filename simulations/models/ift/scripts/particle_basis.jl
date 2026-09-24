"""
    IFTParticleBasis

Construct excitation tensors for particle overlaps with the Ising MPS states.
The vacuum uses a one-site unit cell. Tensors returned as arrays have indices
`(left bond, physical spin, right bond)` and use the mixed-canonical convention
`… AL B AR …` of MPSKit's quasiparticle ansatz.

For a two-particle position state, the left insertion must obey the left
gauge condition and the right insertion the right gauge condition. These
conditions make states with different ordered insertion positions orthogonal.
Their norms, and overlaps between different tensor choices at the same
positions, must still be evaluated when forming a two-particle basis.
"""
module IFTParticleBasis

using LinearAlgebra
using MPSKit
using TensorKit

export vacuum_arrays, right_gauge_tensor, excitation_tensors
export gauge_residuals, orthonormal_tensor_basis

"""
    vacuum_arrays(vac)

Return `(; AL, AR, C, Cinv)` for a one-site uniform vacuum MPS. `AL` and `AR`
are dense complex arrays with shape `(D, d, D)`, and `C` is the vacuum bond
matrix, with `AL[:,s,:] * C = C * AR[:,s,:]`. `Cinv` is obtained by solving
against the identity; no small Schmidt values are discarded.

The arrays are copies, so changing them does not alter `vac`. A unit cell
longer than one site, incompatible tensor dimensions, or a singular bond
matrix cannot be used by this helper. A poorly conditioned `C` may amplify
roundoff in a subsequent contraction involving `Cinv`; this function does
not truncate it or replace it by a pseudoinverse.
"""
function vacuum_arrays(vac::InfiniteMPS)
    length(vac) == 1 || throw(ArgumentError("the vacuum must have a one-site unit cell"))
    AL = ComplexF64.(convert(Array, vac.AL[1]))
    AR = ComplexF64.(convert(Array, vac.AR[1]))
    C = ComplexF64.(convert(Array, vac.C[1]))
    D, _, Dr = size(AL)
    D == Dr && size(AR) == size(AL) && size(C) == (D, D) ||
        throw(DimensionMismatch("vacuum tensors must have matching square virtual spaces"))
    Cinv = C \ Matrix{ComplexF64}(I, D, D)
    return (; AL, AR, C, Cinv)
end

"""
    right_gauge_tensor(vac, B, k)

Express a left-gauged excitation tensor `B` in the right gauge at lattice
momentum `k`. Both representations describe the same momentum eigenstate
`sum_n exp(im*k*n) |… AL B(n) AR …⟩`. They generally describe different
localized states before this sum is taken.

`B` must have shape `(D,d,D)` in the mixed-canonical vacuum convention, with
`sum_s AL[:,s,:]' * B[:,s,:] = 0`. The returned array satisfies
`sum_s BR[:,s,:] * AR[:,s,:]' = 0`. The conversion uses MPSKit's
momentum-dependent transfer-matrix equation, rather than projecting `B`
directly onto the right null space. It preserves a global phase multiplying
`B`; it neither chooses a new phase nor renormalizes the tensor.

Only the topologically trivial excitation with a one-dimensional auxiliary
space is supported here. An input outside the left-gauged tensor space by
more than `1e-8 * max(1,norm(B))` is rejected, since the quasiparticle setter
would otherwise silently project away that component. MPSKit reports a
warning if its transfer-matrix solve fails to converge.
"""
function right_gauge_tensor(vac::InfiniteMPS, B::AbstractArray{<:Number,3}, k::Real)
    length(vac) == 1 || throw(ArgumentError("the vacuum must have a one-site unit cell"))
    isfinite(k) || throw(ArgumentError("the momentum must be finite"))
    all(isfinite, B) || throw(ArgumentError("the excitation tensor must be finite"))
    qp = LeftGaugedQP(zero, vac; momentum=Float64(k))
    template = qp[1]
    expected = size(convert(Array, template))
    expected[3] == 1 || throw(ArgumentError("the excitation auxiliary space must be one-dimensional"))
    size(B) == (expected[1], expected[2], expected[4]) ||
        throw(DimensionMismatch("the excitation tensor does not match the vacuum virtual and physical spaces"))
    qp[1] = TensorMap(
        reshape(ComplexF64.(B), expected), codomain(template), domain(template)
    )
    reconstructed = convert(Array, qp[1])[:, :, 1, :]
    norm(reconstructed - B) <= 1e-8 * max(1, norm(B)) ||
        throw(ArgumentError("the supplied excitation tensor does not satisfy the left gauge condition"))
    right_qp = convert(RightGaugedQP, qp)
    return ComplexF64.(convert(Array, right_qp[1])[:, :, 1, :])
end

"""
    excitation_tensors(vac, ham, momenta; num=1)

Calculate `num` low-energy quasiparticle solutions at each supplied lattice
momentum. Return `(; momenta, energies, left, right)`. The last three fields
are matrices indexed by `[momentum index, excitation index]`; entries of
`left` and `right` are dense `(D,d,D)` excitation tensors in the corresponding
gauge. `energies` contains gaps above the vacuum energy, in lattice units.

The computation calls `MPSKit.excitations` on the saved vacuum, without
optimizing a new ground state. Each right-gauged tensor is converted from its
left-gauged partner, so the two carry the same overall phase. Phases between
different momenta or excitation indices are not fixed here.

An excitation index orders the eigenvalues returned by the eigensolver; it
does not identify a particle species across momenta. In particular, a
solution near or above a multiparticle continuum must not be named a stable
particle merely because the solver returned it. Momentum-dependent overlaps
with a chosen particle require its tensors at the momenta being analyzed.
"""
function excitation_tensors(vac::InfiniteMPS, ham, momenta; num::Integer=1)
    length(vac) == 1 || throw(ArgumentError("the vacuum must have a one-site unit cell"))
    num > 0 || throw(ArgumentError("num must be positive"))
    ks = Float64.(collect(momenta))
    isempty(ks) && throw(ArgumentError("at least one momentum is required"))
    all(isfinite, ks) || throw(ArgumentError("all momenta must be finite"))
    energies, qps = excitations(ham, QuasiparticleAnsatz(), ks, vac; num=Int(num))
    left = Matrix{Array{ComplexF64,3}}(undef, length(ks), num)
    right = similar(left)
    for j in axes(left, 2), i in axes(left, 1)
        left[i, j] = ComplexF64.(convert(Array, qps[i, j][1])[:, :, 1, :])
        right[i, j] = right_gauge_tensor(vac, left[i, j], ks[i])
    end
    return (; momenta=ks, energies, left, right)
end

"""
    gauge_residuals(vac, BL, BR)

Measure the canonical and excitation gauge conditions using dense arrays.
Return their unscaled Frobenius norms as a named tuple:

- `left_isometry`: `norm(sum_s AL_s' * AL_s - I)`.
- `right_isometry`: `norm(sum_s AR_s * AR_s' - I)`.
- `canonical_match`: the combined norm of `AL_s*C - C*AR_s` over all spins.
- `left_gauge`: `norm(sum_s AL_s' * BL_s)`.
- `right_gauge`: `norm(sum_s BR_s * AR_s')`.
- `left_norm`, `right_norm`: Frobenius norms of `BL` and `BR`.

For a one-site mixed-canonical vacuum and a tensor obeying its corresponding
gauge condition, the tensor norm equals the single-excitation variational
norm. It is not the norm of a two-particle insertion state. These residuals
check tensor conventions and numerical contractions, not whether the
excitation is a stable particle or whether the vacuum bond dimension is
sufficient for a scattering calculation.
"""
function gauge_residuals(
    vac::InfiniteMPS, BL::AbstractArray{<:Number,3}, BR::AbstractArray{<:Number,3}
)
    (; AL, AR, C) = vacuum_arrays(vac)
    size(BL) == size(AL) && size(BR) == size(AR) ||
        throw(DimensionMismatch("the excitation tensors must have the vacuum tensor dimensions"))
    D, d, _ = size(AL)
    left_iso = zeros(ComplexF64, D, D)
    right_iso = zeros(ComplexF64, D, D)
    left_gauge = zeros(ComplexF64, D, D)
    right_gauge = zeros(ComplexF64, D, D)
    canonical_match_squared = 0.0
    for s in 1:d
        al, ar = @view(AL[:, s, :]), @view(AR[:, s, :])
        bl, br = @view(BL[:, s, :]), @view(BR[:, s, :])
        left_iso .+= al' * al
        right_iso .+= ar * ar'
        left_gauge .+= al' * bl
        right_gauge .+= br * ar'
        canonical_match_squared += norm(al * C - C * ar)^2
    end
    return (
        left_isometry=norm(left_iso - I),
        right_isometry=norm(right_iso - I),
        canonical_match=sqrt(canonical_match_squared),
        left_gauge=norm(left_gauge),
        right_gauge=norm(right_gauge),
        left_norm=norm(BL),
        right_norm=norm(BR),
    )
end

"""
    orthonormal_tensor_basis(tensors; rtol=1e-10)

Find an orthonormal basis for a collection of dense excitation tensors by
applying an SVD to their flattened columns. All tensors must have the same
shape `(D,d,D)` and the same canonical and excitation gauge conventions.
The inner product is the ordinary complex Frobenius inner product, which
is the single-excitation inner product in this mixed-canonical convention.

Return `(; basis, coefficients, singular_values, relative_errors)`, where
`basis` is a vector of arrays and

```julia
tensors[j] ≈ sum(coefficients[a,j] * basis[a] for a in eachindex(basis))
```

Singular values no larger than `rtol` times the largest are discarded.
`singular_values` contains the full singular-value list; `relative_errors[j]`
is the norm of the omitted part divided by `norm(tensors[j])`, or zero for a
zero input tensor. Empty input and an entirely zero collection are rejected.
The coefficients retain complex phases; reconstructing a bra overlap uses
their complex conjugates.

This is a reference basis for evaluating many overlaps efficiently. Its
vectors need not be energy eigenstates or distinct particle species.
Summing overlap weights over all reference vectors therefore does not give
the probability for one chosen particle branch. For that probability,
reconstruct the momentum-dependent excitation tensors before summing the
squared momentum-space overlaps, as in Milsted et al., arXiv:2012.07243,
Appendix D.2.1.
"""
function orthonormal_tensor_basis(tensors; rtol::Real=1e-10)
    isfinite(rtol) && 0 <= rtol < 1 || throw(ArgumentError("rtol must satisfy 0 ≤ rtol < 1"))
    values = collect(tensors)
    isempty(values) && throw(ArgumentError("at least one tensor is required"))
    shape = size(first(values))
    length(shape) == 3 || throw(DimensionMismatch("excitation tensors must have three indices"))
    all(B -> size(B) == shape, values) || throw(DimensionMismatch("all excitation tensors must have the same shape"))
    all(B -> all(isfinite, B), values) || throw(ArgumentError("the excitation tensors must be finite"))
    columns = hcat((vec(ComplexF64.(B)) for B in values)...)
    decomposition = svd(columns; full=false)
    largest = first(decomposition.S)
    largest > 0 || throw(ArgumentError("the tensor collection is entirely zero"))
    kept = findall(s -> s > rtol * largest, decomposition.S)
    vectors = decomposition.U[:, kept]
    coefficients = Diagonal(decomposition.S[kept]) * decomposition.Vt[kept, :]
    basis = [reshape(copy(vectors[:, a]), shape) for a in axes(vectors, 2)]
    remainder = columns - vectors * coefficients
    relative_errors = [
        iszero(norm(@view columns[:, j])) ? 0.0 :
        norm(@view remainder[:, j]) / norm(@view columns[:, j])
        for j in axes(columns, 2)
    ]
    return (; basis, coefficients, singular_values=decomposition.S, relative_errors)
end

end
