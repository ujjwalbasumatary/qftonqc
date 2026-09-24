"""
    IFTThreeParticleMomentum

Contract a finite MPS with three ordered excitation insertions, retaining
the excitation tensor's dependence on each particle's momentum. The
position sums are combined into left and right environments before the
middle excitation is contracted.
"""
module IFTThreeParticleMomentum

using LinearAlgebra

include(joinpath(@__DIR__, "two_particle_overlap.jl"))
using .IFTTwoParticleOverlap: _finite_array, _left_transfer, _right_transfer

export three_particle_momentum_amplitudes

function _momenta(values, name)
    momenta = collect(values)
    isempty(momenta) && throw(ArgumentError("$name must contain at least one momentum"))
    all(k -> k isa Real && isfinite(k), momenta) ||
        throw(ArgumentError("$name must contain finite real momenta"))
    result = Float64.(momenta)
    all(isfinite, result) || throw(ArgumentError("$name is not finite after conversion to Float64"))
    return result
end

function _bank(tensors, momenta, shape, name)
    length(tensors) == length(momenta) ||
        throw(ArgumentError("$name must contain one tensor for each momentum"))
    bank = [_finite_array(B, 3, "$name[$i]") for (i,B) in enumerate(tensors)]
    all(B -> size(B) == shape, bank) ||
        throw(ArgumentError("every tensor in $name must have the same dimensions as AL"))
    return bank
end

function _multiply_center(bank, Cinv)
    result = [similar(B) for B in bank]
    for i in eachindex(bank), s in axes(bank[i],2)
        mul!(@view(result[i][:,s,:]), @view(bank[i][:,s,:]), Cinv)
    end
    all(B -> all(isfinite,B), result) ||
        throw(ArgumentError("multiplication by Cinv produced nonfinite entries"))
    return result
end

raw"""
    three_particle_momentum_amplitudes(tensors, AL, AR, Cinv,
        left_tensors, middle_tensors, right_tensors, kL, kM, kR;
        minimum_separation=1, state_norm2=1.0,
        fourier_length=length(tensors))

Return a complex array with indices `(left momentum, middle momentum,
right momentum)`. Each entry is the ordered position sum

```math
A_{ijk}=\frac{1}{L_F^{3/2}\sqrt{\langle\psi|\psi\rangle}}
\sum_{\substack{n<m<r\\m-n\ge g,\;r-m\ge g}}
e^{-i(k_{L,i}n+k_{M,j}m+k_{R,k}r)}
\langle n,B_L(k_{L,i});m,B_M(k_{M,j});r,B_R(k_{R,k})|\psi\rangle.
```

Here `g = minimum_separation`, `L = length(tensors)`,
`L_F = fourier_length`, and the norm in the denominator is the supplied
`state_norm2`. Its default of one leaves the ket's norm unchanged. A caller
normalizing a saved state should pass `real(dot(psi,psi))`, rather than the
square root of that quantity.

All site tensors have indices `(left bond, physical index, right bond)`.
`tensors` represents the ket as one finite MPS, for example
`AC[1], AR[2], …, AR[L]`. Its internal bond dimensions may vary. The two
external bonds must match the uniform vacuum's bond dimension, and the
ket and reference must have the same canonical infinite boundaries, whose
contractions are identities in these gauges.

The reference has `AL` before `n`, `left_tensors[i]*Cinv` at `n`, `AL`
between insertions, `middle_tensors[j]*Cinv` at `m`, `right_tensors[k]` at
`r`, and `AR` after `r`. Both multiplications by `Cinv` act on right
bonds. The first two excitation tensors use the left gauge and the final
one the right gauge. No tensor is normalized or rephased here. Thus a
phase multiplying any supplied excitation tensor enters its amplitudes
complex conjugated, as required for a bra overlap.

Each tensor collection must have one tensor per corresponding momentum.
The three momentum grids may differ and need not be uniformly spaced.
`fourier_length` is finite and positive and defaults to the window length.
For uniform grids with a common spacing `Δk`, set it to `2π/Δk` when the
squared amplitudes are to include the Fourier quadrature factor
`(Δk/2π)^3`. In particular, enlarging a window from 320 to 480 sites while
retaining spacing `2π/320` requires `fourier_length=320`, not 480. The
function does not infer this length from the grids. For unequal or
nonuniform spacings, the caller must instead supply the appropriate
momentum quadrature weights. Sites are numbered from one. The separation
must be positive; if no ordered triple fits, the returned array is zero.

For each middle site, the sums over allowed first and last positions are
formed as environments carrying the appropriate momentum phases. They
require `O((K_L+K_R)Lg)` transfer operations at fixed bond dimensions.
Matrix products then combine both environments with the local ket tensor
and all middle excitation tensors. The stored environments occupy
`O((K_L+K_R)LDχ)` complex numbers for vacuum bond dimension `D` and ket
bond dimension `χ`, together with a per-site array of size
`D²d*K_L*K_R`. No array indexed by all three positions is constructed.

These are raw Fourier overlaps. Their squared magnitudes alone do not
establish normalized particle probabilities: finite-separation references
need not be orthogonal, selected momentum intervals may omit amplitude,
and variational excitation branches require physical identification. The
function retains the complex amplitudes so that those questions can be
examined separately.
"""
function three_particle_momentum_amplitudes(tensors, AL, AR, Cinv,
        left_tensors, middle_tensors, right_tensors, kL, kM, kR;
        minimum_separation::Integer=1, state_norm2::Real=1.0,
        fourier_length::Real=length(tensors))
    minimum_separation >= 1 || throw(ArgumentError("minimum_separation must be positive"))
    isfinite(state_norm2) && state_norm2 > 0 ||
        throw(ArgumentError("state_norm2 must be finite and positive"))
    isfinite(fourier_length) && fourier_length > 0 ||
        throw(ArgumentError("fourier_length must be finite and positive"))
    ks_left = _momenta(kL,"kL")
    ks_middle = _momenta(kM,"kM")
    ks_right = _momenta(kR,"kR")
    L = length(tensors)
    L >= 1 || throw(ArgumentError("tensors must contain at least one site"))
    left = _finite_array(AL,3,"AL")
    right = _finite_array(AR,3,"AR")
    center_inverse = _finite_array(Cinv,2,"Cinv")
    D,d,Dr = size(left)
    D == Dr || throw(ArgumentError("AL must have equal left and right bond dimensions"))
    size(right) == size(left) || throw(ArgumentError("AR must have the same dimensions as AL"))
    size(center_inverse) == (D,D) || throw(ArgumentError("Cinv must have size ($D,$D)"))
    left_bank = _bank(left_tensors,ks_left,size(left),"left_tensors")
    middle_bank = _bank(middle_tensors,ks_middle,size(left),"middle_tensors")
    right_bank = _bank(right_tensors,ks_right,size(left),"right_tensors")
    first_bank = _multiply_center(left_bank,center_inverse)
    middle_bank = _multiply_center(middle_bank,center_inverse)
    ket = [_finite_array(A,3,"tensors[$i]") for (i,A) in enumerate(tensors)]
    for n in 1:L
        size(ket[n],2) == d || throw(ArgumentError("tensors[$n] has a different physical dimension from AL"))
        n == L || size(ket[n],3) == size(ket[n+1],1) ||
            throw(ArgumentError("ket bond dimensions disagree between sites $n and $(n+1)"))
    end
    size(ket[1],1) == D || throw(ArgumentError("the left external ket bond must match AL"))
    size(ket[end],3) == D || throw(ArgumentError("the right external ket bond must match AR"))
    KL,KM,KR = length(ks_left),length(ks_middle),length(ks_right)
    amplitudes = zeros(ComplexF64,KL,KM,KR)
    g = minimum_separation
    g > div(L-1,2) && return amplitudes

    left_before = Vector{Matrix{ComplexF64}}(undef,L)
    right_after = Vector{Matrix{ComplexF64}}(undef,L)
    left_before[1] = Matrix{ComplexF64}(I,D,D)
    right_after[L] = Matrix{ComplexF64}(I,D,D)
    for n in 2:L
        left_before[n] = _left_transfer(left_before[n-1],left,ket[n-1])
    end
    for r in L-1:-1:1
        right_after[r] = _right_transfer(right_after[r+1],right,ket[r+1])
    end

    middle_sites = g+1:L-g
    left_sums = Vector{Matrix{ComplexF64}}(undef,L)
    right_sums = Vector{Matrix{ComplexF64}}(undef,L)
    for m in middle_sites
        left_sums[m] = Matrix{ComplexF64}(undef,D*size(ket[m],1),KL)
        right_sums[m] = Matrix{ComplexF64}(undef,size(ket[m],3)*D,KR)
    end
    for i in 1:KL
        environment = zeros(ComplexF64,D,size(ket[g+1],1))
        for m in middle_sites
            m > g+1 && (environment = _left_transfer(environment,left,ket[m-1]))
            n = m-g
            source = _left_transfer(left_before[n],first_bank[i],ket[n])
            for site in n+1:m-1
                source = _left_transfer(source,left,ket[site])
            end
            environment .+= cis(-ks_left[i]*n) .* source
            left_sums[m][:,i] = vec(environment)
        end
    end
    for k in 1:KR
        environment = zeros(ComplexF64,size(ket[L-g],3),D)
        for m in reverse(middle_sites)
            m < L-g && (environment = _right_transfer(environment,left,ket[m+1]))
            r = m+g
            source = _right_transfer(right_after[r],right_bank[k],ket[r])
            for site in r-1:-1:m+1
                source = _right_transfer(source,left,ket[site])
            end
            environment .+= cis(-ks_right[k]*r) .* source
            right_sums[m][:,k] = vec(environment)
        end
    end

    middle_columns = hcat(vec.(middle_bank)...)
    open_middle = Matrix{ComplexF64}(undef,D*d*D,KL*KR)
    open_middle_view = reshape(open_middle,D,d,D,KL,KR)
    for m in middle_sites
        χleft,_,χright = size(ket[m])
        # Stack the vacuum index and kL on the row side; stack the other
        # vacuum index and kR on the column side. No bra conjugation is
        # applied here because it is already present in the environments.
        left_rows = reshape(permutedims(reshape(left_sums[m],D,χleft,KL),(1,3,2)),D*KL,χleft)
        right_columns = reshape(right_sums[m],χright,D*KR)
        for s in 1:d
            contraction = (left_rows * @view(ket[m][:,s,:])) * right_columns
            @views open_middle_view[:,s,:,:,:] .= permutedims(reshape(contraction,D,KL,D,KR),(1,3,2,4))
        end
        middle_amplitudes = middle_columns' * open_middle
        for j in 1:KM
            phase = cis(-ks_middle[j]*m)
            for k in 1:KR,i in 1:KL
                amplitudes[i,j,k] += phase * middle_amplitudes[j,i+(k-1)*KL]
            end
        end
    end
    normalization = fourier_length^(3/2) * sqrt(state_norm2)
    isfinite(normalization) && normalization > 0 ||
        throw(ArgumentError("the Fourier and state normalization factor must be finite and positive"))
    amplitudes ./= normalization
    all(isfinite,amplitudes) || throw(ArgumentError("the momentum amplitudes have nonfinite entries"))
    return amplitudes
end

end # module IFTThreeParticleMomentum
