module IFTThreeParticleOverlapTests

using Test, LinearAlgebra, Random

include(joinpath(@__DIR__,"..","models","ift","scripts","three_particle_overlap.jl"))
using .IFTThreeParticleOverlap

function configuration_matrix(tensors,configuration)
    result = Matrix(@view tensors[1][:,configuration[1],:])
    for site in 2:length(tensors)
        result = result * @view(tensors[site][:,configuration[site],:])
    end
    return result
end

function dense_overlap(bra,ket)
    d,L = size(first(ket),2),length(ket)
    return sum(dot(configuration_matrix(bra,c),configuration_matrix(ket,c))
               for c in Iterators.product(ntuple(_->1:d,L)...))
end

function inserted(tensor,Cinv)
    result = similar(tensor)
    for s in axes(tensor,2)
        result[:,s,:] = tensor[:,s,:] * Cinv
    end
    return result
end

function triple_reference(AL,AR,Cinv,BL,BM,BR,L,n,m,r)
    DL,DM = inserted(BL,Cinv),inserted(BM,Cinv)
    return [j < n ? AL : j == n ? DL : j < m ? AL : j == m ? DM :
            j < r ? AL : j == r ? BR : AR for j in 1:L]
end

function sum_mps(references,coefficients)
    L,N = length(first(references)),length(references)
    D,d,_ = size(first(first(references)))
    result = [zeros(ComplexF64,j == 1 ? D : N*D,d,j == L ? D : N*D) for j in 1:L]
    for a in 1:N,j in 1:L
        left_indices = j == 1 ? (1:D) : ((a-1)*D+1:a*D)
        right_indices = j == L ? (1:D) : ((a-1)*D+1:a*D)
        result[j][left_indices,:,right_indices] .= (j == 1 ? coefficients[a] : 1) .* references[a][j]
    end
    return result
end

function canonical_reference(rng)
    D,d = 2,4
    matrices = [ComplexF64[1 0;0 1],ComplexF64[0 1;1 0],
                ComplexF64[0 -im;im 0],ComplexF64[1 0;0 -1]]
    probabilities = [0.4,0.1,0.2,0.3]
    A = cat((sqrt(probabilities[s]).*matrices[s] for s in 1:d)...;dims=3)
    A = permutedims(A,(1,3,2))
    function left_excitation()
        B = randn(rng,ComplexF64,D,d,D)
        component = sum(A[:,s,:]'*B[:,s,:] for s in 1:d)
        for s in 1:d
            B[:,s,:] .-= A[:,s,:] * component
        end
        return B/norm(B)
    end
    BR = randn(rng,ComplexF64,D,d,D)
    component = sum(BR[:,s,:]*A[:,s,:]' for s in 1:d)
    for s in 1:d
        BR[:,s,:] .-= component * A[:,s,:]
    end
    return A,sqrt(2).*Matrix{ComplexF64}(I,D,D),left_excitation(),left_excitation(),BR/norm(BR)
end

@testset "Localized three-insertion overlaps and middle Gram matrices" begin
    rng = MersenneTwister(78142)
    up = reshape(ComplexF64[1,0],1,2,1)
    down = reshape(ComplexF64[0,1],1,2,1)
    C = ones(ComplexF64,1,1)

    @testset "All complex amplitudes agree with direct contractions" begin
        for D in (1,2)
            d,L = 2,6
            AL,AR,BL,BM,BR = (randn(rng,ComplexF64,D,d,D)/3 for _ in 1:5)
            Cinv = randn(rng,ComplexF64,D,D)
            dimensions = [D,2,3,2,3,2,D]
            ket = [randn(rng,ComplexF64,dimensions[j],d,dimensions[j+1])/3 for j in 1:L]
            blocks = three_particle_overlaps(ket,AL,AR,Cinv,BL,BM,BR)
            @test sum(length(block.overlaps) for block in values(blocks)) == binomial(L,3)
            for ((n,r),block) in blocks,(m,value) in zip(block.middle_sites,block.overlaps)
                reference = triple_reference(AL,AR,Cinv,BL,BM,BR,L,n,m,r)
                @test value ≈ dense_overlap(reference,ket) atol=1e-13 rtol=1e-11
            end
            restricted = three_particle_overlaps(ket,AL,AR,Cinv,BL,BM,BR;
                minimum_separation=2,first_sites=[2,1,1],middle_sites=[3,4],last_sites=[6,5])
            @test sum(length(block.overlaps) for block in values(restricted)) == 4
            for ((n,r),block) in restricted,(m,value) in zip(block.middle_sites,block.overlaps)
                @test n in (1,2) && m in (3,4) && r in (5,6) && min(m-n,r-m)>=2
                @test value ≈ dense_overlap(triple_reference(AL,AR,Cinv,BL,BM,BR,L,n,m,r),ket) atol=1e-13 rtol=1e-11
            end
        end
    end

    @testset "Gram off-diagonal entries and exact diagonal norms" begin
        D,d,span = 2,2,5
        AL,AR,BL,BM,BR = (randn(rng,ComplexF64,D,d,D)/3 for _ in 1:5)
        Cinv = randn(rng,ComplexF64,D,D)
        offsets = [4,1,3,2]
        gram = middle_position_gram(AL,Cinv,BL,BM,BR,span;middle_offsets=offsets)
        references = [triple_reference(AL,AR,Cinv,BL,BM,BR,span+1,1,1+a,span+1) for a in offsets]
        @test gram == gram'
        @test minimum(eigvals(Hermitian(gram))) >= -1e-12
        for a in eachindex(offsets),b in eachindex(offsets)
            @test gram[a,b] ≈ dense_overlap(references[a],references[b]) atol=1e-13 rtol=1e-11
        end
        norms = localized_triple_norms(AL,Cinv,BL,BM,BR,span;middle_offsets=offsets)
        @test norms ≈ real.(diag(gram))
        @test size(middle_position_gram(AL,Cinv,BL,BM,BR,span;middle_offsets=Int[])) == (0,0)
        @test isempty(localized_triple_norms(AL,Cinv,BL,BM,BR,span;middle_offsets=Int[]))
    end

    @testset "Product vacuum and definite excitation counts" begin
        L = 7
        three = [j in (1,4,7) ? down : up for j in 1:L]
        two = [j in (1,7) ? down : up for j in 1:L]
        grams = Dict(span=>middle_position_gram(up,C,down,down,down,span) for span in 2:L-1)
        @test all(gram -> gram == Matrix{ComplexF64}(I,size(gram)...),values(grams))
        blocks = three_particle_overlaps(three,up,up,C,down,down,down)
        projection = three_particle_weight(blocks,grams,1.0)
        @test projection.weight == 1
        @test projection.discarded_overlap_norm_squared == 0
        @test projection.maximum_retained_condition_number == 1
        @test projection.triple_count == binomial(L,3)
        @test three_particle_weight(blocks,grams,1.0;minimum_separation=3,gram_minimum_separation=1).weight == 1
        @test three_particle_weight(blocks,grams,1.0;minimum_separation=4,gram_minimum_separation=1).weight == 0
        two_blocks = three_particle_overlaps(two,up,up,C,down,down,down)
        @test all(block -> iszero(block.overlaps),values(two_blocks))
        @test three_particle_weight(two_blocks,grams,1.0).weight == 0
        mixed = sum_mps([two,three],[sqrt(0.6),im*sqrt(0.4)])
        mixed_blocks = three_particle_overlaps(mixed,up,up,C,down,down,down)
        @test three_particle_weight(mixed_blocks,grams,1.0).weight ≈ 0.4
        scale = 0.73cis(0.21)
        scaled = deepcopy(mixed)
        scaled[1] .*= scale
        scaled_blocks = three_particle_overlaps(scaled,up,up,C,down,down,down)
        @test three_particle_weight(scaled_blocks,grams,abs2(scale)).weight ≈ 0.4
        for outer in keys(mixed_blocks)
            @test scaled_blocks[outer].overlaps ≈ scale .* mixed_blocks[outer].overlaps
        end
    end

    @testset "Entangled canonical vacuum and an exact Gram projection" begin
        A,Cinv,BL,BM,BR = canonical_reference(rng)
        d = size(A,2)
        @test sum(A[:,s,:]'*A[:,s,:] for s in 1:d) ≈ I
        @test sum(A[:,s,:]*A[:,s,:]' for s in 1:d) ≈ I
        @test norm(sum(A[:,s,:]'*BL[:,s,:] for s in 1:d)) < 1e-13
        @test norm(sum(A[:,s,:]'*BM[:,s,:] for s in 1:d)) < 1e-13
        @test norm(sum(BR[:,s,:]*A[:,s,:]' for s in 1:d)) < 1e-13
        L,span = 5,4
        gram = middle_position_gram(A,Cinv,BL,BM,BR,span)
        @test norm(gram-Diagonal(diag(gram))) > 1e-5
        references = [triple_reference(A,A,Cinv,BL,BM,BR,L,1,m,L) for m in 2:L-1]
        coefficients = randn(rng,ComplexF64,L-2)
        state = sum_mps(references,coefficients)
        norm2 = real(dot(coefficients,gram*coefficients))
        @test norm2 ≈ real(dense_overlap(state,state)) atol=1e-12 rtol=1e-11
        blocks = three_particle_overlaps(state,A,A,Cinv,BL,BM,BR;first_sites=[1],last_sites=[L])
        @test blocks[(1,L)].overlaps ≈ gram*coefficients atol=1e-12 rtol=1e-11
        result = three_particle_weight(blocks,Dict(span=>gram),norm2)
        @test result.weight ≈ 1 atol=1e-11
        @test result.minimum_retained_rank == L-2
        @test result.discarded_overlap_norm_squared ≈ 0 atol=1e-12
        narrower = three_particle_weight(blocks,Dict(span=>gram),norm2;
            minimum_separation=2,gram_minimum_separation=1)
        @test 0 <= narrower.weight <= result.weight+1e-12

        # The outer-position gauge conditions make different blocks orthogonal.
        for (n,m,r) in ((1,2,3),(2,3,5),(1,3,4))
            other = triple_reference(A,A,Cinv,BL,BM,BR,L,n,m,r)
            for reference in references
                @test abs(dense_overlap(other,reference)) < 1e-12
            end
        end
        factorL,factorM,factorR = 2cis(0.4),0.7cis(-0.2),1.2cis(0.7)
        changed_blocks = three_particle_overlaps(state,A,A,Cinv,factorL*BL,factorM*BM,factorR*BR;
            first_sites=[1],last_sites=[L])
        changed_gram = middle_position_gram(A,Cinv,factorL*BL,factorM*BM,factorR*BR,span)
        @test changed_blocks[(1,L)].overlaps ≈ conj(factorL*factorM*factorR).*blocks[(1,L)].overlaps
        @test changed_gram ≈ abs2(factorL*factorM*factorR).*gram
        @test three_particle_weight(changed_blocks,Dict(span=>changed_gram),norm2).weight ≈ 1 atol=1e-11
    end

    @testset "Dependent middle states and cutoff diagnostics" begin
        span = 4
        gram = middle_position_gram(up,C,down,up,down,span)
        @test gram == ones(span-1,span-1)
        state = [down,up,up,up,down]
        blocks = three_particle_overlaps(state,up,up,C,down,up,down;first_sites=[1],last_sites=[5])
        for rtol in (1e-8,1e-10,1e-12)
            result = three_particle_weight(blocks,Dict(span=>gram),1.0;rtol)
            @test result.weight ≈ 1
            @test result.minimum_retained_rank == 1
            @test result.maximum_retained_condition_number ≈ 1
            @test result.discarded_overlap_norm_squared < 1e-28
        end
        zero_blocks = Dict((1,3)=>(middle_sites=[2],overlaps=ComplexF64[0]))
        result = three_particle_weight(zero_blocks,Dict(2=>zeros(1,1)),1.0)
        @test result.weight == 0 && result.maximum_retained_rank == 0
        @test three_particle_weight(Dict(),Dict(),1.0).weight == 0
    end

    @testset "Pair-triple cross overlaps and their phase convention" begin
        span = 5
        @test iszero(pair_triple_cross_gram(up,C,down,down,down,down,down,span))
        @test pair_triple_cross_gram(up,C,down,down,down,up,down,span) == ones(span-1)
        @test isempty(pair_triple_cross_gram(up,C,down,down,down,down,down,span;middle_offsets=Int[]))

        D,d = 2,2
        AL,BLpair,BRpair,BLtriple,BMtriple,BRtriple =
            (randn(rng,ComplexF64,D,d,D)/3 for _ in 1:6)
        Cinv = randn(rng,ComplexF64,D,D)
        offsets = [4,1,3]
        cross = pair_triple_cross_gram(AL,Cinv,BLpair,BRpair,BLtriple,BMtriple,BRtriple,
                                      span;middle_offsets=offsets)
        pair = [j == 1 ? inserted(BLpair,Cinv) : j == span+1 ? BRpair : AL for j in 1:span+1]
        for (i,a) in enumerate(offsets)
            triple = triple_reference(AL,AL,Cinv,BLtriple,BMtriple,BRtriple,span+1,1,a+1,span+1)
            @test cross[i] ≈ dense_overlap(triple,pair) atol=1e-13 rtol=1e-11
            @test conj(cross[i]) ≈ dense_overlap(pair,triple) atol=1e-13 rtol=1e-11
        end
        pL,pR,tL,tM,tR = 1.2cis(0.3),0.8cis(-0.5),1.1cis(0.7),0.6cis(-0.2),2cis(0.4)
        rescaled = pair_triple_cross_gram(AL,Cinv,pL*BLpair,pR*BRpair,
            tL*BLtriple,tM*BMtriple,tR*BRtriple,span;middle_offsets=offsets)
        @test rescaled ≈ (pL*pR*conj(tL*tM*tR)) .* cross atol=1e-13 rtol=1e-11

        # On an entangled vacuum the pair contamination can be nonzero.
        A,Ccenter,BL,BM,BR = canonical_reference(rng)
        span = 4
        gram = middle_position_gram(A,Ccenter,BL,BM,BR,span)
        cross = pair_triple_cross_gram(A,Ccenter,BL,BR,BL,BM,BR,span)
        pair = [j == 1 ? inserted(BL,Ccenter) : j == span+1 ? BR : A for j in 1:span+1]
        norm2 = real(dense_overlap(pair,pair))
        @test norm(cross) > 1e-5
        blocks = three_particle_overlaps(pair,A,A,Ccenter,BL,BM,BR;first_sites=[1],last_sites=[span+1])
        @test cross ≈ blocks[(1,span+1)].overlaps atol=1e-12 rtol=1e-11
        projection = three_particle_weight(blocks,Dict(span=>gram),norm2)
        @test projection.weight ≈ real(dot(cross,gram\cross))/norm2 atol=1e-12 rtol=1e-11
        @test 0 <= projection.weight <= 1

        @test_throws ArgumentError pair_triple_cross_gram(up,C,down,down,down,down,down,1)
        @test_throws ArgumentError pair_triple_cross_gram(up,C,down,down,down,down,down,4;middle_offsets=[1,1])
        @test_throws ArgumentError pair_triple_cross_gram(up,C,down,down,down,down,down,4;middle_offsets=[0])
        @test_throws ArgumentError pair_triple_cross_gram(up,C,down,down,down,down,down,4;middle_offsets=[4])
        @test_throws ArgumentError pair_triple_cross_gram(up,C,zeros(1,3,1),down,down,down,down,4)
        @test_throws ArgumentError pair_triple_cross_gram(up,C,down,NaN*down,down,down,down,4)
    end

    @testset "Invalid data and empty selections" begin
        state = [down,up,down]
        @test isempty(three_particle_overlaps(state,up,up,C,down,down,down;minimum_separation=2))
        @test isempty(three_particle_overlaps(state,up,up,C,down,down,down;middle_sites=Int[]))
        @test_throws ArgumentError three_particle_overlaps(state,up,up,C,down,down,down;minimum_separation=0)
        @test_throws ArgumentError three_particle_overlaps(state,up,up,C,down,down,down;first_sites=[0])
        @test_throws ArgumentError three_particle_overlaps(state,up,up,C,down,down,down;last_sites=[4])
        @test_throws ArgumentError three_particle_overlaps([],up,up,C,down,down,down)
        @test_throws ArgumentError three_particle_overlaps(state,up,up,C,down,NaN*down,down)
        @test_throws ArgumentError three_particle_overlaps([zeros(2,2,1),up],up,up,C,down,down,down)
        @test_throws ArgumentError middle_position_gram(up,C,down,down,down,1)
        @test_throws ArgumentError middle_position_gram(up,C,down,down,down,4;middle_offsets=[1,1])
        @test_throws ArgumentError middle_position_gram(up,C,down,down,down,4;middle_offsets=[0])
        @test_throws ArgumentError middle_position_gram(up,C,down,down,down,4;middle_offsets=[4])
        block = Dict((1,4)=>(middle_sites=[2,3],overlaps=ComplexF64[1,2]))
        @test_throws ArgumentError three_particle_weight(block,Dict(),1.0)
        @test_throws ArgumentError three_particle_weight(block,Dict(3=>ones(1,1)),1.0)
        @test_throws ArgumentError three_particle_weight(block,Dict(3=>[1 1;0 1]),1.0)
        @test_throws ArgumentError three_particle_weight(block,Dict(3=>[1 0;0 -1]),1.0)
        @test_throws ArgumentError three_particle_weight(block,Dict(3=>fill(NaN,2,2)),1.0)
        for norm2 in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError three_particle_weight(block,Dict(3=>Matrix{Float64}(I,2,2)),norm2)
        end
        @test_throws ArgumentError three_particle_weight(block,Dict(),1.0;minimum_separation=0)
        @test_throws ArgumentError three_particle_weight(block,Dict(),1.0;minimum_separation=1,gram_minimum_separation=2)
        @test_throws ArgumentError three_particle_weight(block,Dict(),1.0;rtol=-1)
        @test_throws ArgumentError three_particle_weight(block,Dict(),1.0;rtol=1)
    end
end

end # module IFTThreeParticleOverlapTests
