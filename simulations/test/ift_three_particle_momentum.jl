module IFTThreeParticleMomentumTests

using Test, LinearAlgebra, Random

include(joinpath(@__DIR__,"..","models","ift","scripts","three_particle_momentum.jl"))
include(joinpath(@__DIR__,"..","models","ift","scripts","three_particle_overlap.jl"))
using .IFTThreeParticleMomentum, .IFTThreeParticleOverlap

function brute_position_fourier(ket,AL,AR,Cinv,BLs,BMs,BRs,kL,kM,kR;
                                gap=1,norm2=1.0)
    L = length(ket)
    result = zeros(ComplexF64,length(kL),length(kM),length(kR))
    for i in eachindex(kL),j in eachindex(kM),k in eachindex(kR)
        blocks = three_particle_overlaps(ket,AL,AR,Cinv,BLs[i],BMs[j],BRs[k];minimum_separation=gap)
        for ((n,r),block) in blocks,(m,value) in zip(block.middle_sites,block.overlaps)
            result[i,j,k] += cis(-kL[i]*n-kM[j]*m-kR[k]*r)*value
        end
    end
    return result/(L^(3/2)*sqrt(norm2))
end

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

function middle_insert(B,Cinv)
    result = similar(B)
    for s in axes(B,2)
        result[:,s,:] = B[:,s,:]*Cinv
    end
    return result
end

@testset "Three momentum-dependent excitation insertions" begin
    rng = MersenneTwister(384192)
    up = reshape(ComplexF64[1,0],1,2,1)
    down = reshape(ComplexF64[0,1],1,2,1)
    C = ones(ComplexF64,1,1)
    kL = [-0.71,-0.2,0.13]
    kM = [-0.09,0.17]
    kR = [0.2,0.55]

    @testset "Known product-state amplitude and Fourier normalization" begin
        L = 5
        ket = [down,up,down,up,down]
        banks = ([down for _ in grid] for grid in (kL,kM,kR))
        BLs,BMs,BRs = banks
        for gap in (1,2)
            values = three_particle_momentum_amplitudes(ket,up,up,C,BLs,BMs,BRs,kL,kM,kR;
                minimum_separation=gap)
            for i in eachindex(kL),j in eachindex(kM),k in eachindex(kR)
                @test values[i,j,k] ≈ cis(-kL[i]-3kM[j]-5kR[k])/L^(3/2) atol=1e-14
            end
        end
        @test iszero(three_particle_momentum_amplitudes(ket,up,up,C,BLs,BMs,BRs,kL,kM,kR;
            minimum_separation=3))
        pair = [down,up,up,up,down]
        @test iszero(three_particle_momentum_amplitudes(pair,up,up,C,BLs,BMs,BRs,kL,kM,kR))
        @test iszero(three_particle_momentum_amplitudes([up],up,up,C,BLs,BMs,BRs,kL,kM,kR))

        # A complete unitary momentum grid preserves the position amplitude norm.
        full_grid = (0:L-1).*(2pi/L)
        full_bank = [down for _ in full_grid]
        complete = three_particle_momentum_amplitudes(ket,up,up,C,full_bank,full_bank,full_bank,
            full_grid,full_grid,full_grid;minimum_separation=2)
        @test sum(abs2,complete) ≈ 1 atol=1e-13
        other_length = three_particle_momentum_amplitudes(ket,up,up,C,full_bank,full_bank,full_bank,
            full_grid,full_grid,full_grid;minimum_separation=2,fourier_length=8.0)
        @test other_length ≈ (L/8)^(3/2).*complete atol=1e-14
        @test sum(abs2,other_length) ≈ (L/8)^3 atol=1e-13
        explicit_length = three_particle_momentum_amplitudes(ket,up,up,C,full_bank,full_bank,full_bank,
            full_grid,full_grid,full_grid;minimum_separation=2,fourier_length=L)
        @test explicit_length == complete
    end

    @testset "Momentum-dependent tensors agree with explicit position sums" begin
        for D in (1,2),L in (5,6)
            AL,AR = (randn(rng,ComplexF64,D,2,D)/2 for _ in 1:2)
            Cinv = randn(rng,ComplexF64,D,D)
            dimensions = [D;[isodd(j) ? 2 : 3 for j in 1:L-1];D]
            ket = [randn(rng,ComplexF64,dimensions[j],2,dimensions[j+1])/2 for j in 1:L]
            BLs = [randn(rng,ComplexF64,D,2,D) for _ in kL]
            BMs = [randn(rng,ComplexF64,D,2,D) for _ in kM]
            BRs = [randn(rng,ComplexF64,D,2,D) for _ in kR]
            for gap in (1,2,4)
                expected = brute_position_fourier(ket,AL,AR,Cinv,BLs,BMs,BRs,kL,kM,kR;gap,norm2=2.3)
                result = three_particle_momentum_amplitudes(ket,AL,AR,Cinv,BLs,BMs,BRs,kL,kM,kR;
                    minimum_separation=gap,state_norm2=2.3)
                @test size(result) == (length(kL),length(kM),length(kR))
                @test result ≈ expected atol=1e-12 rtol=1e-11
            end
        end
    end

    @testset "Independent dense Hilbert-space check with complex D=2 tensors" begin
        D,d,L = 2,2,5
        AL,AR = (randn(rng,ComplexF64,D,d,D)/3 for _ in 1:2)
        Cinv = randn(rng,ComplexF64,D,D)
        ket = [randn(rng,ComplexF64,D,d,D)/3 for _ in 1:L]
        left_grid,middle_grid,right_grid = [-0.31,0.11],[-0.08,0.04],[0.17,0.42]
        BLs,BMs,BRs = ([randn(rng,ComplexF64,D,d,D)/2 for _ in grid]
                        for grid in (left_grid,middle_grid,right_grid))
        result = three_particle_momentum_amplitudes(ket,AL,AR,Cinv,BLs,BMs,BRs,
            left_grid,middle_grid,right_grid)
        for i in 1:2,j in 1:2,k in 1:2
            total = zero(ComplexF64)
            DL,DM = middle_insert(BLs[i],Cinv),middle_insert(BMs[j],Cinv)
            for n in 1:L-2,m in n+1:L-1,r in m+1:L
                reference = [site==n ? DL : site==m ? DM : site==r ? BRs[k] : site<r ? AL : AR
                             for site in 1:L]
                total += cis(-left_grid[i]*n-middle_grid[j]*m-right_grid[k]*r)*dense_overlap(reference,ket)
            end
            @test result[i,j,k] ≈ total/L^(3/2) atol=1e-13 rtol=1e-11
        end
    end

    @testset "Independent tensor phases, magnitudes, and state normalization" begin
        D,L = 2,6
        AL,AR = (randn(rng,ComplexF64,D,2,D)/2 for _ in 1:2)
        Cinv = randn(rng,ComplexF64,D,D)
        ket = [randn(rng,ComplexF64,D,2,D)/2 for _ in 1:L]
        BLs,BMs,BRs = ([randn(rng,ComplexF64,D,2,D)/2 for _ in grid] for grid in (kL,kM,kR))
        original = three_particle_momentum_amplitudes(ket,AL,AR,Cinv,BLs,BMs,BRs,kL,kM,kR)
        zL,zM,zR = ([complex(0.5+rand(rng))*cis(randn(rng)) for _ in grid] for grid in (kL,kM,kR))
        changed = three_particle_momentum_amplitudes(ket,AL,AR,Cinv,
            [z*B for (z,B) in zip(zL,BLs)],[z*B for (z,B) in zip(zM,BMs)],
            [z*B for (z,B) in zip(zR,BRs)],kL,kM,kR)
        for i in eachindex(kL),j in eachindex(kM),k in eachindex(kR)
            @test changed[i,j,k] ≈ conj(zL[i]*zM[j]*zR[k])*original[i,j,k] atol=1e-12 rtol=1e-11
        end
        scale = 0.72cis(-0.19)
        scaled_ket = deepcopy(ket)
        scaled_ket[1] .*= scale
        scaled = three_particle_momentum_amplitudes(scaled_ket,AL,AR,Cinv,BLs,BMs,BRs,kL,kM,kR;
            state_norm2=abs2(scale))
        @test scaled ≈ scale/abs(scale).*original atol=1e-12 rtol=1e-11

        # The tensor-grid ordering is preserved; no momentum sorting is assumed.
        shuffled = three_particle_momentum_amplitudes(ket,AL,AR,Cinv,reverse(BLs),BMs,reverse(BRs),
            reverse(kL),kM,reverse(kR))
        @test shuffled ≈ original[end:-1:1,:,end:-1:1] atol=1e-12 rtol=1e-11
    end

    @testset "Reject incompatible inputs" begin
        ket = [down,up,down,up,down]
        banks = [down]
        grid = [0.0]
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,
            grid,grid,grid;minimum_separation=0)
        for norm2 in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,
                grid,grid,grid;state_norm2=norm2)
        end
        for length_scale in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,
                grid,grid,grid;fourier_length=length_scale)
        end
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,[],grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,[Inf],grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,banks,[im],grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,[],banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,[zeros(1,3,1)],banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,C,banks,banks,[Inf*down],grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,zeros(1,3,1),C,banks,banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes(ket,up,up,ones(2,2),banks,banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes([],up,up,C,banks,banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes([zeros(2,2,1),up],up,up,C,banks,banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes([up,zeros(1,2,2)],up,up,C,banks,banks,banks,grid,grid,grid)
        @test_throws ArgumentError three_particle_momentum_amplitudes([up,zeros(2,2,1)],up,up,C,banks,banks,banks,grid,grid,grid)
    end
end

end # module IFTThreeParticleMomentumTests
