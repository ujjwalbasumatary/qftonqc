module IFTTwoParticleOverlapTests

using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "two_particle_overlap.jl"))
using .IFTTwoParticleOverlap

"Multiply the matrices for one physical configuration, leaving both external bonds open."
function configuration_matrix(tensors, configuration)
    result = Matrix(@view tensors[1][:, configuration[1], :])
    for site in 2:length(tensors)
        result = result * @view(tensors[site][:, configuration[site], :])
    end
    return result
end

"Directly enumerate physical configurations and contract the external bonds with identities."
function dense_overlap(reference, ket)
    d = size(first(ket), 2)
    L = length(ket)
    result = zero(ComplexF64)
    for configuration in Iterators.product(ntuple(_ -> 1:d, L)...)
        bra_matrix = configuration_matrix(reference, configuration)
        ket_matrix = configuration_matrix(ket, configuration)
        result += dot(bra_matrix, ket_matrix)
    end
    return result
end

function pair_reference(AL, AR, Cinv, BL, BR, L, n, m)
    first_tensor = similar(BL)
    for s in axes(BL, 2)
        first_tensor[:, s, :] = BL[:, s, :] * Cinv
    end
    return [site < n ? AL : site == n ? first_tensor : site < m ? AL : site == m ? BR : AR
            for site in 1:L]
end

function product_state(bits)
    return [reshape(bit == 0 ? ComplexF64[1, 0] : ComplexF64[0, 1], 1, 2, 1)
            for bit in bits]
end

"Represent a sum of product states by placing their labels on the internal bonds."
function sum_product_states(terms, coefficients)
    L = length(first(terms))
    number = length(terms)
    tensors = [zeros(ComplexF64, site == 1 ? 1 : number, 2, site == L ? 1 : number)
               for site in 1:L]
    for (term, (bits, coefficient)) in enumerate(zip(terms, coefficients))
        for site in 1:L
            l = site == 1 ? 1 : term
            r = site == L ? 1 : term
            tensors[site][l, bits[site]+1, r] = site == 1 ? coefficient : 1
        end
    end
    return tensors
end

@testset "Localized two-particle overlaps" begin
    rng = MersenneTwister(284107)
    up = reshape(ComplexF64[1, 0], 1, 2, 1)
    down = reshape(ComplexF64[0, 1], 1, 2, 1)
    center_inverse = ones(ComplexF64, 1, 1)

    @testset "Every overlap agrees with a dense contraction" begin
        for L in (5, 6)
            dimensions = [1; [isodd(site) ? 2 : 3 for site in 1:(L-1)]; 1]
            ket = [randn(rng, ComplexF64, dimensions[site], 2, dimensions[site+1]) / 2
                   for site in 1:L]
            overlaps = two_particle_overlaps(ket, up, up, center_inverse, down, down)
            @test size(overlaps) == (L, L)
            @test eltype(overlaps) == ComplexF64
            for n in 1:L, m in 1:L
                if m > n
                    reference = pair_reference(up, up, center_inverse, down, down, L, n, m)
                    @test overlaps[n, m] ≈ dense_overlap(reference, ket) atol=1e-13 rtol=1e-12
                else
                    @test iszero(overlaps[n, m])
                end
            end
        end
    end

    @testset "Independent complex bond contractions and pair norms" begin
        D, d, L = 2, 2, 5
        AL = randn(rng, ComplexF64, D, d, D) / 3
        AR = randn(rng, ComplexF64, D, d, D) / 3
        BL = randn(rng, ComplexF64, D, d, D) / 2
        BR = randn(rng, ComplexF64, D, d, D) / 2
        Cinv = randn(rng, ComplexF64, D, D)
        ket_dimensions = [D, 3, 2, 3, 2, D]
        ket = [randn(rng, ComplexF64, ket_dimensions[site], d, ket_dimensions[site+1]) / 3
               for site in 1:L]
        overlaps = two_particle_overlaps(ket, AL, AR, Cinv, BL, BR)
        for n in 1:(L-1), m in (n+1):L
            reference = pair_reference(AL, AR, Cinv, BL, BR, L, n, m)
            @test overlaps[n, m] ≈ dense_overlap(reference, ket) atol=1e-14 rtol=1e-12
        end
        norms = localized_pair_norms(AL, Cinv, BL, BR, L-1)
        for r in 1:(L-1)
            reference = pair_reference(AL, AR, Cinv, BL, BR, r+1, 1, r+1)
            @test norms[r] ≈ real(dense_overlap(reference, reference)) atol=1e-14 rtol=1e-12
        end
    end

    @testset "Two separated packets and normalization" begin
        L = 6
        left_packet = ComplexF64[1, 2im] / sqrt(5)
        right_packet = ComplexF64[2, -im] / sqrt(5)
        terms = Vector{Vector{Int}}()
        coefficients = ComplexF64[]
        for (i, n) in enumerate((1, 2)), (j, m) in enumerate((5, 6))
            bits = zeros(Int, L)
            bits[n] = bits[m] = 1
            push!(terms, bits)
            push!(coefficients, left_packet[i] * right_packet[j])
        end
        ket = sum_product_states(terms, coefficients)
        overlaps = two_particle_overlaps(ket, up, up, center_inverse, down, down)
        norms = localized_pair_norms(up, center_inverse, down, down, L-1)
        @test norms == ones(L-1)
        for (i, n) in enumerate((1, 2)), (j, m) in enumerate((5, 6))
            @test overlaps[n, m] ≈ left_packet[i] * right_packet[j]
        end
        @test two_particle_weight(overlaps, norms, 1.0) ≈ 1.0
        @test two_particle_weight(overlaps, norms, 1.0; minimum_separation=4) ≈ 1-abs2(left_packet[2]*right_packet[1])
        @test two_particle_weight(overlaps, norms, 1.0; minimum_separation=5) ≈ abs2(left_packet[1]*right_packet[2])
        @test iszero(two_particle_weight(overlaps, Float64[], 1.0; minimum_separation=L))

        scale = 0.83 * cis(0.713)
        scaled_ket = deepcopy(ket)
        scaled_ket[1] .*= scale
        scaled_overlaps = two_particle_overlaps(scaled_ket, up, up, center_inverse, down, down)
        @test scaled_overlaps ≈ scale * overlaps
        @test two_particle_weight(scaled_overlaps, norms, abs2(scale)) ≈ 1.0

        # Excitation phases belong to the bra and therefore enter conjugated.
        left_phase, right_phase = cis(0.23), cis(-0.71)
        rephased = two_particle_overlaps(ket, up, up, center_inverse, left_phase*down, right_phase*down)
        @test rephased ≈ conj(left_phase*right_phase) * overlaps

        # Norm factors remove a change in the size of either excitation tensor.
        rescaled = two_particle_overlaps(ket, up, up, center_inverse, 2down, 3down)
        rescaled_norms = localized_pair_norms(up, center_inverse, 2down, 3down, L-1)
        @test rescaled_norms == fill(36.0, L-1)
        @test two_particle_weight(rescaled, rescaled_norms, 1.0) ≈ 1.0
    end

    @testset "Vacuum, three-particle states, and a mixed particle count" begin
        L = 6
        vacuum = product_state(zeros(Int, L))
        three = product_state([1, 0, 1, 0, 1, 0])
        @test iszero(two_particle_overlaps(vacuum, up, up, center_inverse, down, down))
        @test iszero(two_particle_overlaps(three, up, up, center_inverse, down, down))
        mixed = sum_product_states([[0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0]], [sqrt(0.3), im*sqrt(0.7)])
        overlaps = two_particle_overlaps(mixed, up, up, center_inverse, down, down)
        @test two_particle_weight(overlaps, ones(L-1), 1.0) ≈ 0.3
    end

    @testset "Separation and the edges of the window" begin
        ket = product_state([1, 0, 0, 0, 1])
        all_pairs = two_particle_overlaps(ket, up, up, center_inverse, down, down)
        for gap in (1, 2, 4, 5, 8)
            selected = two_particle_overlaps(ket, up, up, center_inverse, down, down; minimum_separation=gap)
            for n in 1:5, m in 1:5
                @test selected[n, m] == (m-n >= gap ? all_pairs[n, m] : 0)
            end
        end
        @test all_pairs[1, 5] == 1
        @test iszero(two_particle_overlaps([up], up, up, center_inverse, down, down))
        @test iszero(two_particle_weight(zeros(1, 1), Float64[], 1.0))
    end

    @testset "Reject incompatible dimensions and undefined normalization" begin
        ket = product_state([1, 0, 1])
        for gap in (0, -1)
            @test_throws ArgumentError two_particle_overlaps(ket, up, up, center_inverse, down, down; minimum_separation=gap)
            @test_throws ArgumentError two_particle_weight(zeros(3, 3), ones(2), 1.0; minimum_separation=gap)
        end
        @test_throws ArgumentError two_particle_overlaps([], up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps(ket, up, up, ones(2, 2), down, down)
        @test_throws ArgumentError two_particle_overlaps(ket, up, zeros(1, 3, 1), center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps(ket, up, up, center_inverse, zeros(1, 2, 2), down)
        @test_throws ArgumentError two_particle_overlaps(ket, up, up, center_inverse, down, zeros(2, 2, 1))
        @test_throws ArgumentError two_particle_overlaps(ket, up[:, :, 1], up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps(ket, zeros(1, 2, 2), up, center_inverse, down, down)
        bad_bond = [up, zeros(2, 2, 1), up]
        @test_throws ArgumentError two_particle_overlaps(bad_bond, up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps([zeros(2, 2, 1)], up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps([zeros(1, 2, 2)], up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps([zeros(1, 3, 1)], up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps([fill(ComplexF64(NaN), 1, 2, 1)], up, up, center_inverse, down, down)
        @test_throws ArgumentError two_particle_overlaps(ket, up, up, center_inverse, Inf*down, down)
        @test_throws ArgumentError localized_pair_norms(up, center_inverse, down, down, 0)
        @test_throws ArgumentError localized_pair_norms(up, center_inverse, zero(down), down, 2)
        @test_throws ArgumentError two_particle_weight(zeros(2, 3), ones(2), 1.0)
        @test_throws ArgumentError two_particle_weight(fill(ComplexF64(Inf), 3, 3), ones(2), 1.0)
        @test_throws ArgumentError two_particle_weight(zeros(3, 3), [1.0], 1.0)
        @test_throws ArgumentError two_particle_weight(zeros(3, 3), [1.0, 0.0], 1.0)
        @test_throws ArgumentError two_particle_weight(zeros(3, 3), [1.0, NaN], 1.0)
        for state_norm2 in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError two_particle_weight(zeros(3, 3), ones(2), state_norm2)
        end
    end
end

end # module IFTTwoParticleOverlapTests
