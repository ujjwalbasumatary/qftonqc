module IFTParticleBasisTests

using Test
using LinearAlgebra
using Random
using MPSKit
using TensorKit

include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "particle_basis.jl"))
using .IFTParticleBasis

@testset "Ising particle basis" begin
    rng = MersenneTwister(20260925)
    up = reshape(ComplexF64[1, 0], 1, 2, 1)
    up_tensor = TensorMap(up, ℂ^1 ⊗ ℂ^2 ← ℂ^1)
    product_vacuum = InfiniteMPS([up_tensor])
    random_tensor = TensorMap(randn(rng, ComplexF64, 2, 2, 2), ℂ^2 ⊗ ℂ^2 ← ℂ^2)
    entangled_vacuum = InfiniteMPS([random_tensor])

    @testset "Left and right gauge on $label" for (label, vacuum) in (
        ("a product vacuum", product_vacuum),
        ("a two-dimensional virtual space", entangled_vacuum),
    )
        arrays = vacuum_arrays(vacuum)
        D, d, _ = size(arrays.AL)
        @test size(arrays.AR) == (D, d, D)
        @test arrays.C * arrays.Cinv ≈ Matrix{ComplexF64}(I, D, D) atol=1e-11
        @test arrays.AL == convert(Array, vacuum.AL[1])
        @test arrays.AR == convert(Array, vacuum.AR[1])
        @test arrays.C == convert(Array, vacuum.C[1])

        for k in (0.0, -0.38, 0.38)
            random_values = (T, dims...) -> randn(rng, T, dims...)
            qp = LeftGaugedQP(random_values, vacuum; momentum=k)
            normalize!(qp)
            B = convert(Array, qp[1])[:, :, 1, :]
            original_B = copy(B)
            BR = right_gauge_tensor(vacuum, B, k)
            checks = gauge_residuals(vacuum, B, BR)
            @test checks.left_isometry < 1e-10
            @test checks.right_isometry < 1e-10
            @test checks.canonical_match < 1e-10
            @test checks.left_gauge < 1e-10
            @test checks.right_gauge < 1e-10
            @test checks.left_norm ≈ 1 atol=1e-10
            @test checks.right_norm ≈ checks.left_norm atol=2e-8 rtol=2e-8
            @test B == original_B
            @test norm(qp) ≈ norm(B) atol=1e-11

            right_qp = convert(RightGaugedQP, qp)
            @test convert(Array, right_qp[1])[:, :, 1, :] ≈ BR atol=1e-11
            @test norm(right_qp) ≈ norm(BR) atol=1e-11
            restored_qp = convert(LeftGaugedQP, right_qp)
            @test convert(Array, restored_qp[1])[:, :, 1, :] ≈ B atol=2e-8 rtol=2e-8

            factor = 1.7 * exp(0.731im)
            @test right_gauge_tensor(vacuum, factor .* B, k) ≈ factor .* BR atol=2e-8 rtol=2e-8
            @test_throws ArgumentError right_gauge_tensor(vacuum, arrays.AL, k)
            @test_throws ArgumentError right_gauge_tensor(vacuum, B, NaN)
            @test_throws ArgumentError right_gauge_tensor(vacuum, B, Inf)
            @test_throws ArgumentError right_gauge_tensor(vacuum, fill(ComplexF64(NaN), size(B)), k)
            @test_throws DimensionMismatch right_gauge_tensor(vacuum, zeros(ComplexF64, D + 1, d, D), k)
            @test_throws DimensionMismatch gauge_residuals(vacuum, B[:, 1:1, :], BR)
        end

        # Returned arrays must not share storage with the MPS.
        arrays.AL[1] += 3
        arrays.AR[1] += 4
        arrays.C[1] += 5
        fresh = vacuum_arrays(vacuum)
        @test arrays.AL != fresh.AL
        @test arrays.AR != fresh.AR
        @test arrays.C != fresh.C
    end

    @testset "Unit cell and excitation arguments" begin
        doubled_vacuum = InfiniteMPS([copy(up_tensor), copy(up_tensor)])
        B = reshape(ComplexF64[0, 1], 1, 2, 1)
        @test_throws ArgumentError vacuum_arrays(doubled_vacuum)
        @test_throws ArgumentError right_gauge_tensor(doubled_vacuum, B, 0.38)
        @test_throws ArgumentError excitation_tensors(doubled_vacuum, nothing, [0.38])
        @test_throws ArgumentError excitation_tensors(product_vacuum, nothing, Float64[])
        @test_throws ArgumentError excitation_tensors(product_vacuum, nothing, [NaN])
        @test_throws ArgumentError excitation_tensors(product_vacuum, nothing, [Inf])
        @test_throws ArgumentError excitation_tensors(product_vacuum, nothing, [0.38]; num=0)
        @test_throws ArgumentError excitation_tensors(product_vacuum, nothing, [0.38]; num=-1)
    end

    @testset "Reference tensor basis and complex coefficients" begin
        tensors = [randn(rng, ComplexF64, 2, 2, 2) for _ in 1:3]
        original = deepcopy(tensors)
        data = orthonormal_tensor_basis(tensors)
        @test length(data.basis) == 3
        @test size(data.coefficients) == (3, 3)
        @test tensors == original
        gram = [dot(A, B) for A in data.basis, B in data.basis]
        @test gram ≈ Matrix{ComplexF64}(I, 3, 3) atol=1e-12
        @test maximum(data.relative_errors) < 1e-12
        @test issorted(data.singular_values; rev=true)
        for j in eachindex(tensors)
            reconstructed = sum(data.coefficients[a, j] .* data.basis[a] for a in eachindex(data.basis))
            @test reconstructed ≈ tensors[j] atol=1e-12
            for a in eachindex(data.basis)
                @test data.coefficients[a, j] ≈ dot(data.basis[a], tensors[j]) atol=1e-12
            end
        end

        # A complex multiple of a tensor adds no new basis direction.
        factor = 2.3 * exp(0.219im)
        dependent = [tensors[1], factor .* tensors[1], zero(tensors[1])]
        reduced = orthonormal_tensor_basis(dependent)
        @test length(reduced.basis) == 1
        @test reduced.coefficients[1, 2] ≈ factor * reduced.coefficients[1, 1] atol=1e-12
        @test maximum(reduced.relative_errors) < 1e-12
        @test reduced.relative_errors[3] == 0

        single = orthonormal_tensor_basis([tensors[1]])
        @test length(single.basis) == 1
        @test single.coefficients[1, 1] .* single.basis[1] ≈ tensors[1] atol=1e-12

        # The reported omitted norm must agree with the discarded SVD direction.
        first_direction = reshape(ComplexF64[1, 0], 1, 2, 1)
        second_direction = reshape(ComplexF64[0, 1e-12], 1, 2, 1)
        truncated = orthonormal_tensor_basis([first_direction, second_direction]; rtol=1e-10)
        @test length(truncated.basis) == 1
        @test truncated.relative_errors ≈ [0, 1] atol=1e-12
        untruncated = orthonormal_tensor_basis([first_direction, second_direction]; rtol=0)
        @test length(untruncated.basis) == 2
        @test maximum(untruncated.relative_errors) < 1e-12

        @test_throws ArgumentError orthonormal_tensor_basis(Array{ComplexF64,3}[])
        @test_throws ArgumentError orthonormal_tensor_basis([zeros(ComplexF64, 1, 2, 1)])
        @test_throws ArgumentError orthonormal_tensor_basis([fill(ComplexF64(NaN), 1, 2, 1)])
        for rtol in (-1.0, 1.0, NaN, Inf)
            @test_throws ArgumentError orthonormal_tensor_basis(tensors; rtol)
        end
        @test_throws DimensionMismatch orthonormal_tensor_basis([zeros(2, 2)])
        @test_throws DimensionMismatch orthonormal_tensor_basis([tensors[1], zeros(ComplexF64, 1, 2, 1)])
    end
end

end
