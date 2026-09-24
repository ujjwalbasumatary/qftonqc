module IFTBoundStateSpectrumTests

using Test
include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "bound_state_spectrum.jl"))
using .IFTBoundStateSpectrum

@testset "Ising bound-state spectrum comparisons" begin
    @testset "Sampled two-particle thresholds" begin
        momenta = [0.0, 0.05, 0.1, 0.15, 0.2]
        energies = sqrt.(1 .+ momenta .^ 2)
        rest = sampled_threshold(0.0, momenta, energies)
        @test rest.energy == 2.0
        @test rest.left_momentum == 0.0
        @test rest.right_momentum == 0.0
        @test rest.count == 1
        moving = sampled_threshold(0.2, momenta, energies)
        @test moving.energy ≈ 2sqrt(1.01)
        @test moving.left_momentum == 0.1
        @test moving.right_momentum == 0.1
        @test moving.count == 5
        shifted = sampled_threshold(0.3, momenta, energies)
        @test shifted.energy ≈ 2sqrt(1.0225)
        @test shifted.left_momentum == 0.15
        @test shifted.right_momentum == 0.15

        # The helper must minimize the supplied data, not assume a convex dispersion.
        nonconvex = sampled_threshold(0.2, [0.0,0.1,0.2], [1.0,3.0,2.0])
        @test nonconvex.energy == 3.0
        @test nonconvex.left_momentum + nonconvex.right_momentum ≈ 0.2
        @test nonconvex.left_momentum != nonconvex.right_momentum

        @test_throws ArgumentError sampled_threshold(-0.1, momenta, energies)
        @test_throws ArgumentError sampled_threshold(NaN, momenta, energies)
        @test_throws ArgumentError sampled_threshold(0.0, Float64[], Float64[])
        @test_throws DimensionMismatch sampled_threshold(0.0, [0.0], [1.0,2.0])
        @test_throws ArgumentError sampled_threshold(0.0, [-0.1], [1.0])
        @test_throws ArgumentError sampled_threshold(0.0, [NaN], [1.0])
        @test_throws ArgumentError sampled_threshold(0.0, [0.0], [0.0])
        @test_throws ArgumentError sampled_threshold(0.0, [0.0], [Inf])
        @test_throws ArgumentError sampled_threshold(0.4, [0.0,0.1], [1.0,1.1])
    end

    @testset "Excitation solver arguments" begin
        @test_throws ArgumentError excitation_solutions(nothing,nothing,0.0,nothing; num=0)
        @test_throws ArgumentError excitation_solutions(nothing,nothing,0.0,nothing; tolerance=0.0)
        @test_throws ArgumentError excitation_solutions(nothing,nothing,0.0,nothing; tolerance=Inf)
        @test_throws ArgumentError excitation_solutions(nothing,nothing,NaN,nothing)
        @test_throws ArgumentError excitation_solutions(nothing,nothing,0.0,nothing; maxiter=0)
        @test_throws ArgumentError excitation_solutions(nothing,nothing,0.0,nothing; krylovdim=3)
        @test_throws ArgumentError ising_hamiltonian(Inf,0.0)
        @test_throws ArgumentError ising_hamiltonian(1.06,NaN)
    end
end

end
