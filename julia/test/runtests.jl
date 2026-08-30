using Test
using LinearAlgebra
using QFTSimulations
using MPSKit
using TensorKit

@testset "Projected harmonic-oscillator operators" begin
    d = 10
    (; phi, phi2, pi2, phi4) = harmonic_oscillator_matrices(d)

    @test size(phi) == (d, d)
    @test all(LinearAlgebra.ishermitian, (phi, phi2, pi2, phi4))

    # Products of truncated operators agree with projected products except near
    # the Fock cutoff, where virtual transitions above |d-1⟩ are absent.
    p = zeros(ComplexF64, d, d)
    for i in 2:d
        value = sqrt((i - 1) / 2)
        p[i, i - 1] = im * value
        p[i - 1, i] = -im * value
    end
    low_quadratic = 1:(d - 1)
    low_quartic = 1:(d - 2)
    @test phi2[low_quadratic, low_quadratic] ≈
          (phi * phi)[low_quadratic, low_quadratic]
    @test pi2[low_quadratic, low_quadratic] ≈
          (p * p)[low_quadratic, low_quadratic]
    @test phi4[low_quartic, low_quartic] ≈
          (phi^4)[low_quartic, low_quartic]
    @test phi2[d, d] != (phi * phi)[d, d]

    number_identity = Diagonal(ComplexF64.(2 .* (0:(d - 1)) .+ 1))
    @test phi2 + pi2 ≈ number_identity

    one_level = harmonic_oscillator_matrices(1)
    @test one_level.phi == zeros(ComplexF64, 1, 1)
    @test one_level.phi2[1, 1] == 1 / 2
    @test one_level.pi2[1, 1] == 1 / 2
    @test one_level.phi4[1, 1] == 3 / 4

    @test_throws DomainError harmonic_oscillator_matrices(0)
    @test_throws DomainError harmonic_oscillator_matrices(-2)
    @test_throws ArgumentError harmonic_oscillator_matrices(true)
end

@testset "IFT scaling variable and exact free-fermion dispersion" begin
    @test ift_eta_latt(1.0, 0.01) == 0
    @test ift_eta_latt(1.06, 0.006) ≈ 0.92 atol=0.002 # Table VII
    @test ift_eta_latt(1.06, -0.006) == ift_eta_latt(1.06, 0.006)
    @test ift_eta_latt(0.9, 0.0) == -Inf
    @test ift_eta_latt(1.1, 0.0) == Inf

    gx = 1.25
    @test ift_free_fermion_dispersion(0, gx) ≈ 2abs(1 - gx)
    @test ift_free_fermion_dispersion(π, gx) ≈ 2(1 + gx)
    @test ift_free_fermion_dispersion(π / 3, 1) ≈ 4abs(sin(π / 6))
    @test ift_free_fermion_dispersion(-0.37, gx) ≈
          ift_free_fermion_dispersion(0.37, gx)

    @test_throws DomainError ift_eta_latt(1.0, 0.0)
    @test_throws DomainError ift_eta_latt(-0.1, 0.2)
    @test_throws DomainError ift_eta_latt(NaN, 0.2)
    @test_throws DomainError ift_free_fermion_dispersion(0.2, -1)
    @test_throws DomainError ift_free_fermion_dispersion(Inf, 1)
end

@testset "Free Schwinger lattice dispersion" begin
    mu = 0.3
    @test schwinger_free_lattice_dispersion(0, mu) ≈ mu
    @test schwinger_free_lattice_dispersion(π, mu) ≈ sqrt(mu^2 + 4)
    @test schwinger_free_lattice_dispersion(π / 2, 0; kappa=1) ≈ sqrt(2)
    @test schwinger_free_lattice_dispersion(0.4, mu; chi=2, kappa=0.25) ≈
          2sqrt(mu^2 + sin(0.4 / 2)^2)
    @test schwinger_free_lattice_dispersion(-0.4, mu) ≈
          schwinger_free_lattice_dispersion(0.4, mu)

    @test_throws DomainError schwinger_free_lattice_dispersion(0, -0.1)
    @test_throws DomainError schwinger_free_lattice_dispersion(0, mu; chi=0)
    @test_throws DomainError schwinger_free_lattice_dispersion(0, mu; kappa=-1)
    @test_throws DomainError schwinger_free_lattice_dispersion(NaN, mu)
end

@testset "Commensurate grids and Gaussian amplitudes" begin
    n = 8
    grid = commensurate_momentum_grid(n)
    spacing = 2π / n
    @test length(grid) == n
    @test first(grid) ≈ -π
    @test last(grid) ≈ π - spacing
    @test all(≈(spacing), diff(grid))
    @test last(grid) + spacing ≈ first(grid) + 2π
    @test grid[n ÷ 2 + 1] ≈ 0

    weights = gaussian_weights(grid, -π, 0.4)
    @test sum(abs2, weights) ≈ 1
    @test weights[2] ≈ weights[end] # shortest distance across the BZ edge
    @test argmax(weights) == 1

    raw = gaussian_weights(grid, π, 0.4; normalization=:none)
    @test raw[1] ≈ 1
    @test raw[2] ≈ raw[end]
    probabilities = gaussian_weights(grid, 0, 0.4; normalization=:l1)
    @test sum(probabilities) ≈ 1

    @test_throws DomainError commensurate_momentum_grid(1)
    @test_throws DomainError commensurate_momentum_grid(8; period=0)
    @test_throws DomainError commensurate_momentum_grid(8; period=Inf)
    @test_throws ArgumentError gaussian_weights(Float64[], 0, 1)
    @test_throws DomainError gaussian_weights(grid, 0, 0)
    @test_throws DomainError gaussian_weights(grid, Inf, 1)
    @test_throws DomainError gaussian_weights([0.0, NaN], 0, 1)
    @test_throws DomainError gaussian_weights(grid, 0, 1; period=-2π)
    @test_throws ArgumentError gaussian_weights(grid, 0, 1; normalization=:bad)
end

@testset "Exactly two disjoint packet excitations" begin
    # D=1 product vacuum |0⟩ and tangent tensor |1⟩ make particle number exact.
    AL = zeros(ComplexF64, 1, 2, 1)
    AL[1, 1, 1] = 1
    AR = copy(AL)
    B = zeros(ComplexF64, 1, 2, 1)
    B[1, 2, 1] = 1

    weighted_packet(weights) = [weight .* B for weight in weights]
    left_packet = weighted_packet([0.5, 1.0, 0.25])
    right_packet = weighted_packet([0.2, 0.75, 1.0])
    tensors = two_particle_packet_tensors(
        AL, AR, ones(ComplexF64, 1, 1), left_packet, right_packet
    )

    function amplitude(site_tensors, occupations)
        transfer = ones(ComplexF64, 1, size(first(site_tensors), 1))
        for (tensor, occupation) in zip(site_tensors, occupations)
            transfer *= tensor[:, occupation + 1, :]
        end
        return only(transfer)
    end

    norm2 = 0.0
    total_number = 0.0
    left_number = 0.0
    right_number = 0.0
    for occupations in Iterators.product(fill(0:1, length(tensors))...)
        probability = abs2(amplitude(tensors, occupations))
        norm2 += probability
        total_number += probability * sum(occupations)
        left_number += probability * sum(occupations[1:length(left_packet)])
        right_number += probability * sum(occupations[(length(left_packet) + 1):end])
    end
    @test norm2 > 0
    @test total_number / norm2 ≈ 2 atol=1e-12
    @test left_number / norm2 ≈ 1 atol=1e-12
    @test right_number / norm2 ≈ 1 atol=1e-12

    one_site = single_particle_packet_tensors(AL, AR, [B])
    @test length(one_site) == 1
    @test one_site[1] == B
    @test_throws ArgumentError single_particle_packet_tensors(AL, AR, typeof(B)[])
    @test_throws DimensionMismatch two_particle_packet_tensors(
        AL, AR, ones(ComplexF64, 2, 2), left_packet, right_packet
    )

    @testset "MPSKit WindowMPS integration" begin
        vacuum_tensor = TensorMap(AL, ℂ^1 ⊗ ℂ^2 ← ℂ^1)
        vacuum = InfiniteMPS([vacuum_tensor])
        C = vacuum.C[]
        Cinv = convert(Array, C \ id(domain(C)))
        dense_window = two_particle_packet_tensors(
            AL, AR, Cinv, left_packet, right_packet
        )
        window_tensors = [
            TensorMap(tensor, ℂ^(size(tensor, 1)) ⊗ ℂ^2 ← ℂ^(size(tensor, 3)))
            for tensor in dense_window
        ]
        state = WindowMPS(vacuum, window_tensors, vacuum)
        normalize!(state)

        number_operator = TensorMap(ComplexF64[0 0; 0 1], ℂ^2 ← ℂ^2)
        occupations = [
            real(expectation_value(state, site => number_operator))
            for site in eachindex(window_tensors)
        ]
        @test sum(occupations) ≈ 2 atol=1e-12
        @test sum(occupations[1:length(left_packet)]) ≈ 1 atol=1e-12
        @test sum(occupations[(length(left_packet) + 1):end]) ≈ 1 atol=1e-12
        @test norm(state) ≈ 1 atol=1e-12
    end
end
