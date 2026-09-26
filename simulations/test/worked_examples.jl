module WorkedExampleTests

using Test, LinearAlgebra, JLD2
include(joinpath(@__DIR__, "..", "examples", "ising.jl"))
include(joinpath(@__DIR__, "..", "examples", "overlaps.jl"))
include(joinpath(@__DIR__, "..", "examples", "schwinger.jl"))

@testset "Worked examples" begin
    mktempdir() do root
        @testset "Exact excitation counts and Gram matrix" begin
            for p in (0.0, 0.3, 1.0)
                result = OverlapExample.run_example(; root, probability_three=p)
                @test result.weight_two ≈ 1-p atol=1e-12
                @test result.weight_three ≈ p atol=1e-12
                @test result.naive_sum ≈ 1.5
                @test result.projection ≈ 1.0
                @test isfile(joinpath(result.directory, "weights.toml"))
            end
            @test_throws ArgumentError OverlapExample.run_example(; root, probability_three=-0.1)
        end
        @testset "Ising dispersion" begin
            result = IsingExamples.spectrum(; root)
            @test size(result.energies) == (3, 2)
            @test maximum(abs, result.energies[:, 2] - result.exact) < 1e-3
            @test all(isfinite, result.lengths)
            @test isfile(joinpath(result.directory, "spectrum.csv"))
        end
        @testset "Initial packets and saved observables" begin
            result = IsingExamples.packets(; root)
            saved = IsingExamples.load_ift_state(result.path)
            @test length(saved["state"]) == 64
            @test saved["time"] == 0
            @test saved["state_norm"] ≈ 1 atol=1e-10
            @test saved["energy_density"] ≈ result.measured.energy
            @test sum(result.measured.energy) > 0
            @test maximum(abs, result.measured.spin) < 1e-3
        end
        @testset "Short evolution and reloading" begin
            result = IsingExamples.evolution(; root, final_time=0.2, save_interval=0.1)
            @test result.times ≈ [0, 0.1, 0.2]
            @test length(result.paths) == 3
            @test size(result.energy) == (3, 63)
            @test maximum(abs, result.norm_squared .- 1) < 1e-3
            @test abs(result.reload_difference) < 1e-10
            @test load(result.data_path, "times") == result.times
            @test IsingExamples.load_ift_state(last(result.paths))["time"] ≈ 0.2
            @test_throws ArgumentError IsingExamples.evolution(; root, dt=0.0)
            @test_throws ArgumentError IsingExamples.evolution(; root, final_time=0.15)
            @test_throws ArgumentError IsingExamples.evolution(; root, save_interval=0.15)
        end
        @testset "Schwinger onsite cutoff and source quench" begin
            result = SchwingerExample.run_example(; root, final_time=0.15, dt=0.1)
            @test result.times ≈ [0.0, 0.1, 0.15]
            @test size(result.field) == (3, 12)
            @test all(isfinite, result.field)
            @test maximum(abs, result.basis_difference) < 1e-3
            @test maximum(abs, result.norm_squared .- 1) < 1e-8
            @test abs(result.field[end, 6] - result.field[1, 6]) > 1e-5
            @test isfile(result.data_path)
        end
    end
end

end
