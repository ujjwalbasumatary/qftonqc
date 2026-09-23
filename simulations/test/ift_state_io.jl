module IFTStateIOTests

using Test
using LinearAlgebra
using MPSKit
using TensorKit
using JLD2

include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "state_io.jl"))
using .IFTStateIO

# This object makes JLD2 stop after opening the temporary file, so the test
# below can check that an unfinished write does not leave a state file behind.
struct UnwritableReference end
JLD2.writeas(::Type{UnwritableReference}) = error("deliberate test write failure")

@testset "Saving Ising collision states" begin
    @testset "Which time steps are saved" begin
        @test [step for step in 0:12 if should_save_state(step, 12, 5)] == [0, 5, 10, 12]
        @test [step for step in 0:10 if should_save_state(step, 10, 5)] == [0, 5, 10]
        @test [step for step in 0:3 if should_save_state(step, 3, 1)] == [0, 1, 2, 3]
        @test [step for step in 0:3 if should_save_state(step, 3, 10)] == [0, 3]
        @test [step for step in 0:3 if should_save_state(step, 3, 0)] == [0, 3]
        for interval in (0, 1, 10)
            @test should_save_state(0, 0, interval)
        end
        @test_throws ArgumentError should_save_state(0, -1, 1)
        @test_throws ArgumentError should_save_state(-1, 10, 1)
        @test_throws ArgumentError should_save_state(11, 10, 1)
        @test_throws ArgumentError should_save_state(0, 10, -1)
    end

    @testset "Sources and Julia environment saved with the state" begin
        mktempdir() do directory
            project_directory = joinpath(directory, "simulations")
            mkpath(project_directory)
            project_text = "name = \"StateIOTest\"\n"
            manifest_text = "julia_version = \"$(VERSION)\"\nmanifest_format = \"2.0\"\n"
            source_text = "# A source file stored with the Ising state.\n"
            write(joinpath(project_directory, "Project.toml"), project_text)
            write(joinpath(project_directory, "Manifest.toml"), manifest_text)
            source_path = joinpath(project_directory, "prepare.jl")
            write(source_path, source_text)

            provenance = state_provenance(project_directory, [source_path])
            @test provenance["project_toml"] == project_text
            @test provenance["manifest_toml"] == manifest_text
            @test provenance["julia_version"] == string(VERSION)
            @test provenance["package_versions"]["MPSKit"] == string(pkgversion(MPSKit))
            @test provenance["package_versions"]["TensorKit"] == string(pkgversion(TensorKit))
            @test provenance["package_versions"]["JLD2"] == string(pkgversion(JLD2))
            @test provenance["source_files"] == Dict(joinpath("simulations", "prepare.jl") => source_text)
            @test provenance["git_commit"] === nothing
            @test provenance["git_dirty"] === nothing

            project_without_manifest = joinpath(directory, "without_manifest")
            mkpath(project_without_manifest)
            write(joinpath(project_without_manifest, "Project.toml"), project_text)
            @test state_provenance(project_without_manifest, String[])["manifest_toml"] === nothing
        end
    end

    @testset "The finite window and its infinite boundaries survive reloading" begin
        up = reshape(ComplexF64[1, 0], 1, 2, 1)
        plus = reshape(ComplexF64[1, 1] / sqrt(2), 1, 2, 1)
        vacuum = InfiniteMPS([TensorMap(up, ℂ^1 ⊗ ℂ^2 ← ℂ^1)])
        tensors = [TensorMap(site == 2 ? plus : up, ℂ^1 ⊗ ℂ^2 ← ℂ^1) for site in 1:4]
        state = WindowMPS(vacuum, tensors, vacuum)
        normalize!(state)
        σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
        σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
        hamiltonian = InfiniteMPOHamiltonian(
            PeriodicVector([ℂ^2]), 1 => -0.3 * σz, (1, 2) => -σz ⊗ σz
        )
        excitation_tensors = [reshape(ComplexF64[0, 1], 1, 2, 1)]
        reference = Dict{String, Any}(
            "vacuum" => vacuum,
            "hamiltonian" => hamiltonian,
            "excitation_tensors" => excitation_tensors,
            "momenta" => [0.3, -0.3],
            "energies" => [1.0, 1.0],
            "vacuum_energy" => -1.3,
            "vacuum_spin" => 1.0,
        )
        parameters = Dict{String, Any}("length" => 4, "time_step" => 0.1)
        provenance = Dict{String, Any}("julia_version" => string(VERSION))
        energy_density = [0.1, 0.2, -0.1]
        spin_density = [real(expectation_value(state, site => σz)) for site in 1:4]
        common = (; dt=0.1, parameters, reference, energy_density, spin_density, provenance)

        mktempdir() do directory
            state_directory = joinpath(directory, "states")
            initial_path = joinpath(state_directory, "step_000000.jld2")
            @test !isdir(state_directory)
            @test save_ift_state(initial_path, state; step=0, common...) == abspath(initial_path)
            @test readdir(state_directory) == ["step_000000.jld2"]
            initial = load_ift_state(initial_path)
            @test initial["format_version"] == 1
            @test initial["step"] == 0
            @test initial["time"] == 0.0
            @test initial["state"] isa WindowMPS
            @test typeof(initial["state"]) == typeof(state)
            @test length(initial["state"]) == 4
            @test norm(initial["state"]) ≈ norm(state) atol=1e-12
            @test initial["state_norm"] ≈ norm(state) atol=1e-12
            @test norm(initial["state"].left_gs) ≈ 1 atol=1e-12
            @test norm(initial["state"].right_gs) ≈ 1 atol=1e-12
            @test expectation_value(initial["state"].left_gs, 1 => σz) ≈ 1 atol=1e-12
            @test expectation_value(initial["state"].right_gs, 1 => σz) ≈ 1 atol=1e-12
            for site in 1:4
                @test expectation_value(initial["state"], site => σx) ≈
                      expectation_value(state, site => σx) atol=1e-12
                @test expectation_value(initial["state"], site => σz) ≈
                      expectation_value(state, site => σz) atol=1e-12
            end
            @test initial["parameters"] == parameters
            @test initial["energy_density"] == energy_density
            @test initial["spin_density"] == spin_density
            @test initial["provenance"] == provenance
            @test initial["reference"]["vacuum"] isa InfiniteMPS
            @test norm(initial["reference"]["vacuum"]) ≈ 1 atol=1e-12
            @test initial["reference"]["hamiltonian"] isa InfiniteMPOHamiltonian
            @test initial["reference"]["excitation_tensors"] == excitation_tensors
            @test initial["reference"]["momenta"] == reference["momenta"]
            @test initial["reference"]["energies"] == reference["energies"]
            @test initial["reference"]["vacuum_energy"] == reference["vacuum_energy"]
            @test initial["reference"]["vacuum_spin"] == reference["vacuum_spin"]

            # Saving a later state must retain its norm rather than normalize it.
            later_path = joinpath(state_directory, "step_000007.jld2")
            scaled_state = 1.7 * state
            save_ift_state(later_path, scaled_state; step=7, common...)
            later = load_ift_state(later_path)
            @test later["step"] == 7
            @test later["time"] == 7 * common.dt
            @test later["state_norm"] ≈ 1.7 atol=1e-12
            @test norm(later["state"]) ≈ 1.7 atol=1e-12
            @test norm(state) ≈ 1 atol=1e-12

            original_bytes = read(initial_path)
            existing_files = readdir(state_directory)
            @test_throws ArgumentError save_ift_state(initial_path, state; step=1, common...)
            @test read(initial_path) == original_bytes
            @test readdir(state_directory) == existing_files

            invalid_path = joinpath(state_directory, "invalid.jld2")
            @test_throws ArgumentError save_ift_state(invalid_path, state; common..., step=-1)
            for dt in (0.0, -0.1, NaN, Inf, -Inf)
                @test_throws ArgumentError save_ift_state(invalid_path, state; common..., step=0, dt)
            end
            @test_throws DimensionMismatch save_ift_state(
                invalid_path, state; common..., step=0, energy_density=zeros(4)
            )
            @test_throws DimensionMismatch save_ift_state(
                invalid_path, state; common..., step=0, spin_density=zeros(3)
            )
            @test !ispath(invalid_path)
            @test readdir(state_directory) == existing_files

            @test_throws ErrorException save_ift_state(
                invalid_path, state; common..., step=0, reference=UnwritableReference()
            )
            @test !ispath(invalid_path)
            @test readdir(state_directory) == existing_files
        end
    end

    @testset "Unrecognized state files" begin
        mktempdir() do directory
            for (name, entries) in (
                ("wrong_version", Dict("format_version" => 2, "state" => zeros(2))),
                ("missing_version", Dict("state" => zeros(2))),
                ("wrong_state", Dict("format_version" => 1, "state" => zeros(2))),
                ("missing_state", Dict("format_version" => 1)),
            )
                path = joinpath(directory, "$name.jld2")
                jldopen(path, "w") do file
                    for (key, value) in entries
                        file[key] = value
                    end
                end
                @test_throws ArgumentError load_ift_state(path)
            end
        end
    end
end

end
