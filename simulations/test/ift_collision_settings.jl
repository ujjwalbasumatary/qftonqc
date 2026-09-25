module IFTCollisionSettingsTests
using Test, TOML
include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "collision_from_settings.jl"))
using .IFTCollisionFromSettings
const Settings = IFTCollisionFromSettings

@testset "Lattice-time schedule and window-end checks" begin
    @test Settings.step_count(160.0, 0.1) == 1600
    @test Settings.step_count(5.0, 0.1) == 50
    @test Settings.step_count(0.0, 0.1) == 0
    for (time, dt) in ((-1, 0.1), (NaN, 0.1), (1, 0), (1, Inf), (1.05, 0.1))
        @test_throws ArgumentError Settings.step_count(time, dt)
    end
    @test Settings.edge_energy_fraction([0.1, -0.1, 1.0, -0.2, 0.1], 2, 2.0) ≈ 0.25
    @test Settings.edge_energy_fraction([0.0, 1.0, 0.0], 1, 1.0) == 0
    @test_throws ArgumentError Settings.edge_energy_fraction(ones(3), 2, 1.0)
    @test_throws ArgumentError Settings.edge_energy_fraction(ones(3), 1, 0.0)
    @test_throws ArgumentError Settings.edge_energy_fraction([0.0, NaN, 0.0], 1, 1.0)
end

@testset "The settings runner retains an initial state that reaches the edge bound" begin
    mktempdir() do directory
        config = Dict{String,Any}("random_seed"=>20260926,
            "final_time"=>0.2, "save_interval"=>0.1,
            "edge_width"=>5, "maximum_edge_fraction"=>1e-8,
            "parameters"=>Dict{String,Any}("bond_dimension"=>2,
                "evolution_bond_dimension"=>4, "length"=>12, "n_center"=>3,
                "kappa"=>0.38, "sigma"=>1.0, "time_step"=>0.1,
                "h_x"=>1.4, "h_z"=>0.05))
        settings_path = joinpath(directory, "settings.toml")
        open(settings_path, "w") do io
            TOML.print(io, config)
        end
        output = joinpath(directory, "collision")
        @test_throws Settings.WindowEdgeReached Settings.run(settings_path, output)
        @test length(readlines(joinpath(output, "checkpoints.csv"))) == 2
        @test isfile(joinpath(output, "collision_from_settings.jl"))
        @test !isfile(joinpath(output, "completed.toml"))
        saved_settings = TOML.parsefile(joinpath(output, "run-settings.toml"))
        @test saved_settings["parameters"]["total_time"] == 3
        @test saved_settings["parameters"]["save_every"] == 1
        @test_throws ArgumentError Settings.run(settings_path, output)
    end
end

@testset "A checkpoint exists before its observer can stop evolution" begin
    mktempdir() do directory
        args = Dict{String,Any}("bond_dimension"=>2, "evolution_bond_dimension"=>4,
            "length"=>12, "n_center"=>3, "kappa"=>0.38, "sigma"=>1.0,
            "total_time"=>3, "time_step"=>0.1, "h_x"=>1.4, "h_z"=>0.05,
            "save_every"=>1, "output_dir"=>directory)
        observed = Int[]
        callback = function(saved)
            @test isfile(saved.state_path)
            @test length(saved.energy_density) == 11
            @test length(saved.spin_density) == 12
            push!(observed, saved.step)
            saved.step == 1 && error("deliberate stop after one evolved checkpoint")
        end
        exception = try
            Settings.Collision.main(args; checkpoint_callback=callback)
            nothing
        catch error_value
            error_value
        end
        @test exception isa ErrorException
        @test occursin("deliberate stop", sprint(showerror, exception))
        @test observed == [0, 1]
        states = joinpath(directory, "states", only(readdir(joinpath(directory, "states"))))
        @test isfile(joinpath(states, "step_000000.jld2"))
        @test isfile(joinpath(states, "step_000001.jld2"))
        @test !isfile(joinpath(states, "step_000002.jld2"))
    end
end
end
