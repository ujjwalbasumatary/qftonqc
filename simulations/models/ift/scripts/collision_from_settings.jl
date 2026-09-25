"""
    IFTCollisionFromSettings

Run a fresh Ising collision from a TOML file. The file specifies lattice time,
packet parameters, a random seed, and an optional bound on the energy near the
window ends. Evolution uses `collide_fixed_momentum.jl` without changing its
Hamiltonian, packet construction, or two-site TDVP update.
"""
module IFTCollisionFromSettings

using Dates, LinearAlgebra, Printf, Random, TOML

module Collision
include(joinpath(@__DIR__, "collide_fixed_momentum.jl"))
end

"A checkpoint was saved, but its energy near the window ends exceeded the requested bound."
struct WindowEdgeReached <: Exception
    time::Float64
    fraction::Float64
    path::String
end
Base.showerror(io::IO, e::WindowEdgeReached) = print(io,
    "At lattice time ", e.time, ", the edge-energy fraction reached ",
    e.fraction, ". Evolution stopped after saving ", e.path)

"""
    step_count(time, dt)

Convert a nonnegative lattice time to a step count. Both inputs must be finite,
`dt` must be positive, and the requested time must lie on the time-step grid.
"""
function step_count(time::Real, dt::Real)
    isfinite(time) && time >= 0 || throw(ArgumentError("time must be finite and nonnegative"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    ratio = time / dt
    isfinite(ratio) && ratio < typemax(Int) || throw(ArgumentError("too many time steps"))
    steps = round(Int, ratio)
    isapprox(steps * dt, time; atol=1e-10 * max(1, time), rtol=0) ||
        throw(ArgumentError("time must be a multiple of dt"))
    return steps
end

"""
    edge_energy_fraction(energy, width, initial_energy)

Sum the absolute bond-energy differences in the first and last `width` bonds,
and divide by the initial summed energy above the vacuum. The strips must not
overlap and the denominator must be finite and positive. Absolute values keep
positive and negative contributions from cancelling in this check. This
quantity checks the window ends; it is not a particle probability or a bound
on the error in the evolved state.
"""
function edge_energy_fraction(energy, width::Integer, initial_energy::Real)
    0 < 2width <= length(energy) || throw(ArgumentError("edge strips overlap or are empty"))
    isfinite(initial_energy) && initial_energy > 0 || throw(ArgumentError("invalid initial energy"))
    all(isfinite, energy) || throw(ArgumentError("nonfinite bond energy"))
    return (sum(abs, view(energy, 1:width)) +
            sum(abs, view(energy, (length(energy)-width+1):length(energy)))) / initial_energy
end

"""
    run(settings_path, output_directory)

Read `[parameters]` using the names accepted by the collision program. Top-level
`final_time` and `save_interval` are lattice times and replace its step-count
arguments. `random_seed` is required. The output directory must be new or empty.

At every saved state, write the norm squared, summed bond energy, relative
energy change, and edge-energy fraction to `checkpoints.csv`. If that fraction
exceeds `maximum_edge_fraction` (default 0.01), throw `WindowEdgeReached` after
the state has been saved. `edge_width` defaults to 32 bonds on each side. The
exception leaves the latest checkpoint and this CSV available for inspection.

The observer also honours a `STOP_AFTER_CHECKPOINT` file in the output directory.
This permits stopping only this calculation after a completed state file; it
does not remove, overwrite, normalize, or restart any saved state. No automatic
continuation is attempted. A successful full evolution writes `completed.toml`.
"""
function run(settings_path::AbstractString, output_directory::AbstractString)
    settings = TOML.parsefile(settings_path)
    output = abspath(output_directory)
    isdir(output) && !isempty(readdir(output)) &&
        throw(ArgumentError("the output directory is not empty: $output"))
    mkpath(output)
    parameters = Dict{String,Any}(settings["parameters"])
    dt = parameters["time_step"]
    parameters["total_time"] = step_count(settings["final_time"], dt) + 1
    parameters["save_every"] = step_count(settings["save_interval"], dt)
    parameters["output_dir"] = output
    edge_width = get(settings, "edge_width", 32)
    0 < 2edge_width <= parameters["length"] - 1 || throw(ArgumentError("invalid edge width"))
    maximum_edge_fraction = get(settings, "maximum_edge_fraction", 0.01)
    0 < maximum_edge_fraction < 1 || throw(ArgumentError("invalid edge fraction limit"))
    settings["parameters"] = parameters
    settings["julia_version"] = string(VERSION)
    settings["julia_threads"] = Threads.nthreads()
    settings["blas_threads"] = BLAS.get_num_threads()
    settings["started_at"] = string(now())
    open(joinpath(output, "run-settings.toml"), "w") do io
        TOML.print(io, settings; sorted=true)
    end
    cp(@__FILE__, joinpath(output, "collision_from_settings.jl"))
    initial_energy = Ref(NaN)
    open(joinpath(output, "checkpoints.csv"), "w") do io
        println(io, "time,step,norm_squared,internal_energy,relative_energy_change,edge_energy_fraction,state_path")
    end
    function observe(saved)
        energy = sum(saved.energy_density)
        saved.step == 0 && (initial_energy[] = energy)
        fraction = edge_energy_fraction(saved.energy_density, edge_width, initial_energy[])
        norm2 = abs2(norm(saved.state))
        isfinite(norm2) && norm2 > 0 || error("invalid state norm")
        drift = (energy - initial_energy[]) / initial_energy[]
        open(joinpath(output, "checkpoints.csv"), "a") do io
            println(io, join((saved.time, saved.step, norm2, energy, drift,
                              fraction, relpath(saved.state_path, output)), ','))
        end
        @printf("Saved t = %.1f; norm² = %.10f; energy change = %+.6f%%; edge energy = %.6f%%\n",
                saved.time, norm2, 100drift, 100fraction)
        flush(stdout)
        if fraction > maximum_edge_fraction
            throw(WindowEdgeReached(saved.time, fraction, saved.state_path))
        end
        isfile(joinpath(output, "STOP_AFTER_CHECKPOINT")) &&
            error("Stop requested; last saved lattice time is $(saved.time), state $(saved.state_path)")
        return nothing
    end
    println("Julia ", VERSION, "; computation threads = ", Threads.nthreads(),
            "; BLAS threads = ", BLAS.get_num_threads())
    println("Final lattice time = ", settings["final_time"],
            "; states saved every ", settings["save_interval"])
    flush(stdout)
    Random.seed!(settings["random_seed"])
    result = Collision.main(parameters; checkpoint_callback=observe)
    open(joinpath(output, "completed.toml"), "w") do io
        TOML.print(io, Dict("finished_at" => string(now()),
            "final_time" => settings["final_time"], "state_directory" => result.state_directory))
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: julia --project=simulations collision_from_settings.jl SETTINGS.toml OUTPUT_DIRECTORY")
    run(ARGS[1], ARGS[2])
end

end
