"""
Small Ising calculations used in the worked examples.

From the repository root, run `julia --project=simulations
simulations/examples/ising.jl spectrum`, replacing `spectrum` with `packets`
or `evolution` for the other examples. Each command makes a new directory in
`results/examples/`; a second argument can select a different parent directory.
Including this file defines the functions without running a calculation.
"""
module IsingExamples

using LinearAlgebra, Random, Printf, TOML, JLD2
using MPSKit, TensorKit
using QFTSimulations: ift_free_fermion_dispersion

module Collision
include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "collide_fixed_momentum.jl"))
end
using .Collision.IFTStateIO

const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "..", "results", "examples"))

"Create a new example directory, leaving the outputs of previous runs in place."
function output_directory(label; root=OUTPUT_ROOT)
    mkpath(root)
    directory = mktempdir(abspath(root); prefix=label * "-", cleanup=false)
    println("Outputs: ", directory)
    flush(stdout)
    return directory
end

"""
    spectrum(; root=OUTPUT_ROOT)

Find the uniform vacuum at `h_x=1.5`, `h_z=0` with bond dimensions 4 and 8,
then calculate the lowest tangent-space excitation at momenta 0, 0.4, and
0.8. Compare each energy with the exact transverse-field Ising dispersion.
The random seed is reset before each vacuum calculation. Write the energies,
correlation lengths, and parameters to a fresh directory and return them.
No finite chain or real-time evolution is involved.
"""
function spectrum(; root=OUTPUT_ROOT)
    BLAS.set_num_threads(1)
    directory = output_directory("ising-spectrum"; root)
    hx, hz = 1.5, 0.0
    momenta = [0.0, 0.4, 0.8]
    dimensions = [4, 8]
    ham = Collision.get_ham(hx, hz)
    exact = ift_free_fermion_dispersion.(momenta, hx)
    energies = zeros(length(momenta), length(dimensions))
    lengths = zeros(length(dimensions))
    for (column, D) in enumerate(dimensions)
        Random.seed!(20260926)
        vacuum = Collision.prep_gs(D, ham)
        lengths[column] = correlation_length(vacuum)
        values, _ = Collision.get_QPstate(vacuum, ham, momenta)
        energies[:, column] = real.(values[:, 1])
        @printf("D = %d; correlation length = %.6f sites\n", D, lengths[column])
        for row in eachindex(momenta)
            @printf("  p = %.1f   E_MPS = %.8f   E_exact = %.8f   difference = %+.3e\n",
                momenta[row], energies[row, column], exact[row], energies[row, column]-exact[row])
        end
        flush(stdout)
    end
    open(joinpath(directory, "spectrum.csv"), "w") do io
        println(io, "bond_dimension,momentum,energy,exact_energy,difference")
        for (column, D) in enumerate(dimensions), row in eachindex(momenta)
            println(io, join((D, momenta[row], energies[row, column], exact[row],
                energies[row, column]-exact[row]), ','))
        end
    end
    open(joinpath(directory, "parameters.toml"), "w") do io
        TOML.print(io, Dict("h_x"=>hx, "h_z"=>hz, "random_seed"=>20260926,
            "bond_dimensions"=>dimensions, "correlation_lengths"=>lengths,
            "momenta"=>momenta, "julia_version"=>string(VERSION)))
    end
    return (; directory, momenta, dimensions, energies, exact, lengths)
end

"""
    prepare_packets()

Prepare a normalized 64-site `WindowMPS` above the uniform Ising vacuum at
`h_x=1.5`, `h_z=0`, with vacuum bond dimension 4. Packet centres are 16 and
48, central momenta are +0.6 and -0.6, and the amplitude width is sigma=6
in `exp(-(n-n_center)^2/sigma^2)`. The Gaussian probability width is sigma/2.

Use the Hamiltonian, vacuum solver, excitation solver, and packet assembly
from `collide_fixed_momentum.jl`. Return the state, Hamiltonian, vacuum,
parameters, and reference data needed to measure and save the state. The
stored time step is 0.1 and the proposed evolution bond dimension is 12.
The function does not evolve the state or write files.
"""
function prepare_packets()
    BLAS.set_num_threads(1)
    Random.seed!(20260926)
    parameters = Dict{String,Any}("h_x"=>1.5, "h_z"=>0.0,
        "length"=>64, "bond_dimension"=>4, "evolution_bond_dimension"=>12,
        "n_center"=>16, "kappa"=>0.6, "sigma"=>6.0, "time_step"=>0.1,
        "random_seed"=>20260926)
    ham = Collision.get_ham(parameters["h_x"], parameters["h_z"])
    vacuum = Collision.prep_gs(parameters["bond_dimension"], ham)
    momenta = [parameters["kappa"], -parameters["kappa"]]
    energies, excitations = Collision.get_QPstate(vacuum, ham, momenta)
    B = Collision.get_B_tensor_list(excitations)
    tensors = Collision.create_stacked_tensor(vacuum, B, parameters["length"],
        parameters["n_center"], parameters["kappa"], parameters["sigma"])
    state = WindowMPS(vacuum, tensors)
    normalize!(state)
    sx, _, sz = Collision.get_ops()
    identity = TensorMap(Matrix{ComplexF64}(I, 2, 2), ℂ^2 ← ℂ^2)
    field = parameters["h_x"] * sx + parameters["h_z"] * sz
    bond = -sz ⊗ sz - (field ⊗ identity + identity ⊗ field) / 2
    vacuum_energy = real(expectation_value(vacuum, (1, 2) => bond))
    vacuum_spin = real(expectation_value(vacuum, 1 => sz))
    reference = Dict{String,Any}("vacuum"=>vacuum, "hamiltonian"=>ham,
        "momenta"=>momenta, "excitation_energies"=>energies,
        "excitation_tensors"=>B, "correlation_length"=>correlation_length(vacuum),
        "vacuum_energy_density"=>fill(vacuum_energy, length(state)-1),
        "vacuum_spin_density"=>fill(vacuum_spin, length(state)))
    return (; state, ham, vacuum, parameters, reference, bond, sz)
end

"""
    measure(state, preparation)

Return the energy on each of the L-1 internal bonds and the longitudinal spin
on each of L sites, minus their values in the uniform vacuum. `preparation`
is returned by `prepare_packets`. Neither the state nor its norm is changed.
"""
function measure(state, preparation)
    energy = [real(expectation_value(state, (n, n+1) => preparation.bond))
        for n in 1:(length(state)-1)] .- preparation.reference["vacuum_energy_density"]
    spin = [real(expectation_value(state, n => preparation.sz))
        for n in 1:length(state)] .- preparation.reference["vacuum_spin_density"]
    return (; energy, spin)
end

"""
    calculation_sources()

Record the example, the reused collision and state-saving functions, and
the shared packet construction, together with the installed Julia packages.
These data are embedded in each saved MPS file by `save_ift_state`.
"""
function calculation_sources()
    project = normpath(joinpath(@__DIR__, ".."))
    return state_provenance(project, [@__FILE__,
        joinpath(project, "models", "ift", "scripts", "collide_fixed_momentum.jl"),
        joinpath(project, "models", "ift", "scripts", "state_io.jl"),
        joinpath(project, "src", "QFTSimulations.jl")])
end

"""
    packets(; root=OUTPUT_ROOT)

Prepare the two packets, measure their initial bond energies, and save the
MPS at t=0. Write `packet-profile.csv` with one row per internal bond and
`state.jld2` in a fresh directory. Return the state and measurements, so the
same example can be inspected interactively without reading a file.
"""
function packets(; root=OUTPUT_ROOT)
    directory = output_directory("ising-packets"; root)
    prep = prepare_packets()
    measured = measure(prep.state, prep)
    path = save_ift_state(joinpath(directory, "state.jld2"), prep.state;
        step=0, dt=prep.parameters["time_step"], parameters=prep.parameters,
        reference=prep.reference, energy_density=measured.energy,
        spin_density=measured.spin, provenance=calculation_sources())
    open(joinpath(directory, "packet-profile.csv"), "w") do io
        println(io, "bond,energy_difference")
        for n in eachindex(measured.energy)
            println(io, n, ',', measured.energy[n])
        end
    end
    println("Correlation length = ", prep.reference["correlation_length"], " sites")
    println("State norm squared = ", abs2(norm(prep.state)))
    println("Summed internal bond energy above the vacuum = ", sum(measured.energy))
    println("Saved t = 0 to ", path)
    return (; directory, path, preparation=prep, measured)
end

"""
    evolution(; root=OUTPUT_ROOT, final_time=12.0, dt=0.1, save_interval=3.0)

Prepare the same two packets as `packets` and evolve them with two-site TDVP
and maximum bond dimension 12. Times must be nonnegative (positive for `dt`
and `save_interval`) and integer multiples of `dt`. Save the initial MPS,
each requested intermediate MPS, and the final MPS. Save the complete bond
energy and longitudinal-spin histories to `observables.jld2`.

Write lattice time, norm squared, and summed internal bond energy to
`evolution.csv`. Reload the final MPS and compare its central bond energy
with the value saved during evolution. Return the paths, time grid, arrays,
norms, and reloading difference. No plotting or particle projection is run.
"""
function evolution(; root=OUTPUT_ROOT, final_time=12.0, dt=0.1, save_interval=3.0)
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(final_time) && final_time >= 0 || throw(ArgumentError("invalid final time"))
    isfinite(save_interval) && save_interval > 0 || throw(ArgumentError("invalid save interval"))
    steps, save_every = round(Int, final_time/dt), round(Int, save_interval/dt)
    isapprox(steps*dt, final_time; atol=1e-12, rtol=0) ||
        throw(ArgumentError("final_time must be a multiple of dt"))
    save_every >= 1 && isapprox(save_every*dt, save_interval; atol=1e-12, rtol=0) ||
        throw(ArgumentError("save_interval must be a positive multiple of dt"))
    directory = output_directory("ising-evolution"; root)
    prep = prepare_packets()
    state = prep.state
    parameters = merge(prep.parameters, Dict("time_step"=>dt, "final_time"=>final_time,
        "save_interval"=>save_interval))
    times = collect(0:steps) .* dt
    energy = zeros(length(times), length(state)-1)
    spin = zeros(length(times), length(state))
    norm_squared = zeros(length(times))
    algorithm = TDVP2(; trscheme=truncrank(parameters["evolution_bond_dimension"]))
    sources = calculation_sources()
    paths = String[]
    open(joinpath(directory, "evolution.csv"), "w") do io
        println(io, "time,norm_squared,internal_energy")
        for row in eachindex(times)
            if row > 1
                state, _ = timestep(state, prep.ham, times[row-1], dt, algorithm)
            end
            measured = measure(state, prep)
            energy[row, :] = measured.energy
            spin[row, :] = measured.spin
            norm_squared[row] = abs2(norm(state))
            println(io, join((times[row], norm_squared[row], sum(measured.energy)), ','))
            flush(io)
            step = row-1
            if should_save_state(step, steps, save_every)
                path = joinpath(directory, "states", "step_$(lpad(step, 6, '0')).jld2")
                push!(paths, save_ift_state(path, state; step, dt, parameters,
                    reference=prep.reference, energy_density=measured.energy,
                    spin_density=measured.spin, provenance=sources))
            end
            @printf("Lattice time t = %.1f; norm² = %.10f; internal energy = %.8f\n",
                times[row], norm_squared[row], sum(measured.energy))
            flush(stdout)
        end
    end
    data_path = joinpath(directory, "observables.jld2")
    jldsave(data_path; times, energy, spin, norm_squared, parameters)
    saved = load_ift_state(last(paths))
    n = length(state) ÷ 2
    remeasured = real(expectation_value(saved["state"], (n, n+1) => prep.bond)) -
        saved["reference"]["vacuum_energy_density"][n]
    reload_difference = remeasured - saved["energy_density"][n]
    println("Final state: ", last(paths))
    println("Central bond energy after reloading minus its saved value = ", reload_difference)
    return (; directory, paths, data_path, times, energy, spin, norm_squared, reload_difference)
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error("Use: ising.jl spectrum|packets|evolution [OUTPUT_PARENT]")
    choices = Dict("spectrum"=>spectrum, "packets"=>packets, "evolution"=>evolution)
    haskey(choices, ARGS[1]) || error("Choose spectrum, packets, or evolution")
    root = length(ARGS) == 2 ? ARGS[2] : OUTPUT_ROOT
    println("Running the Ising ", ARGS[1], " example. The first call can take several minutes to compile.")
    flush(stdout)
    @time choices[ARGS[1]](; root)
end

end
