"""
A 12-site source quench using the onsite basis and Hamiltonian from
`source_quench.jl`. Run with `julia --project=simulations
simulations/examples/schwinger.jl`. Including the file defines `run_example`
without starting DMRG or time evolution.
"""
module SchwingerExample

using LinearAlgebra, Random, Printf, JLD2
using MPSKit, TensorKit
module Quench
include(joinpath(@__DIR__, "..", "models", "schwinger", "scripts", "source_quench.jl"))
end

"""
    run_example(; root=..., final_time=1.0, dt=0.1)

Diagonalize the onsite Hamiltonian in 24 and 32 oscillator states and compare
the lowest four energies. Use the 24-state calculation to retain four states
per site on a 12-site chain, with MPS bond dimension 8. The couplings are
beta=sqrt(4pi), m=1, mu=0.2, theta=0, and kappa=1 in the program's convention.
Find the ground state with a source J0=0.2 on sites 4:8, then set J1=0 and
evolve with one-site TDVP to `final_time`, saving the field, norm squared,
and energy at each time. A shorter last interval reaches `final_time` exactly.

`field.csv` and `quench.jld2` go in a fresh directory beneath `root`. The
JLD2 file also stores both onsite spectra, their difference, parameters,
and the DMRG residual. Return the arrays, comparisons, and file locations.
"""
function run_example(; root=normpath(joinpath(@__DIR__, "..", "..", "results", "examples")),
        final_time=1.0, dt=0.1)
    isfinite(final_time) && final_time >= 0 || throw(ArgumentError("invalid final time"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("invalid time step"))
    BLAS.set_num_threads(1)
    Random.seed!(20260926)
    L, retained, D = 12, 4, 8
    beta, mu, mass, theta, kappa = sqrt(4pi), 0.2, 1.0, 0.0, 1.0
    J0, J1 = 0.2, 0.0
    basis = Quench.get_elems(retained; d=24, beta, mu, m=mass, theta)
    larger_basis = Quench.get_elems(retained; d=32, beta, mu, m=mass, theta)
    basis_difference = larger_basis.energies - basis.energies
    println("Onsite energies for d=24: ", basis.energies)
    println("Change on increasing d to 32: ", basis_difference)
    flush(stdout)

    H0 = Quench.build_hamiltonian(L, retained, basis.phi, basis.phi_sq, basis.onsite;
        source_strength=J0, kappa)
    H1 = Quench.build_hamiltonian(L, retained, basis.phi, basis.phi_sq, basis.onsite;
        source_strength=J1, kappa)
    state = FiniteMPS(L, ℂ^retained, ℂ^D)
    state, _, dmrg_residual = find_groundstate(state, H0, DMRG())
    normalize!(state)
    times = Quench.simulation_times(Float64(final_time), Float64(dt))
    field = zeros(length(times), L)
    norm_squared = zeros(length(times))
    energy = zeros(length(times))
    algorithm = TDVP()
    envs = environments(state, H1)
    for row in eachindex(times)
        if row > 1
            state, envs = timestep(state, H1, times[row-1], times[row]-times[row-1], algorithm, envs)
        end
        for n in 1:L
            field[row, n] = real(expectation_value(state, n => basis.phi))
        end
        norm_squared[row] = abs2(norm(state))
        energy[row] = real(expectation_value(state, H1))
        @printf("Lattice time t = %.1f; <phi_6> = %+.8f; norm² = %.10f\n",
            times[row], field[row, 6], norm_squared[row])
        flush(stdout)
    end
    mkpath(root)
    directory = mktempdir(abspath(root); prefix="schwinger-quench-", cleanup=false)
    parameters = Dict("length"=>L, "d"=>24, "d_trunc"=>retained, "D"=>D,
        "beta"=>beta, "mu"=>mu, "m"=>mass, "theta"=>theta, "kappa"=>kappa,
        "J0"=>J0, "J1"=>J1, "final_time"=>final_time, "time_step"=>dt,
        "random_seed"=>20260926)
    data_path = joinpath(directory, "quench.jld2")
    jldsave(data_path; times, field, norm_squared, energy, parameters, dmrg_residual,
        onsite_energies=basis.energies, onsite_energies_d32=larger_basis.energies,
        basis_difference, julia_version=string(VERSION))
    open(joinpath(directory, "field.csv"), "w") do io
        println(io, "time,site,field")
        for row in eachindex(times), n in 1:L
            println(io, join((times[row], n, field[row, n]), ','))
        end
    end
    println("Outputs: ", directory)
    return (; directory, data_path, times, field, norm_squared, energy, basis_difference, dmrg_residual)
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) || error("Use: julia --project=simulations simulations/examples/schwinger.jl")
    println("Running the Schwinger example. The first call can take several minutes to compile.")
    flush(stdout)
    @time run_example()
end

end
