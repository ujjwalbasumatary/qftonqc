"""
    IFTBoundStateSpectrum

Calculate several Ising excitation energies while increasing the vacuum MPS
bond dimension. The rest energies are saved first for every requested bond
dimension; a momentum scan then checks how the low eigenvalues disperse.
No collision is evolved, and the new vacua are not substituted into saved
collision states.
"""
module IFTBoundStateSpectrum

using ArgParse
using Dates
using JLD2
using LinearAlgebra
using MPSKit
using Random
using TensorKit
using TOML

include(joinpath(@__DIR__, "particle_basis.jl"))
using .IFTParticleBasis: right_gauge_tensor

export parse_cmdline, ising_hamiltonian, excitation_solutions, sampled_threshold, main

"""Read the fields, bond dimensions, solver tolerances, and output directory."""
function parse_cmdline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--gx"
            arg_type = Float64
            default = 1.06
        "--gz"
            arg_type = Float64
            default = 0.006
        "--bond-dimensions"
            arg_type = Int
            nargs = '+'
            default = [16, 24, 32]
        "--momenta"
            arg_type = Float64
            nargs = '+'
            default = [0.1, 0.2, 0.3, 0.38]
        "--num"
            help = "Number of variational eigenvalues at each momentum"
            arg_type = Int
            default = 3
        "--vacuum-tolerance"
            arg_type = Float64
            default = 1e-10
        "--excitation-tolerance"
            arg_type = Float64
            default = 1e-9
        "--maxiter"
            arg_type = Int
            default = 250
        "--krylovdim"
            arg_type = Int
            default = 40
        "--seed"
            arg_type = Int
            default = 20260925
        "--output-dir"
            help = "A new subdirectory is created here; existing results are retained"
            arg_type = String
            default = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "results", "ift", "spectra"))
        "--rest-only"
            help = "Stop after the zero-momentum calculation for all bond dimensions"
            action = :store_true
    end
    return parse_args(ARGS, settings)
end

"""
    ising_hamiltonian(gx, gz)

Return the one-site uniform MPO for
`H = -sum_j (σz[j]σz[j+1] + gx*σx[j] + gz*σz[j])`, with coupling and
lattice spacing equal to one. The fields must be finite and real.
"""
function ising_hamiltonian(gx::Real, gz::Real)
    isfinite(gx) && isfinite(gz) || throw(ArgumentError("the fields must be finite"))
    sx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    sz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    return InfiniteMPOHamiltonian(
        PeriodicVector([ℂ^2]), 1 => -gx * sx - gz * sz, (1, 2) => -sz ⊗ sz
    )
end

"""
    excitation_solutions(vacuum, hamiltonian, momentum, environments;
                         num=3, tolerance=1e-9, maxiter=250, krylovdim=40)

Compute `num` low eigenvalues of the one-site quasiparticle Hamiltonian at a
given lattice momentum. Return the energies, quasiparticle states, dense left-
and right-gauged tensors, and separately evaluated eigenvector residuals
`norm(H_eff*B - E*B)/norm(B)` in a named tuple.

The returned `residual_converged` flags compare those residuals to `tolerance`;
the eigensolver also prints its own convergence warnings. Passing this test
only establishes that an eigenpair solves the chosen finite-dimensional
variational problem. It does not establish that the eigenstate is a stable
particle. Near the two-particle continuum, its identity and energy must be
checked as the vacuum bond dimension and excitation ansatz are varied.

MPSKit does not return its excitation convergence record through the public
`excitations` interface. The additional residual evaluation therefore uses
its `EffectiveExcitationHamiltonian`, with the same vacuum environments and
environment tolerance as the eigensolver. The package versions are saved by
the calling program because this effective-operator interface is internal.
"""
function excitation_solutions(vacuum, hamiltonian, momentum, envs;
    num::Integer=3, tolerance::Real=1e-9, maxiter::Integer=250, krylovdim::Integer=40)
    num > 0 || throw(ArgumentError("num must be positive"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    isfinite(momentum) || throw(ArgumentError("momentum must be finite"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    krylovdim > num || throw(ArgumentError("krylovdim must exceed num"))
    environment_tolerance = min(1e-11, tolerance / 100)
    algorithm = QuasiparticleAnsatz(;
        tol=Float64(tolerance), maxiter=Int(maxiter), krylovdim=Int(krylovdim),
        alg_environments=(; tol=environment_tolerance, maxiter=Int(maxiter), verbosity=0)
    )
    values, states = excitations(hamiltonian, algorithm, Float64(momentum), vacuum, envs; num=Int(num))
    length(values) >= num || error("the eigensolver returned fewer eigenpairs than requested")
    values, states = values[1:num], states[1:num]
    renormalization_energy = MPSKit.effective_excitation_renormalization_energy(
        hamiltonian, first(states), envs, envs
    )
    effective = MPSKit.EffectiveExcitationHamiltonian(hamiltonian, envs, envs, renormalization_energy)
    residuals = Float64[]
    left_tensors, right_tensors = Array{ComplexF64,3}[], Array{ComplexF64,3}[]
    for (energy, state) in zip(values, states)
        action = effective(state; tol=environment_tolerance, maxiter=Int(maxiter), verbosity=0)
        push!(residuals, norm(action - energy * state) / norm(state))
        tensor = ComplexF64.(convert(Array, state[1])[:, :, 1, :])
        push!(left_tensors, tensor)
        push!(right_tensors, right_gauge_tensor(vacuum, tensor, momentum))
    end
    return (; momentum=Float64(momentum), energies=values, states, left_tensors,
        right_tensors, residuals, residual_converged=residuals .<= tolerance,
        tolerance=Float64(tolerance), environment_tolerance)
end

"""
    sampled_threshold(total_momentum, samples, lowest_energies)

Take the minimum of `E1(q) + E1(total_momentum-q)` over the supplied
nonnegative momentum samples whenever both arguments occur in `samples`.
Return `(; energy, left_momentum, right_momentum, count)`.

This is a sampled estimate, and therefore an upper bound on the minimum
over the entire Brillouin zone, not a proof that a proposed particle lies
below the continuum. The zero-momentum value is `2E1(0)` if the lightest
dispersion has its minimum at zero. Away from zero, the scan compares the
symmetric split with several unequal splits; denser sampling is needed if
the minimum is not well resolved. Inputs are finite, nonnegative momenta
and their positive lowest-branch energies, with equal lengths.
"""
function sampled_threshold(total_momentum::Real, samples, lowest_energies)
    isfinite(total_momentum) && total_momentum >= 0 || throw(ArgumentError("total momentum must be finite and nonnegative"))
    length(samples) == length(lowest_energies) || throw(DimensionMismatch("momenta and energies must have equal lengths"))
    isempty(samples) && throw(ArgumentError("the momentum sample cannot be empty"))
    all(k -> isfinite(k) && k >= 0, samples) || throw(ArgumentError("sampled momenta must be finite and nonnegative"))
    all(E -> isfinite(E) && E > 0, lowest_energies) || throw(ArgumentError("energies must be finite and positive"))
    candidates = Tuple{Float64,Float64,Float64}[]
    for (i, q) in enumerate(samples)
        other = total_momentum - q
        other >= -1e-12 || continue
        j = findfirst(k -> abs(k - other) < 1e-11, samples)
        j === nothing && continue
        push!(candidates, (Float64(lowest_energies[i] + lowest_energies[j]), Float64(q), Float64(samples[j])))
    end
    isempty(candidates) && throw(ArgumentError("no sampled momenta add to the requested total momentum"))
    best = candidates[argmin(first.(candidates))]
    return (; energy=best[1], left_momentum=best[2], right_momentum=best[3], count=length(candidates))
end

"""
    main(args)

Find independent vacuum MPS approximations at each requested bond dimension,
then compute the first few zero-momentum excitation energies. Save each vacuum
immediately after VUMPS and each spectrum immediately after its calculation.
The saved vacuum gradient residual and excitation residuals describe their
respective stopping errors; neither measures finite-bond-dimension error.

Unless `rest-only` is set, revisit these vacua for the requested nonzero
momenta, including their quarter-, half-, and three-quarter-momenta to sample
the two-particle threshold. The rest-energy calculations for all bond
dimensions precede this scan. Each momentum produces a flushed progress line
and a new JLD2 file. CSV tables contain eigenvalues, residuals, and comparisons
with sampled thresholds. No file from an earlier invocation is overwritten.

The labels `eigen1`, `eigen2`, and so on are eigenvalue orderings, not assigned
particle names. In particular, `eigen2 < 2*eigen1` at zero momentum is only a
candidate bound-state observation until its dependence on bond dimension and
the variational ansatz has been examined. The stored tensors use their own
new vacuum; they cannot be inserted directly into a saved collision prepared
with a different vacuum and different boundary spaces.
"""
function main(args)
    BLAS.set_num_threads(1)
    Random.seed!(args["seed"])
    dimensions = Int.(args["bond-dimensions"])
    all(>(0), dimensions) && !isempty(dimensions) || throw(ArgumentError("positive bond dimensions are required"))
    length(unique(dimensions)) == length(dimensions) || throw(ArgumentError("bond dimensions must not repeat"))
    momenta = Float64.(args["momenta"])
    all(k -> isfinite(k) && 0 < k <= pi, momenta) || throw(ArgumentError("requested momenta must lie in (0,pi]"))
    args["num"] >= 2 || throw(ArgumentError("request at least two eigenvalues to look for a second particle"))
    args["vacuum-tolerance"] > 0 || throw(ArgumentError("vacuum-tolerance must be positive"))
    args["maxiter"] > 0 || throw(ArgumentError("maxiter must be positive"))
    root = abspath(args["output-dir"])
    mkpath(root)
    output = mktempdir(root; prefix="mass-spectrum_", cleanup=false)
    println("Ising bound-state spectrum started ", now())
    println("Output directory: ", output)
    println("Fields: gx=", args["gx"], ", gz=", args["gz"], "; bond dimensions: ", dimensions)
    flush(stdout)
    settings = merge(copy(args), Dict("julia_version" => string(VERSION),
        "MPSKit_version" => string(pkgversion(MPSKit)), "TensorKit_version" => string(pkgversion(TensorKit))))
    open(joinpath(output, "settings.toml"), "w") do io
        TOML.print(io, settings)
    end
    hamiltonian = ising_hamiltonian(args["gx"], args["gz"])
    eigen_keywords = (; num=args["num"], tolerance=args["excitation-tolerance"],
        maxiter=args["maxiter"], krylovdim=args["krylovdim"])
    vacua, environment_list, rest = Dict(), Dict(), Dict()
    table_path = joinpath(output, "energies.csv")
    open(table_path, "w") do io
        println(io, "vacuum_bond_dimension,momentum,eigenvalue_index,energy,eigen_residual,residual_converged,vacuum_gradient_residual")
    end
    vacuum_residuals = Dict{Int,Float64}()
    function save_solution(D, solution)
        filename = "D$(D)_k$(replace(string(solution.momentum),'.'=>'p')).jld2"
        jldsave(joinpath(output, filename); solution, vacuum_bond_dimension=D,
            vacuum_gradient_residual=vacuum_residuals[D])
        open(table_path, "a") do io
            for i in eachindex(solution.energies)
                println(io, join((D,solution.momentum,i,real(solution.energies[i]),
                    solution.residuals[i],solution.residual_converged[i],vacuum_residuals[D]), ','))
            end
        end
    end
    for D in dimensions
        println("D=$D: beginning vacuum calculation at ", now())
        flush(stdout)
        start = time()
        function progress(iter, state, H, envs)
            if iter == 0 || iter % 10 == 0
                println("D=$D: vacuum iteration $iter, elapsed ", round(time()-start; digits=1), " s")
                flush(stdout)
            end
            return state, envs
        end
        vacuum, envs, gradient = find_groundstate(InfiniteMPS(ℂ^2, ℂ^D), hamiltonian,
            VUMPS(; tol=args["vacuum-tolerance"], maxiter=args["maxiter"], finalize=progress))
        vacua[D], environment_list[D], vacuum_residuals[D] = vacuum, envs, Float64(gradient)
        vacuum_energy = real(sum(expectation_value(vacuum, hamiltonian, envs)))
        schmidt_values = svdvals(convert(Array, vacuum.C[1]))
        jldsave(joinpath(output,"vacuum_D$(D).jld2"); vacuum, hamiltonian,
            gradient_residual=gradient, vacuum_energy, schmidt_values,
            correlation_length=correlation_length(vacuum), settings)
        println("D=$D: vacuum saved; gradient residual=$gradient, energy/site=$vacuum_energy, smallest Schmidt value=", minimum(schmidt_values))
        println("D=$D: beginning k=0 excitation solve at ", now())
        flush(stdout)
        solution = excitation_solutions(vacuum, hamiltonian, 0.0, envs; eigen_keywords...)
        rest[D] = solution
        save_solution(D, solution)
        mass1 = real(solution.energies[1])
        println("D=$D REST RESULT: energies=", real.(solution.energies), "; residuals=", solution.residuals,
            "; eigen2/(2m1)=", real(solution.energies[2])/(2mass1), "; 2m1-eigen2=", 2mass1-real(solution.energies[2]))
        println("D=$D: completed rest calculation at ", now(), ", elapsed ", round(time()-start; digits=1), " s")
        flush(stdout)
    end
    args["rest-only"] && return output
    support = sort(unique(round.(vcat([0.0], [fraction*k for k in momenta for fraction in (0.25,0.5,0.75,1.0)]); digits=12)))
    threshold_path = joinpath(output,"sampled_thresholds.csv")
    open(threshold_path,"w") do io
        println(io,"vacuum_bond_dimension,total_momentum,eigen2,sampled_threshold,equal_split_threshold,minimizing_q,minimizing_other_q,sampled_pairs")
    end
    for D in dimensions
        solutions = Dict(0.0 => rest[D])
        for k in support[2:end]
            println("D=$D: beginning k=$k excitation solve at ", now())
            flush(stdout)
            solution = excitation_solutions(vacua[D], hamiltonian, k, environment_list[D]; eigen_keywords...)
            solutions[k] = solution
            save_solution(D, solution)
            println("D=$D k=$k RESULT: energies=", real.(solution.energies), "; residuals=", solution.residuals)
            flush(stdout)
        end
        lowest = [real(solutions[k].energies[1]) for k in support]
        for k in vcat([0.0], momenta)
            threshold = sampled_threshold(k, support, lowest)
            energy2 = real(solutions[round(k;digits=12)].energies[2])
            equal_split = 2real(solutions[round(k/2;digits=12)].energies[1])
            open(threshold_path,"a") do io
                println(io,join((D,k,energy2,threshold.energy,equal_split,threshold.left_momentum,
                    threshold.right_momentum,threshold.count),','))
            end
        end
    end
    println("Spectrum scan completed ", now())
    flush(stdout)
    return output
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    IFTBoundStateSpectrum.main(IFTBoundStateSpectrum.parse_cmdline())
end
