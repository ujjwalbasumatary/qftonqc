using ArgParse
using LinearAlgebra
using MPSKit
using TensorKit

using QFTSimulations: ift_eta_latt, ift_free_fermion_dispersion

BLAS.set_num_threads(1)

"""
    parse_cmdline()

Read the spectrum parameters from `ARGS` and return the dictionary produced by
`ArgParse.parse_args`. The momentum grid contains `points` equally spaced
values from zero through `k-max`, including both endpoints. `target-cm-ratio`
is the desired value of `2E(k)/E(0)` for two equal and opposite momenta.

Calling this function with `--help` prints the option list and exits through
ArgParse.
"""
function parse_cmdline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--gx"
        help = "transverse field g_x"
        arg_type = Float64
        default = 1.06

        "--gz"
        help = "longitudinal field g_z"
        arg_type = Float64
        default = 0.006

        "--bond-dimension", "-D"
        help = "vacuum uMPS bond dimension"
        arg_type = Int
        default = 10

        "--k-max"
        help = "largest nonnegative momentum in the scan"
        arg_type = Float64
        default = 1.5

        "--points", "-n"
        help = "number of equally spaced momenta from zero through k-max"
        arg_type = Int
        default = 16

        "--target-cm-ratio"
        help = "target center-of-mass energy in units of the lightest mass"
        arg_type = Float64
        default = 6.0

        "--vumps-tolerance"
        help = "VUMPS gradient tolerance"
        arg_type = Float64
        default = 1e-9
    end
    return parse_args(ARGS, settings)
end

@doc raw"""
    main(args)

Calculate a uniform matrix product state for the vacuum and one tangent-space
excitation energy for each momentum in the requested interval. The spin-chain
Hamiltonian is

```math
H=-\sum_j\left(\sigma_j^z\sigma_{j+1}^z
  +g_x\sigma_j^x+g_z\sigma_j^z\right),
```

with the nearest-neighbour coupling and lattice spacing set to one. The vacuum
has a one-site unit cell and bond dimension `args["bond-dimension"]`. VUMPS is
run with the requested gradient tolerance and at most 250 iterations.

`MPSKit.excitations` is called with the default `QuasiparticleAnsatz` at every
momentum. The program uses the single branch returned by that call; it does not
match particle species between momenta. If ``E(k)`` denotes this branch, the
reported mass is ``m_1=E(0)`` and the centre-of-mass estimate is
``E_{\rm cm}/m_1=2E(k)/m_1``. The selected momentum is the grid point nearest
the requested ratio.

The function prints the fields, lattice scaling variable, vacuum correlation
length, ``m_1``, the selected momentum, and a comma-separated table with
columns `k`, `E(k)`, and `E(k)/m_1`. It writes no files. For `g_z == 0`, the
accepted fields have `g_x > 1`, so the scaling variable is positive infinity.
In this case the program also prints the largest absolute difference from

```math
E_{\rm exact}(k)=2\sqrt{1+g_x^2-2g_x\cos k}.
```

At `g_z == 0` and `g_x < 1`, a single kink connects two different ordered
vacua. The excitation ansatz used here has the same vacuum on both sides,
so the function rejects these fields before finding the vacuum. It also
rejects the critical point `(g_x, g_z) == (1, 0)`, where ``E(0)=0`` and the
mass ratios used for target selection are undefined. Both cases throw
`ArgumentError` with the corresponding physical reason.
"""
function main(args)
    gx = args["gx"]
    gz = args["gz"]
    D = args["bond-dimension"]
    kmax = args["k-max"]
    npoints = args["points"]
    target_cm_ratio = args["target-cm-ratio"]
    tolerance = args["vumps-tolerance"]

    gx >= 0 || throw(ArgumentError("gx must be nonnegative"))
    isfinite(gz) || throw(ArgumentError("gz must be finite"))
    D > 0 || throw(ArgumentError("bond-dimension must be positive"))
    0 < kmax <= pi || throw(ArgumentError("k-max must lie in (0, pi]"))
    npoints >= 2 || throw(ArgumentError("points must be at least two"))
    target_cm_ratio >= 2 || throw(ArgumentError("target-cm-ratio must be at least two"))
    tolerance > 0 || throw(ArgumentError("vumps-tolerance must be positive"))

    if iszero(gz)
        gx < 1 && throw(ArgumentError(
            "At gz = 0 and gx < 1, a single kink connects different ordered vacua. " *
            "This spectrum program uses the same vacuum on both sides of an excitation; " *
            "the one-particle mass requires a kink construction."
        ))
        gx == 1 && throw(ArgumentError(
            "At gx = 1 and gz = 0, the excitation gap vanishes. " *
            "E(k)/m_1 and the target centre-of-mass energy ratio are undefined."
        ))
    end

    sigma_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    sigma_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    hamiltonian = InfiniteMPOHamiltonian(
        PeriodicVector([ℂ^2]),
        1 => -gx * sigma_x - gz * sigma_z,
        (1, 2) => -sigma_z ⊗ sigma_z,
    )

    initial = InfiniteMPS(ℂ^2, ℂ^D)
    vacuum, _, _ = find_groundstate(
        initial, hamiltonian, VUMPS(; tol=tolerance, maxiter=250)
    )
    momenta = collect(range(0.0, kmax; length=npoints))
    energies_raw, _ = excitations(
        hamiltonian, QuasiparticleAnsatz(), momenta, vacuum
    )
    energies = vec(real.(energies_raw))
    length(energies) == length(momenta) || error(
        "expected one excitation energy per momentum; received $(size(energies_raw))"
    )

    mass = first(energies)
    ratios = energies ./ mass
    target_single_ratio = target_cm_ratio / 2
    target_index = argmin(abs.(ratios .- target_single_ratio))

    println("g_x = $gx")
    println("g_z = $gz")
    println("eta_latt = ", ift_eta_latt(gx, gz))
    println("vacuum bond dimension = $D")
    println("correlation length = ", correlation_length(vacuum))
    println("m_1 = $mass")
    println("target E_cm / m_1 = $target_cm_ratio")
    println("nearest k = ", momenta[target_index])
    println("achieved E_cm / m_1 = ", 2ratios[target_index])
    println("\nk,E(k),E(k)/m_1")
    for (k, energy, ratio) in zip(momenta, energies, ratios)
        println("$k,$energy,$ratio")
    end

    if iszero(gz)
        exact = ift_free_fermion_dispersion.(momenta, gx)
        println("maximum free-dispersion error = ", maximum(abs.(energies .- exact)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_cmdline())
end
