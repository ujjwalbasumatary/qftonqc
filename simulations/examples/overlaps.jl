"""
A six-site example of two- and three-excitation overlaps. The reference vacuum
is the product state |000000⟩, and an excitation changes |0⟩ to |1⟩. This
example uses small dense tensors and requires no ground-state calculation.
Run with `julia --project=simulations simulations/examples/overlaps.jl`.
"""
module OverlapExample

using LinearAlgebra, TOML
include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "two_particle_overlap.jl"))
include(joinpath(@__DIR__, "..", "models", "ift", "scripts", "three_particle_overlap.jl"))
using .IFTTwoParticleOverlap
using .IFTThreeParticleOverlap

"""
    run_example(; root=..., probability_three=0.3)

Construct sqrt(1-p)|010010⟩ + i sqrt(p)|010110⟩ as six MPS tensors, where
`p=probability_three` lies between zero and one. Calculate both excitation
counts using the pair and triple contraction functions used in the Ising
analysis, and save their weights to `weights.toml` in a fresh directory.

Also calculate the projection of (1,0) onto two overlapping, normalized
reference vectors (1,0) and (1,1)/sqrt(2). Their squared overlaps sum to 1.5;
including the Gram matrix gives 1. This is a separate linear-algebra example
of why the individual norms do not suffice for nonorthogonal references.
Return the weights, overlaps, Gram matrices, and output directory.
"""
function run_example(; root=normpath(joinpath(@__DIR__, "..", "..", "results", "examples")),
        probability_three=0.3)
    0 <= probability_three <= 1 || throw(ArgumentError("probability must lie in [0,1]"))
    BLAS.set_num_threads(1)
    A = reshape(ComplexF64[1, 0], 1, 2, 1)
    B = reshape(ComplexF64[0, 1], 1, 2, 1)
    Cinv = ones(ComplexF64, 1, 1)
    mixed = sqrt(1-probability_three) * A + im * sqrt(probability_three) * B
    tensors = [A, B, A, mixed, B, A]
    state_norm2 = prod(sum(abs2, tensor) for tensor in tensors)

    pairs = two_particle_overlaps(tensors, A, A, Cinv, B, B)
    pair_norms = localized_pair_norms(A, Cinv, B, B, 5)
    weight_two = two_particle_weight(pairs, pair_norms, state_norm2)

    triples = three_particle_overlaps(tensors, A, A, Cinv, B, B, B)
    grams = Dict(span => middle_position_gram(A, Cinv, B, B, B, span) for span in 2:5)
    weight_three = three_particle_weight(triples, grams, state_norm2).weight

    references = [1.0 1/sqrt(2); 0.0 1/sqrt(2)]
    psi = [1.0, 0.0]
    gram = references' * references
    overlaps = references' * psi
    naive_sum = sum(abs2, overlaps)
    projection = real(dot(overlaps, gram \ overlaps)) / sum(abs2, psi)

    mkpath(root)
    directory = mktempdir(abspath(root); prefix="overlaps-", cleanup=false)
    open(joinpath(directory, "weights.toml"), "w") do io
        TOML.print(io, Dict("two_excitations"=>weight_two, "three_excitations"=>weight_three,
            "chosen_probability_three"=>probability_three, "state_norm_squared"=>state_norm2,
            "nonorthogonal_squared_overlap_sum"=>naive_sum,
            "nonorthogonal_projection"=>projection, "julia_version"=>string(VERSION)))
    end
    println("Two excitations: ", weight_two)
    println("Three excitations: ", weight_three)
    println("Nonorthogonal references: squared-overlap sum = ", naive_sum,
        "; projection with the Gram matrix = ", projection)
    println("Outputs: ", directory)
    return (; directory, weight_two, weight_three, pairs, triples, grams, naive_sum, projection)
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) || error("Use: julia --project=simulations simulations/examples/overlaps.jl")
    @time run_example()
end

end
