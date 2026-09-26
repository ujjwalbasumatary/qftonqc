# Two- and three-excitation overlaps

Before applying the overlap functions to an evolved MPS, we can check them
on a state whose excitation content is known. Consider six two-state sites
with product vacuum ``|000000\rangle``. An excitation changes a local
``|0\rangle`` to ``|1\rangle``. Choose

```math
|\psi\rangle=\sqrt{0.7}\,|010010\rangle
+i\sqrt{0.3}\,|010110\rangle.
```

The two terms are orthogonal and contain two and three excitations,
respectively. We should therefore obtain ``P_2=0.7`` and ``P_3=0.3``.
Here the excitation number counts occupied sites in this explicitly chosen
basis. The example tests the contractions used in the Ising analysis.

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/overlaps.jl
```

[`overlaps.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/examples/overlaps.jl)
prints both weights and saves them in `weights.toml`. It uses only small
arrays and needs no vacuum optimization or time evolution.

## Contracting the reference states with the MPS

The tensors for the vacuum and an excitation each have shape ``(1,2,1)``.
Their indices are the left virtual index, physical index, and right virtual
index. The virtual dimension is one because this vacuum is a product state.

```julia
include("simulations/examples/overlaps.jl")
using .OverlapExample.IFTTwoParticleOverlap
using .OverlapExample.IFTThreeParticleOverlap

A = reshape(ComplexF64[1, 0], 1, 2, 1)
B = reshape(ComplexF64[0, 1], 1, 2, 1)
Cinv = ones(ComplexF64, 1, 1)
tensors = [A, B, A, sqrt(0.7)*A + im*sqrt(0.3)*B, B, A]
```

The superposition is entirely at site four, so even ``|\psi\rangle`` has
bond dimension one. We nevertheless contract it with all possible ordered
two- and three-insertion reference states.

```julia
pairs = two_particle_overlaps(tensors, A, A, Cinv, B, B)
pair_norms = localized_pair_norms(A, Cinv, B, B, 5)
P2 = two_particle_weight(pairs, pair_norms, 1.0)

triples = three_particle_overlaps(tensors, A, A, Cinv, B, B, B)
grams = Dict(r => middle_position_gram(A, Cinv, B, B, B, r) for r in 2:5)
P3 = three_particle_weight(triples, grams, 1.0).weight
```

The only nonzero pair amplitude is at sites ``(2,5)``. The only nonzero
triple amplitude is at ``(2,4,5)``. The last argument `1.0` is the norm
squared of the state. Dividing by that norm removes an overall rescaling
of the state from the result.

You can change the chosen probability without editing the tensors by calling
`OverlapExample.run_example(probability_three=0.5)`. The complete script
constructs the corresponding normalized state for you.

## When reference states overlap each other

The product states above are mutually orthogonal. Excitation insertions
above an entangled vacuum need more care. Let the chosen reference states
be ``|\chi_a\rangle``, and define

```math
G_{ab}=\langle\chi_a|\chi_b\rangle,\qquad
o_a=\langle\chi_a|\psi\rangle.
```

``G`` is their Gram matrix. The weight of ``|\psi\rangle`` in their span is

```math
P=\frac{o^\dagger G^+o}{\langle\psi|\psi\rangle},
```

where ``G^+`` is the inverse on the nonzero-eigenvalue subspace. This also
handles linearly dependent reference states without counting the same
direction twice.

The script includes a two-dimensional example with reference vectors
``(1,0)`` and ``(1,1)/\sqrt2`` and state ``\psi=(1,0)``. Both reference
vectors have norm one. Adding their squared overlaps gives ``1+1/2=1.5``;
including the off-diagonal entries of their Gram matrix gives ``P=1``.

In the triple contraction, the Gram matrices above compare different middle
insertion positions at fixed outer positions. Adding the resulting weights
requires different outer-position blocks to be orthogonal. That condition
holds for this product vacuum. For the Ising excitation tensors it must be
established from their canonical and excitation gauges, as described in the
[model-program reference](../reference/programs.md).

The physical particle states in the interacting chain are built from the
vacuum and excitation spectrum of its Hamiltonian. The
[particle-production page](../physics/particle-production.md) explains how
that choice enters the interpretation of outgoing overlaps.
