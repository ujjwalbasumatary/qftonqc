# The Ising vacuum and spectrum

We use the transverse-field Ising chain to compare an MPS calculation with
an exact dispersion relation. The Hamiltonian is

```math
H=-\sum_n\left(\sigma_n^z\sigma_{n+1}^z+h_x\sigma_n^x+h_z\sigma_n^z\right),
```

with ``J=a=\hbar=1``. Here we choose ``h_x=1.5`` and ``h_z=0``. This is the
paramagnetic phase, away from the critical point at ``h_x=1``. Correlations
decay over a few sites, so a modest MPS bond dimension gives a useful
approximation to the vacuum.

Run the example from the repository root with

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl spectrum
```

[`ising.jl`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/simulations/examples/ising.jl)
has the complete example. It calculates three excitation energies with
vacuum bond dimensions ``D=4`` and ``D=8``, and writes them to `spectrum.csv`.
The couplings and correlation lengths are saved in `parameters.toml`.

## Finding the uniform vacuum

In an interactive Julia session, load the example and obtain the Hamiltonian
and vacuum with

```julia
include("simulations/examples/ising.jl")
using MPSKit, Random
Collision = IsingExamples.Collision

Random.seed!(20260926)
H = Collision.get_ham(1.5, 0.0)
vacuum = Collision.prep_gs(4, H)
correlation_length(vacuum)
```

`get_ham` constructs a matrix product operator. `prep_gs` uses the variational
uniform MPS algorithm (VUMPS) to approximate the ground state with a one-site
uniform MPS. The physical
index has dimension two, corresponding to the two spin states. The virtual
indices have dimension ``D`` and carry the entanglement between sites.

The uniform MPS describes an infinite chain. There is no finite chain length
to choose in this example. Its correlation length comes from the eigenvalues
of the MPS transfer matrix. If the largest eigenvalue is normalized to one
and the next largest in magnitude is ``\lambda_2``, then
``\xi=-1/\log|\lambda_2|`` in lattice sites.

## Adding an excitation

An excitation tensor ``B`` replaces the vacuum tensor at one site. Summing
its translations with phase ``e^{ipn}`` gives a state of momentum ``p``.
The quasiparticle calculation varies ``B`` at fixed momentum and returns
the energy above the vacuum.

```julia
momenta = [0.0, 0.4, 0.8]
energies, states = Collision.get_QPstate(vacuum, H, momenta)
energies[:, 1]
```

Rows correspond to momenta. The first column contains the lowest excitation
returned at each momentum. `states` contains the corresponding tangent-space
states; their tensors are used to prepare the packets in the next example.

## Comparing with the exact dispersion

At zero longitudinal field, the exact single-fermion energy is

```math
E(p)=2\sqrt{(h_x-1)^2+4h_x\sin^2(p/2)}.
```

The gap is therefore ``m_1=E(0)=1`` for this choice of ``h_x``. Calculate
the exact values at the same momenta with

```julia
using QFTSimulations: ift_free_fermion_dispersion
exact = ift_free_fermion_dispersion.(momenta, 1.5)
real.(energies[:, 1]) - exact
```

The complete script repeats the calculation at ``D=8``. Compare the change
with ``D`` and the difference from the exact answer separately. Increasing
``D`` enlarges the space of vacuum and excitation tensors available to the
variational calculation.

For the supplied seed and package versions, the output is

| Momentum | Exact energy | MPS energy, ``D=4`` | MPS energy, ``D=8`` |
| --- | --- | --- | --- |
| 0.0 | 1.00000000 | 1.00000000 | 1.00000000 |
| 0.4 | 1.39544547 | 1.39544452 | 1.39544547 |
| 0.8 | 2.15395438 | 2.15395412 | 2.15395438 |

The largest absolute energy difference is about ``9.5\times10^{-7}`` at
``D=4`` and ``2.8\times10^{-10}`` at ``D=8`` for these three momenta.
The correlation lengths are approximately 2.038 and 2.308 sites,
respectively; their change with bond dimension is still appreciable even
though the listed excitation energies agree closely with the exact values.

You can repeat the example at a different ``h_x>1`` by editing `hx` in
`spectrum`. Moving closer to one reduces the gap and increases the
correlation length. You will then need to check the dependence on ``D``
again. The coupling conventions are explained in
[Ising field theory](../physics/ising.md).
