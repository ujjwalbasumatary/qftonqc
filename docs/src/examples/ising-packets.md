# Preparing two packets

A momentum eigenstate extends throughout the chain. To prepare a collision,
we instead want two localized packets moving towards each other. We retain
``h_x=1.5`` and ``h_z=0`` from the spectrum example and place the packets
inside a 64-site window of the infinite MPS vacuum.

```bash
julia --startup-file=no --threads=1 --project=simulations simulations/examples/ising.jl packets
```

The command prepares the state at ``t=0`` and saves it as `state.jld2` in a
new directory under `results/examples/`. `packet-profile.csv` contains the
energy on each internal bond minus its value in the vacuum.

## Packet width and momentum

The left and right packet centres are ``n_L=16`` and ``n_R=48``. Their
amplitudes have the form

```math
f_L(n)=e^{+i\kappa(n-n_L)}e^{-(n-n_L)^2/\sigma^2},\qquad
f_R(n)=e^{-i\kappa(n-n_R)}e^{-(n-n_R)^2/\sigma^2},
```

with ``\kappa=0.6`` and ``\sigma=6``. The excitation tensor is evaluated at
``+\kappa`` for the left packet and ``-\kappa`` for the right packet. We
multiply that tensor by the corresponding amplitude at each site.

The width parameter is not the standard deviation of the position
probability. For this Gaussian, ``|f(n)|^2`` has standard deviation
``\sigma/2=3``. Ignoring the lattice and the finite support, its Fourier
transform has momentum-probability standard deviation ``1/\sigma=1/6``.
A larger ``\sigma`` therefore makes the packet broader in position and
narrower in momentum. The excitation tensor in this construction remains
fixed at the central momentum.

## From excitation tensors to the MPS

The following Julia commands prepare the complete state without writing files.

```julia
include("simulations/examples/ising.jl")
using LinearAlgebra
prep = IsingExamples.prepare_packets()
state = prep.state
norm(state)
prep.reference["correlation_length"]
```

Inside `prepare_packets`, the same functions used by the collision program
calculate the excitation tensors and assemble the window.

```julia
Collision = IsingExamples.Collision
energies, excitations = Collision.get_QPstate(prep.vacuum, prep.ham, [0.6, -0.6])
B = Collision.get_B_tensor_list(excitations)
tensors = Collision.create_stacked_tensor(prep.vacuum, B, 64, 16, 0.6, 6.0)
```

`create_stacked_tensor` uses sites 1–32 for the left packet and sites 33–64
for the right packet. Every term in the resulting superposition has one
excitation insertion in each half. The block structure of the MPS tensors
keeps track of whether an insertion has already occurred as we move along
the chain. The shared function `two_particle_packet_tensors` implements
that construction.

The vacuum bond dimension is four. Before any time evolution, the packet
construction needs virtual dimensions up to eight. `WindowMPS` joins these
finite tensors to the unchanged uniform vacuum on both sides. We normalize
the whole state once after constructing it.

## Measuring the initial energy distribution

We assign half of each onsite field term to each adjacent bond,

```math
h_{n,n+1}=-\sigma_n^z\sigma_{n+1}^z
-\frac{h_x}{2}(\sigma_n^x+\sigma_{n+1}^x)
-\frac{h_z}{2}(\sigma_n^z+\sigma_{n+1}^z).
```

`measure` calculates
``\delta e_n=\langle h_{n,n+1}\rangle_\psi-\langle h_{n,n+1}\rangle_\Omega``
on the 63 internal bonds, where ``|\Omega\rangle`` is the uniform vacuum.

```julia
measured = IsingExamples.measure(state, prep)
measured.energy
sum(measured.energy)
```

The energy is concentrated near the two packet centres. Its sum approaches
the two incoming excitation energies when the packets have a narrow
momentum distribution and are well separated. Here each packet has a finite
momentum width, so compare the sum with ``2E(0.6)`` as an estimate.
For these parameters, the measured sum is approximately 3.54168, while
the two central-momentum energies sum to 3.51908.

To change the packet width or separation, edit `sigma` or `n_center` in
`prepare_packets`. The right centre is always `length - n_center`. Keep
both packets away from the join between the two halves and from the window
ends. The [wave-packet page](../physics/wave-packets.md) gives the full
construction, including the alternative that uses momentum-dependent tensors.
