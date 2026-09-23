# Shared Julia functions

`QFTSimulations` contains the oscillator matrix elements, free dispersions,
momentum grids, and packet tensor constructions used by the model programs.
You can load it from the repository root with

```julia
using QFTSimulations
```

when Julia has been started with `--project=simulations`. In the Julia REPL,
`?two_particle_packet_tensors` opens the same docstring shown below.

```@docs
QFTSimulations
```

## Oscillator basis

```@docs
harmonic_oscillator_matrices
```

## Lattice formulas

```@docs
ift_eta_latt
ift_free_fermion_dispersion
schwinger_free_lattice_dispersion
```

## Momentum grids and wave packets

```@docs
commensurate_momentum_grid
gaussian_weights
single_particle_packet_tensors
two_particle_packet_tensors
```

## Internal definitions

These functions check dimensions and assemble the tensor blocks used above.

```@docs
QFTSimulations._ISING_MAGNETIC_EXPONENT
QFTSimulations._require_finite
QFTSimulations._check_packet_inputs
QFTSimulations._block_upper
QFTSimulations._right_multiply_bond
```
