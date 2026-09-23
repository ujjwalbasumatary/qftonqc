# Shared Julia functions

The model programs use `QFTSimulations` to construct oscillator matrices,
evaluate the free dispersion relations, and assemble wave packets as MPS
tensors. Start Julia from the repository root with `--project=simulations`,
then load the package with

```julia
using QFTSimulations
```

In the Julia REPL, `?two_particle_packet_tensors` opens the same docstring
shown below. Each function description includes its arguments, the returned
arrays or values, and the conventions used in the calculation.

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
