# Companion code for the manuscript revised on 18 October 2026

Tag: [`book-2026-10-18.1`](https://github.com/ujjwalbasumatary/qftonqc/tree/book-2026-10-18.1).

This snapshot accompanies *Lectures on Quantum Field Theory on a Quantum Computer*.
The public paper is [arXiv:2512.02706](https://arxiv.org/abs/2512.02706).
The tag fixes the source version cited in the revised manuscript.
This corrected tag adds the notebook syntax-highlighting dependency required
by a fresh documentation environment. The numerical examples are unchanged.

## Contents

- Four NumPy notebooks work through Schmidt truncation and canonical MPS,
  MPOs and two-site DMRG, TEBD and TDVP, and uniform MPS with VUMPS.
  Their 66 code cells have saved execution results, with 16 tensor diagrams.
- The Qiskit classroom examples have a reading order and links to the relevant
  chapters. The Julia and Python tutorials include installation instructions.
- `course/examples/anharmonic-oscillator-spectrum/anharmonic_cutoff_comparison.py`
  reproduces the low-energy cutoff table using NumPy.
- The local-insertion phi4 programs check the returned ground-state residual
  before evolving the state and pass physical time to the integrator.
- The free-Ising calibration script and the four saved input files used by it
  are included, together with the extracted energies and packet centroids.

## Use this version

```sh
git clone --branch book-2026-10-18.1 https://github.com/ujjwalbasumatary/qftonqc.git
cd qftonqc
julia --startup-file=no --project=simulations -e 'using Pkg; Pkg.instantiate()'
```

`simulations/Manifest.toml` records the Julia dependencies. For Python kernels,
follow [the setup instructions](python-docs/setup.md). Website build instructions
are in [the Python documentation README](python-docs/README.md) and
[the Julia documentation README](docs/README.md).

To repeat the small NumPy comparisons from the repository root:

```sh
python course/examples/anharmonic-oscillator-spectrum/anharmonic_cutoff_comparison.py
python simulations/models/ift/scripts/check_free_reference.py . tmp/free-reference
```

The latter reads `results/ift/2026-09-24-integrable-320/` and does not evolve a state.
The input hashes and analysis conventions are recorded in
`docs/src/assets/ising-collision/ift_free_reference.json`.

## Validation and remaining calculations

The four MPS notebooks have no saved error outputs. The Qiskit classroom
notebooks retain their historical outputs and unfinished cells, identified in
[the circuit tutorial](python-docs/qiskit.md). They were not rerun for this release.
The exact saved notebook for the historical five-qubit oscillator preparation
numbers in Chapter 14 has not been identified. The linked JLP tutorial provides
a related implementation.

The NumPy oscillator cutoff checks pass, and the free-Ising calibration
reproduces the recorded energies and fitted velocities.

The phi4 residual guards passed 20 checks with supplied solver-return values.
These checks test tolerance handling and rejection of unconverged states.
The Python tutorial passed a strict Sphinx build, and the Julia tutorial built
with Documenter. Neither documentation build executes the notebooks or simulations.

The matched Ising dynamics comparison requested by the referee remains
outstanding. The Schwinger example in this snapshot is the source quench
already discussed in the manuscript.
