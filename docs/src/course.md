# Course material

These notebooks and projects accompany HE381, *Quantum Field Theory on a
Quantum Computer*, taught by Prof. Aninda Sinha at IISc in the fall of 2025.
We use NumPy, SciPy, Matplotlib, Qiskit, and Qiskit Aer in the Python
notebooks; you can check each notebook's import cells for the packages it
needs. You can run these notebooks independently of the Julia package in
`simulations/`.

## Teaching notebooks

The numbered directories in
[`course/tutorials/`](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/tutorials)
follow the order in which the topics were taught.

| Topic | Notebooks and calculations |
| --- | --- |
| NumPy and differential equations | [`basicnumpy.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/01-numpy-and-odes/basicnumpy.ipynb) introduces vectors, matrices, and plotting. [`t1solvingdifferentialequations.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/01-numpy-and-odes/t1solvingdifferentialequations.ipynb) solves exponential decay with the forward Euler method. |
| Numerical methods | [`t2.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/02-numerical-methods/t2.ipynb) develops predictor–corrector integration and oscillator time evolution. |
| Qiskit and the Ising model | [`t3.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/03-qiskit-and-tfim/t3.ipynb) covers circuits, measurements, single-spin precession, and two-spin evolution. [`tfim.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/03-qiskit-and-tfim/tfim.ipynb) treats the transverse-field Ising Hamiltonian by exact diagonalization and a variational quantum eigensolver. |
| Fourier transform and phase estimation | [`QFT.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/04-qft-and-phase-estimation/QFT.ipynb) and [`hatQFTetc.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/04-qft-and-phase-estimation/hatQFTetc.ipynb) construct Fourier-transform circuits, estimate phases, and calculate expectation values with the Hadamard test. |
| State preparation | [`tutorial.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/05-state-preparation/state_prep/tutorial.ipynb) discretizes oscillator position and momentum and prepares the harmonic ground state by changing the Hamiltonian. [`sho_adiabatic_state_prep.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/05-state-preparation/state_prep/sho_adiabatic_state_prep.ipynb) introduces a quartic coupling in a truncated oscillator basis. |
| Matrix product states | [`mps_demo.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/06-matrix-product-states/mps_demo.ipynb) and [`MPS_demo_fin.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/06-matrix-product-states/MPS_demo_fin.ipynb) write spin states as products of tensors and examine their bond dimensions and entanglement. |

You can also find supplementary
[differential-equation examples](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/tutorials/01-numpy-and-odes/supplementary)
and separate demonstrations of Qiskit's
[statevector sampler](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/03-qiskit-and-tfim/supplementary/statevector_sampler.ipynb)
and [statevector estimator](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/tutorials/03-qiskit-and-tfim/supplementary/statevector_estimator.ipynb).

## Oscillator spectrum and Mathematica

In
[`noisy_spectrum_qiskit.ipynb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/examples/anharmonic-oscillator-spectrum/noisy_spectrum_qiskit.ipynb),
we represent an anharmonic oscillator in a truncated number basis and use
split time-evolution circuits and Hadamard tests to extract spectral
information. We also simulate the circuits with noise. You can
find the figures beside the notebook in
`course/examples/anharmonic-oscillator-spectrum/`.

The Mathematica notebooks are in
[`course/mathematica/`](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/mathematica).
[`TFIM.nb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/mathematica/TFIM.nb)
contains transverse-field Ising calculations, and
[`stringbreaking.nb`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/mathematica/stringbreaking.nb)
contains Schwinger-model string-breaking calculations.

## Student projects

The reports and accompanying code are in
[`course/projects/`](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/projects).

### Ising field theory — Abhishek Kundu and Ritabrata Ghosh

The [report](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/ift/QFT_QC_Term_Paper%20%281%29.pdf)
is accompanied by notebooks for the
[finite-chain spectrum](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/ift/spectrum.ipynb),
[mass ratios in the zero-momentum sector](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/ift/E8massratio.ipynb),
and [single-particle circuit preparation](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/ift/Single%20Particle%20State%20Preparation%20Final.ipynb).
The [MPS scattering notebook](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/ift/scatter_prob.ipynb)
uses `evoMPS`, which must be installed separately.

### Light–matter interaction — Ankush Kumar and Suman Dafadar

The [term report](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/light-and-matter/HE_381_Term_Paper.pdf)
is in `course/projects/light-and-matter/`.

### Scattering and resonances — Indrayudh Das and Soumyadeep Sarma

The [report](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/scattering-quantum-simulator/Scattering_and_Resonances_on_a_Quantum_Simulator.pdf)
is accompanied by [double-barrier wave-packet evolution](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/scattering-quantum-simulator/Simul.ipynb),
[Wigner time-delay calculations](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/scattering-quantum-simulator/DB_WignerTimeDelay.ipynb),
and a [time-step comparison](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/scattering-quantum-simulator/trotter_error_scaling.ipynb).
The [project directory](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/projects/scattering-quantum-simulator)
also contains Morse-potential, entropy, and circuit-noise calculations.

### Schwinger model — Abhijeet Bhatta and Chayanka Kakati

In their [report](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-1/Final_Term_Paper_Report_AB_CK.pdf)
and [notebook](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-1/Quantum_simulation_of_Schwinger_model_final.ipynb),
Abhijeet and Chayanka develop the lattice Schwinger Hamiltonian from its
staggered-fermion formulation and implement quantum-circuit calculations.

### Schwinger model — Aman Goyal and Nikshay Chugh

The [report](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-2/Schwinger_Model.pdf)
and [notebook](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-2/Term_Paper.ipynb)
contain fermionic and bosonic simulations. The Python programs
[`complete.py`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-2/complete.py)
and [`noisy.py`](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-2/noisy.py)
implement the Hamiltonians and circuit evolution. The submission also includes
[comments on AI use](https://github.com/ujjwalbasumatary/qftonqc/blob/main/course/projects/schwinger/team-2/AI_Usage_and_Comments.pdf).
