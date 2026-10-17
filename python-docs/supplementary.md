---
site:
  hide_outline: true
---

# Additional examples

Use these alongside the [quantum-circuit tutorial](qiskit.md) for individual
measurements, more gate-level detail and oscillator applications.

## Measurements and Fourier transforms

- [Statevector sampler](../course/tutorials/03-qiskit-and-tfim/supplementary/statevector_sampler.ipynb): compare sampled probabilities with statevectors for Bell states and spin precession. This accompanies Chapter 7.
- [Statevector estimator](../course/tutorials/03-qiskit-and-tfim/supplementary/statevector_estimator.ipynb): evaluate Pauli expectation values, following the circuit parameter and measured observable separately. This accompanies Chapter 7.
- [Expanded Fourier and phase-estimation examples](../course/tutorials/04-qft-and-phase-estimation/hatQFTetc.ipynb): inspect intermediate circuits and bit ordering for the algorithms in Chapter 8.

## Oscillator applications

- [Preparation in an oscillator basis](../course/tutorials/05-state-preparation/state_prep/sho_adiabatic_state_prep.ipynb): encode four oscillator levels in two qubits. Compare the retained basis and operators with the position-grid examples in Chapters 9–10.
- [Oscillator dynamics](../course/tutorials/05-state-preparation/state_prep/qmwithqiskit.ipynb): harmonic motion, moving packets and quartic interactions, related to Chapter 14. **Unfinished:** packet calculations contain interrupted runs, and the classical and circuit comparisons still need a common discretized Hamiltonian.
- [Oscillator spectra and noise](../course/examples/anharmonic-oscillator-spectrum/noisy_spectrum_qiskit.ipynb): extract spectral peaks from a Hadamard-test signal, related to Chapters 8 and 14. The noiseless spectrum is saved. **Unfinished:** the comparison of readout errors, gate errors and relaxation was interrupted.

Saved output records the earlier classroom runs. The unfinished calculations
retain their execution history. The [setup page](setup.md) lists the packages
and plotting requirements.

Earlier state-preparation trials remain in the
[source repository](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/tutorials/05-state-preparation/state_prep/random).


```{toctree}
:hidden:
:maxdepth: 1

Statevector sampler <../course/tutorials/03-qiskit-and-tfim/supplementary/statevector_sampler>
Statevector estimator <../course/tutorials/03-qiskit-and-tfim/supplementary/statevector_estimator>
Expanded Fourier examples <../course/tutorials/04-qft-and-phase-estimation/hatQFTetc>
Oscillator basis <../course/tutorials/05-state-preparation/state_prep/sho_adiabatic_state_prep>
Oscillator dynamics <../course/tutorials/05-state-preparation/state_prep/qmwithqiskit>
Oscillator spectra and noise <../course/examples/anharmonic-oscillator-spectrum/noisy_spectrum_qiskit>
```
