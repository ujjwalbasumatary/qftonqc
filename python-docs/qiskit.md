---
site:
  hide_outline: true
---

# Quantum circuits

Build a circuit, relate its gates to a Hamiltonian, and compare the measured
observables with a classical calculation. These six notebooks follow Chapters
7–10 of the lecture notes.

If this is your first visit, [set up the Qiskit kernel](setup.md#qiskit-kernel)
and start with gates and measurements.

## Gates and spin dynamics

1. [Gates and measurements](../course/tutorials/03-qiskit-and-tfim/qiskitbasics.ipynb).
   Prepare a Bell pair and compare statevector amplitudes with measurement counts.
2. [Spin dynamics](../course/tutorials/03-qiskit-and-tfim/t3.ipynb).
   Follow spin precession, then introduce an interaction between two spins.
3. [The Ising chain](../course/tutorials/03-qiskit-and-tfim/tfim.ipynb).
   Compare circuit magnetization with sparse-matrix evolution. The worked sequence
   ends at that comparison. The later variational cells are unfinished drafts.

## Fourier transforms and the oscillator

4. [Fourier transform and phase estimation](../course/tutorials/04-qft-and-phase-estimation/QFT.ipynb).
   Construct the Fourier circuit, estimate an eigenphase and measure an overlap
   with the Hadamard test.
5. [Position-grid encoding](../course/tutorials/05-state-preparation/state_prep/t6.ipynb).
   Represent position and momentum on qubits and express the oscillator
   Hamiltonian through Pauli operators.
6. [Adiabatic state preparation](../course/tutorials/05-state-preparation/state_prep/tutorial.ipynb).
   Start from the two central grid points, turn on the kinetic energy and then
   introduce the quartic interaction.

Each page includes its saved outputs and a notebook download. The repaired final
comparisons in the two position-grid notebooks still need to be rerun.

The [supplementary notebooks](supplementary.md) give more measurement and Fourier
examples, preparation in an oscillator basis, and the spectrum calculation used
alongside Chapter 14. Earlier state-preparation trials remain in the
[source archive](https://github.com/ujjwalbasumatary/qftonqc/tree/main/course/tutorials/05-state-preparation/state_prep/random).


```{toctree}
:hidden:
:maxdepth: 1

Qiskit basics <../course/tutorials/03-qiskit-and-tfim/qiskitbasics>
Spin dynamics <../course/tutorials/03-qiskit-and-tfim/t3>
Ising chain <../course/tutorials/03-qiskit-and-tfim/tfim>
QFT and phase estimation <../course/tutorials/04-qft-and-phase-estimation/QFT>
Position-grid encoding <../course/tutorials/05-state-preparation/state_prep/t6>
Ground-state preparation <../course/tutorials/05-state-preparation/state_prep/tutorial>
```
