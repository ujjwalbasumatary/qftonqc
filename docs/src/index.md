# QFT on a QC

This repository grew out of HE381, *Quantum Field Theory on a Quantum
Computer*, taught by Prof. Aninda Sinha at IISc in the fall of 2025, with
me (Ujjwal Basumatary) as the teaching assistant responsible for implementing the algorithms.
I currently maintain the repository.

It contains the course notebooks and student projects, together with Julia
calculations of quantum fields in one spatial dimension.

We use matrix product states to describe the vacuum and its excitations in
the Julia calculations. For the Ising and lattice ``\phi^4`` models, we
prepare two localized wave packets moving toward each other and evolve
them under the interacting Hamiltonian. We calculate the energy and field
expectation values to track their motion through the lattice. For the
Schwinger model, we prepare a finite chain with a central source, change
the source strength, and evolve the state under the new Hamiltonian.

The Ising calculations draw on
[Real-Time Scattering in Ising Field Theory using Matrix Product States](https://arxiv.org/abs/2411.13645).
The reference for the Schwinger model is
[High-Energy Collision of Quarks and Mesons in the Schwinger Model](https://arxiv.org/abs/2307.02522).

On the model pages, we introduce each Hamiltonian and explain how we
prepare the initial states. You can find the Julia commands in
[Running the calculations](running.md) and the definitions of the saved
arrays in [Data and figures](data.md).
If you want to change a Hamiltonian or measure another observable, we
explain where to make those changes in
[Working on the calculations](continuing.md). You can look up the shared
Julia functions and their arguments in the
[function reference](reference/functions.md).

In the [course material](course.md), you can find notebooks on NumPy,
numerical time evolution, Qiskit, state preparation, and matrix product
states. Student presentations and reports are grouped by project alongside
their notebooks.
