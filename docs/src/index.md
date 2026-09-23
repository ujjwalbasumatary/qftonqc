# QFT on a QC

This repository grew out of HE381, *Quantum Field Theory on a Quantum
Computer*, taught by Prof. Aninda Sinha at IISc in the fall of 2025. It contains
the course notebooks and student projects, together with Julia calculations
of quantum fields in one spatial dimension.

The Julia programs use matrix product states to describe the vacuum and its
excitations. In the Ising and lattice ``\phi^4`` calculations, two localized
wave packets approach each other and evolve under the interacting
Hamiltonian. The saved energy and field expectation values follow their
motion through the lattice. The Schwinger program prepares a finite chain
with a central source and evolves it after the source strength changes.

The Ising calculations draw on
[Real-Time Scattering in Ising Field Theory using Matrix Product States](https://arxiv.org/abs/2411.13645).
The reference for the Schwinger model is
[High-Energy Collision of Quarks and Mesons in the Schwinger Model](https://arxiv.org/abs/2307.02522).

The Hamiltonians and the construction of the initial states are described
on the model pages. You can find the Julia commands in
[Running the calculations](running.md) and the definitions of the saved
arrays in [Data and figures](data.md).
If you want to change a Hamiltonian or measure another observable,
[Working on the calculations](continuing.md) follows the state preparation,
evolution, and measurements through the Julia programs.
The [function reference](reference/functions.md) includes the docstrings from
the shared Julia module.

The [course material](course.md) covers NumPy, numerical time evolution,
Qiskit, state preparation, and matrix product states. Student presentations
and reports are grouped by project alongside their notebooks.
