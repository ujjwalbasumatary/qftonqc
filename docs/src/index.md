# Quantum field theory on a quantum computer

This repository grew out of HE381, *Quantum Field Theory on a Quantum
Computer*, taught by Prof. Aninda Sinha at IISc in the fall of 2025. It contains
the course notebooks and student projects, together with Julia calculations
of quantum fields in one spatial dimension.

The Julia programs use matrix product states to describe the vacuum and its
excitations. In the Ising and lattice phi-four calculations, two localized
wave packets approach each other and evolve under the interacting
Hamiltonian. The saved energy and field expectation values follow their
motion through the lattice. The Schwinger program prepares a finite chain
with a central source and evolves it after the source strength changes.

The physics follows two references:

- [Real-Time Scattering in Ising Field Theory using Matrix Product States](https://arxiv.org/abs/2411.13645).
- [High-Energy Collision of Quarks and Mesons in the Schwinger Model](https://arxiv.org/abs/2307.02522).

The model pages give the Hamiltonians and explain how the states are
constructed. [Running the calculations](running.md) contains the Julia
commands, and [Data and figures](data.md) describes the saved arrays.
The [function reference](reference/functions.md) includes the docstrings from
the shared Julia module.

The [course material](course.md) covers NumPy, numerical time evolution,
Qiskit, state preparation, and matrix product states. Student presentations
and reports are grouped by project alongside their notebooks.
