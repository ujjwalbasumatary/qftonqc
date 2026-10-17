---
site:
  hide_outline: true
---

# Matrix product states

These four notebooks accompany Part III of the notes, Chapters 11–13. Each
implements the contractions and update steps directly in NumPy, with worked
algebra, tensor-network diagrams, numerical checks and exercises.

## Reading order

| Notebook | Calculation | Notes |
| --- | --- | --- |
| [Schmidt decomposition and canonical MPS](../course/tutorials/06-matrix-product-states/01_schmidt_canonical.ipynb) | Truncation, observable errors, sequential SVD, canonical forms and expectation values. | Chapter 11 |
| [MPOs and two-site DMRG](../course/tutorials/06-matrix-product-states/02_mpo_dmrg.ipynb) | The Ising MPO, contraction environments, effective Hamiltonians and complete optimization sweeps. | Chapter 11 |
| [TEBD and TDVP](../course/tutorials/06-matrix-product-states/03_time_evolution.ipynb) | Time evolution, SVD truncation, tangent-space projection and the backward bond step. | Chapter 13 |
| [Uniform MPS and VUMPS](../course/tutorials/06-matrix-product-states/04_vumps.ipynb) | Half-chain environments, site and bond equations, polar reconstruction and an entangled ground state. | Chapter 12 |

The finite-chain examples compare MPS contractions with dense states and
operators. The VUMPS example checks the local equations and the stationarity
of the energy on an infinite chain.

Each notebook defines its own functions and runs independently. Use the
**Download notebook** link at the top of its page, then select the NumPy kernel
from the [setup instructions](setup.md).


```{toctree}
:hidden:
:maxdepth: 1

Schmidt decomposition <../course/tutorials/06-matrix-product-states/01_schmidt_canonical>
MPOs and DMRG <../course/tutorials/06-matrix-product-states/02_mpo_dmrg>
TEBD and TDVP <../course/tutorials/06-matrix-product-states/03_time_evolution>
VUMPS <../course/tutorials/06-matrix-product-states/04_vumps>
```
