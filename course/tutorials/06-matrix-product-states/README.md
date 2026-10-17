# Matrix product states with NumPy

These four notebooks work through the MPS algorithms in Part III of the
[lecture notes](https://arxiv.org/abs/2512.02706).
Each combines algebra with short executable cells and embedded tensor diagrams.
The numerical code uses NumPy and the Python standard library. All functions
needed by a notebook are defined inside it, so the notebooks run independently
and require no tensor-network package or Julia installation.

## Reading order

| Notebook | Calculation and independent checks |
| --- | --- |
| [01 · Schmidt decomposition and canonical MPS](01_schmidt_canonical.ipynb) | Schmidt probabilities, truncation and observable errors, sequential SVD, GHZ and W states, QR motion of the canonical center, and observable contractions checked against dense states. |
| [02 · MPOs and two-site DMRG](02_mpo_dmrg.ipynb) | An Ising MPO, left and right environments, local effective Hamiltonians, both sweep directions, and ground-state comparisons with independent exact diagonalization. |
| [03 · Time evolution](03_time_evolution.ipynb) | Second-order TEBD with SVD truncation and a complete symmetric one-site TDVP sweep, including the backward bond step. Dense exact evolution, an independently assembled tangent projector, and time-reversal checks separate integration and fixed-rank errors. |
| [04 · Uniform MPS and VUMPS](04_vumps.ipynb) | Transfer maps, energy-subtracted half-chain environments, site and bond effective actions, paired polar reconstruction, and converged calculations at bond dimensions one and two. The entangled example checks local contractions, residuals, and independent energy derivatives. |

The examples use small arrays so that contractions, tensor shapes, and complex
conjugations can be inspected. Dense many-body calculations supply independent
checks for the finite-chain examples. Increasing the system size eventually
makes these checks too expensive even when the MPS contractions remain feasible.
Each notebook ends with concrete parameter variations and exercises.

## Run the notebooks

Use Python 3.10 or later and NumPy 1.26 or later. JupyterLab provides the notebook
interface. From the repository root, create a Python environment:

```sh
python -m venv .venv-mps
```

On Linux or macOS:

```sh
.venv-mps/bin/python -m pip install "numpy>=1.26" jupyterlab
.venv-mps/bin/python -m jupyterlab course/tutorials/06-matrix-product-states
```

On Windows, run these commands in PowerShell or Command Prompt:

```powershell
.venv-mps\Scripts\python.exe -m pip install "numpy>=1.26" jupyterlab
.venv-mps\Scripts\python.exe -m jupyterlab course/tutorials/06-matrix-product-states
```

Open a numbered notebook and run its cells from the top. To check that it is
self-contained, restart the kernel and run all cells. The committed notebooks
include their executed outputs. Diagrams are stored as notebook attachments,
so no external image files or diagram-rendering programs are needed to read them.

The [programming tutorial](https://ujjwalbasumatary.github.io/qftonqc/examples/mps-numpy/)
provides the same notebook map alongside the Julia simulation examples.

## Earlier classroom notebooks

[`mps_demo.ipynb`](mps_demo.ipynb) and
[`MPS_demo_fin.ipynb`](MPS_demo_fin.ipynb) are the earlier classroom demonstrations.
They remain available alongside the four worked algorithm notebooks.
