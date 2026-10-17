# Run the notebooks

JupyterLab displays the notebook, while its selected kernel runs the code.
Keeping JupyterLab in the Conda `base` environment lets the NumPy and Qiskit
calculation environments share the same interface.

## JupyterLab and the NumPy kernel

Install JupyterLab in `base`:

```sh
conda install -n base -c conda-forge jupyterlab
```

Create the calculation environment once:

```sh
conda create -n qft-notebooks -c conda-forge python=3.12 numpy ipykernel
```

If `qft-notebooks` already exists, install its packages instead:

```sh
conda install -n qft-notebooks -c conda-forge numpy ipykernel
```

Register the environment as a Jupyter kernel:

```sh
conda activate qft-notebooks
python -m ipykernel install --user --name qft-notebooks --display-name "Python (QFT notebooks)"
```

This follows Jupyter's [Conda installation instructions](https://jupyter.org/install)
and IPython's [kernel registration instructions](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Qiskit kernel

Create a separate environment for the circuit demonstrations:

```sh
conda create -n qft-qiskit -c conda-forge python=3.12 pip
conda activate qft-qiskit
python -m pip install "qiskit[visualization]>=2,<3" qiskit-aer numpy scipy matplotlib ipykernel
python -m ipykernel install --user --name qft-qiskit --display-name "Python (QFT Qiskit)"
```

Qiskit's visualization extras provide the circuit drawings, and Qiskit Aer
provides the circuit and noise simulators. The package installation follows
[IBM's Qiskit guide](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit)
and [the Aer instructions](https://qiskit.github.io/qiskit-aer/getting_started.html).

The oscillator-dynamics notebook saves figures in a `plots/` folder beside
`qmwithqiskit.ipynb`. Create that folder before running its figure-saving cells.
This notebook and the noisy-spectrum example enable Matplotlib's
`text.usetex` setting. Those plot cells need a LaTeX installation, or you can
set `text.usetex` to `False` in their plotting configuration to use Matplotlib's
own math rendering.

The environment supplies the imported packages. The [Qiskit tutorial](qiskit.md)
identifies the older calculations that remain unfinished.

## Open a notebook

From the `qftonqc` repository root, launch JupyterLab:

```sh
conda activate base
jupyter lab course
```

If you downloaded a notebook separately, launch `jupyter lab` from its folder.
Select **Python (QFT notebooks)** for the NumPy MPS examples, or
**Python (QFT Qiskit)** for the circuit examples. Run the cells from the top so
that each function is defined before it is used. The four MPS notebooks are
independent. For the Qiskit notebooks, follow the prerequisites and execution
notes in the [Qiskit tutorial](qiskit.md).

The tensor-network diagrams are embedded in the notebooks. They remain visible
when a notebook is downloaded or moved to another folder.
