# Quantum Field Theory on a Quantum Computer
Implementation of algorithms for HE381: Quantum Field Theory on a Quantum Computer taught by Prof. Aninda Sinha @ IISc during the fall 2025 semester.

## To run the Python scripts and notebooks
- Go to <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html> to install a Conda distribution. We used Miniconda during our tutorials.
- Create a virtual environment (see details here <https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html>) and install the relevant packages.

- We will need
    - `NumPy` for linear algebra.
    - `SciPy` for numerical linear algebra.
    - `Qiskit` to build and simulate quantum circuits.
- All the above can be installed by typing (in your environment) `pip install package-name-in-lowercase`. In addition, we will need
    - Qiskit-aer (`qiskit_aer`) to build noise models.
    - `MatPlotLib` for plotting data.

## To run the Julia scripts and notebooks
- Go to <https://julialang.org/downloads/> to install Julia.  
  - The basic distribution does not come with all the packages that are needed.  
  - `LinearAlgebra.jl` is available by default.

- We are going to need the following packages:
  - `MPSKit.jl` and `TensorKit.jl` for the MPS calculations  
  - `Plots.jl` for nice and beautiful plots  
  - `LaTeXStrings.jl` for LaTeX strings in figures  
  - `JLD2.jl` to save data (this is not necessary; I just like this package)
  - `ArgParse.jl` for argument parsing from the command line.
  - `IJulia.jl` for interactive Julia via Jupyter Lab or Notebook.

All of these can be installed (globally) by opening the Julia REPL (Read-Evaluate-Print-Loop) by typing `Julia` in your shell. Hit `]` to enter the package manager and then type `add("PackageNameWithoutTheJl")` to install the package. Alternatively from the REPL (or a Jupyter instance once you have `IJulia` installed)
```julia
using Pkg
Pkg.add("PackageNameWithoutTheJl")
```
