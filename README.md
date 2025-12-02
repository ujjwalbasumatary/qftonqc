# Quantum Field Theory on a Quantum Computer
Implementation of algorithms for HE381: Quantum Field Theory on a Quantum Computer taught by Prof. Aninda Sinha @ IISc during the fall 2025 semester.

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

All of these can be installed (globally) by opening the Julia REPL (Read-Evaluate-Print-Loop) by typing `Julia` in your shell. Hit `]` to enter the package manager and then type `add("PackageNameWithoutTheJl")` to install the package.
