# Julia documentation website

The Julia tutorial in `src/` is built with Documenter.jl. The function reference
reads the docstrings in `simulations/src/QFTSimulations.jl`.
The Python demonstrations use [Sphinx with Furo](../python-docs/README.md).
Both sites link to one another and are published under the same repository URL.

From the repository root, install the documentation packages and build the
Julia tutorial with

```bash
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl
```

This uses a separate Julia environment from the simulations. Building the
documentation does not run the ground-state or time-evolution calculations.
The Julia tutorial builds independently of the Python site.

To view the Julia tutorial locally, run

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

and open <http://127.0.0.1:8000>. Git ignores `docs/build/`.

## Publishing

In the GitHub repository, open **Settings → Pages** and select **GitHub
Actions** as the source. `.github/workflows/documentation.yml` builds the
Julia tutorial and Python demonstrations separately, then combines their static
outputs. The Julia tutorial remains at <https://ujjwalbasumatary.github.io/qftonqc/>
and the Python demonstrations are at
<https://ujjwalbasumatary.github.io/qftonqc/python/>.
Pull requests check both builds. Publishing occurs from `main`.

After editing Markdown pages or Julia docstrings, run the Julia build again.
Missing docstrings and broken internal references cause that build to fail.
Edit the original `.ipynb` files for the Python demonstrations, following the
[Python documentation instructions](../python-docs/README.md).
