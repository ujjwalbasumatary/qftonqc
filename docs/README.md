# Documentation website

We use Documenter.jl to build the pages in `src/` and generate the function
reference from the docstrings in `simulations/src/QFTSimulations.jl`.

From the repository root, install the documentation packages and build the
site with

```bash
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl
```

We keep the documentation packages in a separate Julia environment. You
can build the site without running any ground-state or time-evolution
calculations.

To view the result locally, run

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

and open <http://127.0.0.1:8000>. Git ignores `docs/build/`.

## Publishing

In the GitHub repository, open **Settings → Pages** and select **GitHub
Actions** as the source. Once the documentation changes are on `main`,
`.github/workflows/documentation.yml` builds and publishes the site at
<https://ujjwalbasumatary.github.io/qftonqc/>. Pull requests build the pages
for checking; publishing occurs from `main`.

After editing Markdown pages or Julia docstrings, run the build again. Missing
docstrings and broken internal references cause the build to fail.
