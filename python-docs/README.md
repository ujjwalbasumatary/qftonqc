# Python notebook documentation

The Python tutorial uses Sphinx, the Furo theme and nbsphinx. Its configuration is
[`conf.py`](conf.py). The small repository-root `index.md` includes
[`index.md`](index.md) here to place the homepage at the tutorial root. The sixteen notebooks under `course/tutorials/` and
`course/examples/` supply both the rendered pages and the downloadable `.ipynb`
files. They are read directly from their existing locations.

The Julia simulation tutorial has its own Documenter.jl build in `docs/`.

## Build locally

Use Python 3.11 or later. For example, create a documentation environment with
Conda:

```sh
conda create -n qft-docs -c conda-forge python=3.12 pip
conda activate qft-docs
python -m pip install -r python-docs/requirements.txt
```

The pinned `pypandoc-binary` package includes Pandoc. The configuration adds that
executable to the build process's search path on Windows, Linux and macOS.

Run the following from the repository root:

```sh
sphinx-build -c python-docs -b dirhtml -d _build/doctrees -W --keep-going . _build/html
```

The static website is written to `_build/html/`. To preview it locally:

```sh
python -m http.server 8000 --directory _build/html
```

Open `http://localhost:8000` in a browser. Rebuild after editing a page.

## Repository website

The documentation workflow combines the Python output under the `python/`
subdirectory of the Julia tutorial. Sphinx generates relative links, so the same
HTML works at the website's `/qftonqc/python/` path and in the local preview.
The Python build does not need Node.js or a `BASE_URL` setting.

## Saved outputs and additions

`nbsphinx_execute = "never"` disables notebook execution during documentation
builds. Pages show the saved outputs, including any saved errors described in
the tutorial introduction. To refresh a calculation, run and save the notebook in
JupyterLab, then rebuild the site.

To add a notebook, put its repository-relative path in `include_patterns` in
`conf.py` and link it from the appropriate tutorial's `toctree`. This explicit list
keeps student projects and exploratory notebooks outside the website. Keep the
algebra, diagrams, code and saved output in the notebook itself. Give each
diagram a distinct description in its Markdown image alt text.

Each notebook page has a **Download notebook** link to its original `.ipynb`
file. The notebook remains readable in JupyterLab, including its embedded diagrams.

## Navigation and appearance

The landing page leads to the circuit and MPS tutorials. Supplementary
notebooks have a separate page. `_templates/page.html` supplies the
fixed titlebar, and `_static/compact.css` sets the compact layout. The native
navigation, search, page contents and theme switch remain available.

See the official documentation for
[Furo](https://pradyunsg.me/furo/),
[nbsphinx configuration](https://nbsphinx.readthedocs.io/en/latest/configuration.html),
[notebook execution](https://nbsphinx.readthedocs.io/en/latest/executing-notebooks.html)
and [MyST Markdown syntax](https://myst-parser.readthedocs.io/en/latest/syntax/organising_content.html).
