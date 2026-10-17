"""Build the Python tutorial directly from the original, saved notebooks."""
from pathlib import Path
import os

import pypandoc

# The pinned wheel supplies Pandoc on Windows, Linux and macOS. Only this
# documentation process needs its directory on PATH.
pandoc_directory = str(Path(pypandoc.get_pandoc_path()).parent)
os.environ["PATH"] = pandoc_directory + os.pathsep + os.environ.get("PATH", "")

project = "QFT on a QC"
author = "Ujjwal Basumatary and contributors"
extensions = ["myst_parser", "nbsphinx", "sphinx_design"]
root_doc = "index"
include_patterns = [
    'index.md',
    'python-docs/setup.md',
    'python-docs/qiskit.md',
    'python-docs/mps.md',
    'python-docs/supplementary.md',
    'course/examples/anharmonic-oscillator-spectrum/noisy_spectrum_qiskit.ipynb',
    'course/tutorials/03-qiskit-and-tfim/qiskitbasics.ipynb',
    'course/tutorials/03-qiskit-and-tfim/supplementary/statevector_estimator.ipynb',
    'course/tutorials/03-qiskit-and-tfim/supplementary/statevector_sampler.ipynb',
    'course/tutorials/03-qiskit-and-tfim/t3.ipynb',
    'course/tutorials/03-qiskit-and-tfim/tfim.ipynb',
    'course/tutorials/04-qft-and-phase-estimation/QFT.ipynb',
    'course/tutorials/04-qft-and-phase-estimation/hatQFTetc.ipynb',
    'course/tutorials/05-state-preparation/state_prep/qmwithqiskit.ipynb',
    'course/tutorials/05-state-preparation/state_prep/sho_adiabatic_state_prep.ipynb',
    'course/tutorials/05-state-preparation/state_prep/t6.ipynb',
    'course/tutorials/05-state-preparation/state_prep/tutorial.ipynb',
    'course/tutorials/06-matrix-product-states/01_schmidt_canonical.ipynb',
    'course/tutorials/06-matrix-product-states/02_mpo_dmrg.ipynb',
    'course/tutorials/06-matrix-product-states/03_time_evolution.ipynb',
    'course/tutorials/06-matrix-product-states/04_vumps.ipynb',
]
exclude_patterns = ["_build", "_site", "tmp", "**/.ipynb_checkpoints"]
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
nbsphinx_execute = "never"
html_theme = "furo"
html_title = "QFT on a QC"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_show_sphinx = False
html_show_copyright = False
html_theme_options = {"top_of_page_buttons": ["view"]}
mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"\left|#1\right\rangle", 1],
            "bra": [r"\left\langle#1\right|", 1],
            "braket": [r"\left\langle#1\right\rangle", 1],
        }
    }
}

html_static_path = ["_static"]
html_css_files = ["compact.css"]
myst_heading_anchors = 4

templates_path = ["_templates"]

# Saved outputs contain no widgets. No RequireJS runtime is needed.
nbsphinx_requirejs_path = ""
