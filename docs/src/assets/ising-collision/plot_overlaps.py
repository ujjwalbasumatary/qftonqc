#!/usr/bin/env python3
"""Redraw the energy-correlation and three-reference figures from their CSVs.

Run ``python plot_overlaps.py`` in any working directory. By default, the
script reads ``three_energy_selected.csv`` and ``three_reference_weights.csv``
beside this file and writes PNG/PDF figures to the same directory. Python
and Matplotlib are required. LaTeX and dvipng supply the published lettering;
``--no-tex`` explicitly selects Matplotlib's built-in mathematical lettering
when those programs are unavailable.

``--data-directory`` and ``--output-directory`` accept alternative directories.
For example, ``python plot_overlaps.py --output-directory /tmp/ising-figures``
redraws the figures without replacing the published copies. Input CSVs are
never rewritten. Matplotlib's cache is kept inside the output directory.

The first figure shows the signed expectation value
``<delta_h_x delta_h_y delta_h_z>`` at lattice times 0, 70, 75, and 80.
Here ``delta_h_n = h_n - <h_n>_vac`` uses the symmetric two-site Ising energy
density, in units J=a=hbar=1. The outer bonds x,z are the unshifted energy
maxima in each half of the window; their positions are recorded in the CSV.
All middle-bond samples are retained, including negative initial values.

The second figure shows the Gram-corrected projection onto the specified
three-reference subspace, divided by the saved state's squared norm. Its
label is W_ref^(3), not P111: the excitation tensors are held at three
reference momenta and only configurations with the stated adjacent
separations are included. Initial reference leakage remains in the CSV;
the figure compares the three late times. Lines only join saved samples.
No smoothing, fitting, absolute values, or changes of normalization are used.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shutil


def read_measurements(path: Path, columns: tuple[str, ...]) -> list[dict[str, float]]:
    """Read required numerical columns and reject missing or nonfinite values.

    Extra columns remain in the input file but are not needed for drawing.
    Conversion to Python floats retains the double-precision measurements;
    no rounding or rescaling is applied.
    """
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        rows = [{column: float(row[column]) for column in columns} for row in reader]
    if not rows or not all(math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Empty or nonfinite measurements in {path}")
    return rows


def apply_style(plt, use_tex: bool) -> None:
    """Set the serif lettering, inward ticks, and sizes used in these figures."""
    plt.rcParams.update({
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif", "font.size": 9,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .8, "lines.linewidth": 1.25,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "black",
        "axes.grid": False, "legend.frameon": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def export(figure, stem: str, output: Path, plt) -> None:
    """Save a PDF and a 300-dpi PNG, then close the figure.

    Existing figures of the same names in the chosen output directory are
    replaced. Select another output directory to retain them.
    """
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=.06)
    plt.close(figure)


def plot_energy_correlations(rows, output: Path, plt) -> None:
    """Draw all signed, unshifted-anchor samples at the four supplied times.

    The linear ordinate displays both the negative initial tails and the
    differences between the three late-time curves. The initial curve is
    retained rather than subtracted from later curves.
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    if {row["time"] for row in rows} != {0, 70, 75, 80}:
        raise ValueError("The energy-correlation CSV must contain times 0, 70, 75, and 80.")
    colours = ("#111111", "#0072B2", "#E69F00", "#009E73")
    styles = ("--", "-", "-.", ":")
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    for time, colour, style in zip((0, 70, 75, 80), colours, styles):
        subset = sorted((row for row in rows if row["time"] == time),
                        key=lambda row: row["middle_bond"])
        if len({(row["left_bond"], row["right_bond"]) for row in subset}) != 1:
            raise ValueError(f"Multiple outer-anchor pairs found at time {time}.")
        if len({row["middle_bond"] for row in subset}) != len(subset):
            raise ValueError(f"Repeated middle bonds at time {time}.")
        ax.plot([row["middle_bond"] for row in subset], [row["triple"] for row in subset],
                color=colour, linestyle=style, label=rf"$t={time}$")
    ax.axhline(0, color="#6B6B6B", linewidth=.7, zorder=0)
    ax.set(xlabel=r"Middle bond $y$", ylabel=r"$\langle\delta h_x\,\delta h_y\,\delta h_z\rangle$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.minorticks_on()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=4,
              columnspacing=1.8, handlelength=2.7, borderaxespad=0)
    export(fig, "three_energy_correlations", output, plt)


def plot_reference_weights(rows, output: Path, plt) -> None:
    """Draw the late Gram-corrected weights for adjacent gaps 20, 30, 40, 50.

    Each plotted value already includes division by the saved state's
    squared norm. Initial leakage is neither subtracted nor used as a
    normalization. The four initial values are printed for comparison.
    """
    from matplotlib.ticker import MaxNLocator

    expected = {(time, gap) for time in (0, 70, 75, 80) for gap in (20, 30, 40, 50)}
    found = {(row["time"], row["minimum_adjacent_separation"]) for row in rows}
    if found != expected or len(rows) != len(expected):
        raise ValueError("Expected one reference weight for each of four times and four gaps.")
    fig, ax = plt.subplots(figsize=(4.6, 3.15), layout="constrained")
    colours = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    markers = ("o", "s", "^", "D")
    for gap, colour, marker in zip((20, 30, 40, 50), colours, markers):
        subset = sorted((row for row in rows if row["minimum_adjacent_separation"] == gap
                         and row["time"] > 0), key=lambda row: row["time"])
        values = [row["reference_projection_weight"] for row in subset]
        if not all(0 <= value <= .45 for value in values):
            raise ValueError("Reference weights fall outside the published figure's ordinate range.")
        ax.plot([row["time"] for row in subset], values, color=colour,
                marker=marker, markerfacecolor="white", label=rf"$g={gap}$")
        initial = next(row["reference_projection_weight"] for row in rows
                       if row["time"] == 0 and row["minimum_adjacent_separation"] == gap)
        print(f"Initial reference weight at g={gap}: {initial:.17g}")
    ax.set(xlabel=r"Evolved time $t$", ylabel=r"$W^{(3)}_{\rm ref}$",
           xlim=(69, 81), ylim=(0, .45), xticks=[70, 75, 80])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.minorticks_on()
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=2,
              columnspacing=1.7, handlelength=2.5, borderaxespad=0)
    export(fig, "three_reference_weights", output, plt)


def main() -> None:
    """Read the adjacent CSVs and generate both figures using the selected lettering."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-directory", type=Path, default=here,
                        help="Directory containing the two selected CSV files; defaults to this script's directory.")
    parser.add_argument("--output-directory", type=Path, default=here,
                        help="Directory for PNG/PDF figures; defaults to this script's directory.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Use Matplotlib mathematical lettering instead of external LaTeX.")
    args = parser.parse_args()
    if not args.no_tex and not all(shutil.which(command) for command in ("latex", "dvipng")):
        parser.error("LaTeX and dvipng are required; use --no-tex to select built-in mathematical lettering.")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt, not args.no_tex)
    correlations = read_measurements(args.data_directory / "three_energy_selected.csv",
                                     ("time", "left_bond", "middle_bond", "right_bond", "triple"))
    weights = read_measurements(args.data_directory / "three_reference_weights.csv",
                                ("time", "minimum_adjacent_separation", "reference_projection_weight"))
    plot_energy_correlations(correlations, output, plt)
    plot_reference_weights(weights, output, plt)
    print("Lettering: " + ("Matplotlib mathtext (--no-tex)" if args.no_tex else "LaTeX"))
    print(f"Wrote both PNG/PDF figures to {output}")


if __name__ == "__main__":
    main()
