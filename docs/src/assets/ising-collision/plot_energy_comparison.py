#!/usr/bin/env python3
"""Draw the integrable and non-integrable Ising collisions on one colour scale.

Supply each completed run with ``--integrable-directory`` and
``--nonintegrable-directory``. The script reads ``data/energy_exp.csv``,
``data/times.csv``, and ``plots/summary.json`` when all three are present.
Otherwise it reads the native ``data/energy_*.jld2`` file, using the optional
Python package h5py; the filename supplies the run parameters. It checks
that the fields are (h_x, h_z) = (1.06, 0) and (1.06, 0.006), respectively,
and that the packet, lattice, and MPS parameters agree between the two runs.
The stored excess energies already have each run's vacuum energy subtracted;
there is no further subtraction, smoothing, or normalization here.

Both panels use inferno with the same logarithmic colour limits: 1e-4 and
the largest excess energy in either run. Nonpositive samples are masked
because they cannot be represented on that scale. They and positive samples
below 1e-4 appear black. No measured columns are omitted: the last CSV column
is an unused zero column, because the simulation measures the L-1 internal
bonds of an L-site window. Its contents are checked before it is dropped.

PNG, PDF, and a JSON record of the inputs and plotting choices are written
beside this script, or to ``--output-directory``. Input files are not changed.
LaTeX is used by default; ``--no-tex`` explicitly selects Matplotlib's own
mathematical lettering when LaTeX and dvipng are unavailable. NumPy and
Matplotlib are required; h5py is needed only for native JLD2 input. A recorded
random seed is kept as metadata, but need not match between runs. Lettering
and export settings come from the adjacent
``plot_overlaps.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from plot_overlaps import apply_style, export


COMMON_PARAMETERS = (
    "h_x", "kappa", "sigma", "length", "n_center", "total_time",
    "bond_dimension", "evolution_bond_dimension", "time_step",
)
INTEGER_PARAMETERS = {
    "length", "n_center", "total_time", "bond_dimension", "evolution_bond_dimension",
}
PARAMETER_PATTERN = re.compile(
    r"scattering_infinite_mom_(?P<kappa>[^_]+)_hx_(?P<h_x>[^_]+)"
    r"_hz_(?P<h_z>[^_]+)_sigma_(?P<sigma>[^_]+)_L_(?P<length>[^_]+)"
    r"_n_(?P<n_center>[^_]+)_T_(?P<total_time>[^_]+)"
    r"_D_(?P<bond_dimension>[^_]+)_Dmax_(?P<evolution_bond_dimension>[^_]+)"
    r"_dt_(?P<time_step>[^_.]+)"
)


def native_times(dataset, count: int, np):
    """Read numeric times or the StepRangeLen saved by the Julia collision code.

    A saved range contains ``ref``, ``step``, ``len``, and a one-based
    ``offset``. Its reference and step each contain high and low Float64
    parts. Add these parts as exact rational numbers before rounding each
    time to Float64; adding them as ordinary floats first loses precision.
    Other compound layouts are rejected rather than inferred.
    """
    from fractions import Fraction

    if dataset.dtype.names is None:
        return np.asarray(dataset, dtype=float).reshape(-1)
    if dataset.shape != () or dataset.dtype.names != ("ref", "step", "len", "offset"):
        raise ValueError("Unsupported compound times dataset in native JLD2 input.")
    saved = dataset[()]
    if int(saved["len"]) != count or not 1 <= int(saved["offset"]) <= count:
        raise ValueError("Saved Julia time range has inconsistent length or offset.")
    parts = []
    for key in ("ref", "step"):
        if saved[key].dtype.names != ("hi", "lo"):
            raise ValueError("Unsupported Julia time-range reference or step.")
        parts.append(sum(Fraction.from_float(float(saved[key][part])) for part in ("hi", "lo")))
    reference, step = parts
    return np.array([float(reference + (index + 1 - int(saved["offset"])) * step)
                     for index in range(count)])


def read_native_run(directory: Path, np) -> tuple[dict, dict, object, object]:
    """Read the native JLD2 arrays and the parameters encoded in their filename.

    Exactly one energy file must be present. Julia's column-major matrix is
    stored with reversed HDF5 dimensions, so the L-by-T array is checked
    against the filename before transposition. h5py reads ``energy_exp`` and
    either a numeric ``times`` array or its stored Julia StepRangeLen fields.
    A seed in run.log is retained when available, but is not required.
    """
    paths = sorted((directory / "data").glob("energy_*.jld2"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native data/energy_*.jld2 file in {directory}")
    path = paths[0]
    match = PARAMETER_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot read the collision parameters from {path.name}")
    parameters = {
        key: int(value) if key in INTEGER_PARAMETERS
        else float(value.replace("p", ".").replace("m", "-"))
        for key, value in match.groupdict().items()
    }
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("Native JLD2 input needs h5py; install it in this Python environment "
                           "or supply the exported CSV files and plots/summary.json.") from error
    with h5py.File(path, "r") as saved:
        energy = np.asarray(saved["energy_exp"], dtype=float)
        times = native_times(saved["times"], parameters["total_time"], np)
    if energy.shape != (parameters["length"], parameters["total_time"]):
        raise ValueError(f"Native JLD2 dimensions disagree with its filename: {path}")
    sources = {"energy_and_times": path}
    log_path = directory / "run.log"
    if log_path.is_file():
        for line in log_path.read_text(errors="replace").splitlines():
            seed = re.fullmatch(r"Random seed: ([0-9]+)", line.strip())
            if seed:
                parameters["random_seed"] = int(seed[1])
                sources["seed_log"] = log_path
                break
    return parameters, sources, energy.T, times


def read_run(directory: Path, expected_hz: float, np) -> dict:
    """Read one run and check its labels, sampling, and unused energy column.

    CSV rows are times and columns are bonds 1 through L-1, followed by one
    unused column. When the CSV/summary set is incomplete, native JLD2 input
    is read with h5py and transposed to this same orientation. The parameters
    must describe the same number of
    times and sites as the arrays, with samples from t=0 in steps of dt.
    A missing field, nonfinite number, unexpected coupling, or inconsistent
    shape raises ValueError rather than assigning a misleading panel title.
    """
    sources = {
        "energy": directory / "data" / "energy_exp.csv",
        "times": directory / "data" / "times.csv",
        "summary": directory / "plots" / "summary.json",
    }
    if all(path.is_file() for path in sources.values()):
        with sources["summary"].open() as stream:
            parameters = json.load(stream)["parameters"]
        energy = np.loadtxt(sources["energy"], delimiter=",", ndmin=2)
        times = np.loadtxt(sources["times"], delimiter=",", ndmin=1)
    else:
        parameters, sources, energy, times = read_native_run(directory, np)
    for name in (*COMMON_PARAMETERS, "h_z"):
        value = parameters.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or nonfinite parameter {name!r} in {directory}")
    if parameters["h_x"] != 1.06 or parameters["h_z"] != expected_hz:
        raise ValueError(f"Expected h_x=1.06, h_z={expected_hz} in {directory}")
    for name in INTEGER_PARAMETERS:
        if int(parameters[name]) != parameters[name]:
            raise ValueError(f"Parameter {name} must be an integer in {directory}")
    if not (parameters["length"] > 2 and parameters["total_time"] > 1
            and parameters["sigma"] > 0 and parameters["time_step"] > 0
            and 0 < parameters["n_center"] < parameters["length"] / 2
            and parameters["bond_dimension"] > 0
            and parameters["evolution_bond_dimension"] >= parameters["bond_dimension"]):
        raise ValueError(f"Invalid lattice, packet, or MPS parameters in {directory}")
    expected_shape = (int(parameters["total_time"]), int(parameters["length"]))
    if energy.shape != expected_shape or times.shape != (expected_shape[0],):
        raise ValueError(f"Arrays do not match the saved dimensions in {directory}")
    if not np.isfinite(energy).all() or not np.isfinite(times).all():
        raise ValueError(f"Nonfinite measurements in {directory}")
    expected_times = np.arange(expected_shape[0]) * parameters["time_step"]
    if not np.all(np.diff(times) > 0) or not np.allclose(times, expected_times,
                                                      rtol=1e-12, atol=1e-12):
        raise ValueError(f"Time samples disagree with the saved time step in {directory}")
    if not np.all(energy[:, -1] == 0):
        raise ValueError(f"The last energy column is not unused and zero in {directory}")
    return {"directory": directory, "sources": sources, "parameters": parameters,
            "energy": energy[:, :-1], "times": times, "raw_shape": list(energy.shape)}


def input_record(run: dict, repository: Path, np) -> dict:
    """Record relative input paths, SHA-256 hashes, parameters, and data ranges.

    Hashes refer to the files as read, including the saved summary or native
    JLD2 file used to check the panel labels. The JSON contains no copied measurement arrays.
    Source paths are relative to the repository containing this script,
    including when an input directory is outside it.
    """
    sources = {}
    for name, path in run["sources"].items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sources[name] = {"path": os.path.relpath(path, repository), "sha256": digest}
    energy, times = run["energy"], run["times"]
    return {
        "directory": os.path.relpath(run["directory"], repository),
        "parameters": run["parameters"],
        "sources": sources,
        "raw_energy_shape": run["raw_shape"],
        "plotted_energy_shape": list(energy.shape),
        "omitted_column": {"one_based_index": run["raw_shape"][1],
                           "reason": "Unused final column; every entry is exactly zero."},
        "initial_time": float(times[0]), "final_time": float(times[-1]),
        "energy_min": float(energy.min()), "energy_max": float(energy.max()),
        "nonpositive_samples": int(np.count_nonzero(energy <= 0)),
        "positive_samples_below_colour_minimum": int(np.count_nonzero(
            (energy > 0) & (energy < 1e-4))),
    }


def draw_comparison(runs: tuple[dict, dict], output: Path, plt, np) -> tuple[float, float]:
    """Draw the saved positive excess energies with shared axes and LogNorm.

    The mesh retains every time and every internal bond; no interpolation or
    smoothing is added. Only the colour assignment is logarithmic. The same
    LogNorm object and colour map are used for both panels and the colour bar.
    The exported PDF rasterizes the dense meshes, keeping labels as vectors.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext, MaxNLocator

    vmin, vmax = 1e-4, max(float(run["energy"].max()) for run in runs)
    if vmax <= vmin:
        raise ValueError("No excess energy exceeds the requested lower colour limit.")
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    fig = plt.figure(figsize=(7.0, 3.65), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .045))
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1], sharex=left, sharey=left)
    right.tick_params(labelleft=False)
    colour_axis = fig.add_subplot(grid[0, 2])
    labels = ("(a) Integrable", "(b) Non-integrable")
    for ax, run, label in zip((left, right), runs, labels):
        energy, times = run["energy"], run["times"]
        bonds = np.arange(1, energy.shape[1] + 1)
        mesh = ax.pcolormesh(bonds, times, np.ma.masked_less_equal(energy, 0),
                             cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        hx, hz = run["parameters"]["h_x"], run["parameters"]["h_z"]
        ax.set_title(label + "\n" + rf"$h_x={hx:g},\quad h_z={hz:g}$", loc="left", pad=8)
        ax.set(xlabel=r"Bond $n$", xlim=(0, run["parameters"]["length"]),
               ylim=(times[0], times[-1]))
        ax.set_facecolor("black")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.minorticks_on()
    left.set_ylabel(r"Lattice time $t$")
    bar = fig.colorbar(mesh, cax=colour_axis, extend="min",
                       format=LogFormatterMathtext(base=10))
    bar.set_label(r"$\delta e_n(t)$")
    bar.set_ticks([1e-4, 1e-3, 1e-2])
    export(fig, "energy-comparison", output, plt)
    return vmin, vmax


def main() -> None:
    """Check both runs, draw their energies, and save the input and colour record."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrable-directory", type=Path, required=True,
                        help="Completed h_x=1.06, h_z=0 run; native JLD2 needs h5py, or use CSV/summary.")
    parser.add_argument("--nonintegrable-directory", type=Path, required=True,
                        help="Completed h_x=1.06, h_z=0.006 run; native JLD2 needs h5py, or use CSV/summary.")
    parser.add_argument("--output-directory", type=Path, default=here,
                        help="Directory for the PDF, PNG, and JSON; defaults beside this script.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Use Matplotlib mathematical lettering instead of external LaTeX.")
    args = parser.parse_args()
    if not args.no_tex and not all(shutil.which(command) for command in ("latex", "dvipng")):
        parser.error("LaTeX and dvipng are required; use --no-tex for built-in lettering.")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = (read_run(args.integrable_directory.resolve(), 0, np),
            read_run(args.nonintegrable_directory.resolve(), .006, np))
    for name in COMMON_PARAMETERS:
        if runs[0]["parameters"][name] != runs[1]["parameters"][name]:
            raise ValueError(f"The two runs have different {name} parameters.")
    if not np.array_equal(runs[0]["times"], runs[1]["times"]):
        raise ValueError("The two runs have different saved times.")
    apply_style(plt, not args.no_tex)
    vmin, vmax = draw_comparison(runs, output, plt, np)
    repository = here.parents[3]
    metadata = {
        "observable": "Internal bond energy minus its value in the vacuum, delta_e_n(t).",
        "orientation": "Rows are times, columns are internal bonds 1 through L-1.",
        "panels": {
            "a": {"label": "Integrable", **input_record(runs[0], repository, np)},
            "b": {"label": "Non-integrable", **input_record(runs[1], repository, np)},
        },
        "colour_scale": {"normalization": "LogNorm", "colormap": "inferno",
                         "vmin": vmin, "vmax": vmax, "shared_between_panels": True,
                         "nonpositive_and_under_range_colour": "black"},
        "units": "J = a = hbar = 1",
        "text_rendering": "Matplotlib mathtext (--no-tex)" if args.no_tex else "LaTeX",
        "data_operations": "Omit verified unused zero column; mask nonpositive values for LogNorm."
                           " No smoothing, interpolation, rescaling, or extra vacuum subtraction.",
    }
    with (output / "energy-comparison.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Shared colour limits: {vmin:.17g} to {vmax:.17g}")
    print(f"Wrote energy-comparison.png, .pdf, and .json to {output}")


if __name__ == "__main__":
    main()
