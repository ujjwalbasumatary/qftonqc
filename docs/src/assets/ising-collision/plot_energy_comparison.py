#!/usr/bin/env python3
"""Draw completed Ising collisions on one colour scale.

For a comparison of packet widths or couplings, repeat ``--run-directory``
for each run, in panel order. These runs must have the same lattice length
and saved times. Each panel lists its own fields, width, and momentum.
``--signed`` adds a signed-energy figure; ``--spin`` adds a figure of
<sigma_z> minus its value in the vacuum, read from the native spin files.
Both keep negative values. Python 3.11 or later is needed when reading TOML.

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
to ``--output-directory``. This directory is required with --run-directory;
the integrable/non-integrable pair otherwise defaults beside this script.
Input files are not changed.
LaTeX is used by default; ``--no-tex`` explicitly selects Matplotlib's own
mathematical lettering when LaTeX and dvipng are unavailable. NumPy and
Matplotlib are required; h5py is needed only for native JLD2 input. A recorded
random seed is kept as metadata, but need not match between runs. Lettering
and export settings come from the adjacent
``plot_overlaps.py``.
"""

from __future__ import annotations

import argparse
import csv
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


def read_run(directory: Path, expected_hz: float | None, np) -> dict:
    """Read one run and check its labels, sampling, and unused energy column.

    CSV rows are times and columns are bonds 1 through L-1, followed by one
    unused column. When the CSV/summary set is incomplete, native JLD2 input
    is read with h5py and transposed to this same orientation. The parameters
    must describe the same number of
    times and sites as the arrays, with samples from t=0 in steps of dt.
    A missing field, nonfinite number, unexpected coupling, or inconsistent
    shape raises ValueError rather than assigning a misleading panel title.
    With ``expected_hz=None``, either field may vary between runs. If a
    run-settings.toml file is present, its parameters must agree with the
    data filename or CSV summary; its random seed is also retained.
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
    if expected_hz is not None and (parameters["h_x"] != 1.06
                                    or parameters["h_z"] != expected_hz):
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
    settings_path = directory / "run-settings.toml"
    if settings_path.is_file():
        import tomllib
        with settings_path.open("rb") as stream:
            settings = tomllib.load(stream)
        if "parameters" in settings:
            saved_parameters = settings["parameters"]
            final_time = settings["final_time"]
        else:
            names = {
                "h_x": "hx", "h_z": "hz", "kappa": "central_momentum",
                "sigma": "gaussian_amplitude_width", "length": "window_length",
                "n_center": "left_packet_center", "total_time": "number_of_saved_times",
                "bond_dimension": "vacuum_bond_dimension",
                "evolution_bond_dimension": "maximum_evolution_bond_dimension",
                "time_step": "time_step",
            }
            saved_parameters = {key: settings["collision"][name] for key, name in names.items()}
            final_time = settings["collision"]["final_time"]
        for name in (*COMMON_PARAMETERS, "h_z"):
            if saved_parameters.get(name) != parameters[name]:
                raise ValueError(f"Saved settings disagree with the data's {name} in {directory}")
        if not np.isclose(final_time, times[-1], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Saved final time disagrees with the data in {directory}")
        parameters["random_seed"] = settings["random_seed"]
        sources["settings"] = settings_path
    return {"directory": directory, "sources": sources, "parameters": parameters,
            "energy": energy[:, :-1], "times": times, "raw_shape": list(energy.shape)}


def read_spin(run: dict, np) -> None:
    """Add every site's signed spin difference to a run, without rescaling it.

    The simulation already subtracts the uniform vacuum's magnetization.
    Unlike the energy array, the spin array has no unused column. Its file
    must name the same parameters and contain exactly the same times as the
    energy file. The returned array has time along rows and site along columns.
    """
    import h5py
    paths = sorted((run["directory"] / "data").glob("sz_value_*.jld2"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native spin file in {run['directory']}")
    path = paths[0]
    match = PARAMETER_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot read spin-file parameters from {path}")
    for name, value in match.groupdict().items():
        if float(value.replace("p", ".").replace("m", "-")) != run["parameters"][name]:
            raise ValueError(f"Spin-file parameter {name} disagrees with energy in {path}")
    with h5py.File(path, "r") as saved:
        spin = np.asarray(saved["s_z_exp"], dtype=float).T
        times = native_times(saved["times"], len(run["times"]), np)
    if spin.shape != tuple(run["raw_shape"]) or not np.isfinite(spin).all():
        raise ValueError(f"Unexpected dimensions or nonfinite spin values in {path}")
    if not np.array_equal(times, run["times"]):
        raise ValueError(f"Spin and energy times differ in {path}")
    run["spin"] = spin
    run["sources"]["spin_and_times"] = path


def check_saved_energies(run: dict, np) -> dict | None:
    """Compare bond-energy sums with the CSV written when MPS states were saved.

    This reads only checkpoints.csv, not the MPS files. It checks every
    recorded lattice time and energy sum and returns the final norm and
    energy change for the figure's JSON. It does not alter the measurements
    or divide them by the state norm a second time.
    """
    path = run["directory"] / "checkpoints.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty checkpoint table in {path}")
    totals = run["energy"].sum(axis=1)
    indices = [int(row["step"]) for row in rows]
    times = np.array([float(row["time"]) for row in rows])
    saved = np.array([float(row["internal_energy"]) for row in rows])
    if (indices[0] != 0 or indices[-1] != len(totals) - 1
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError(f"Incomplete or unordered checkpoint times in {path}")
    if not np.allclose(times, run["times"][indices], rtol=1e-12, atol=1e-12):
        raise ValueError(f"Checkpoint times disagree with the arrays in {path}")
    if not np.allclose(saved, totals[indices], rtol=1e-11, atol=1e-11):
        raise ValueError(f"Checkpoint energies disagree with the arrays in {path}")
    run["sources"]["checkpoints"] = path
    return {"rows_checked": len(rows),
            "maximum_energy_sum_difference": float(np.max(np.abs(saved - totals[indices]))),
            "final_norm_squared": float(rows[-1]["norm_squared"]),
            "final_relative_energy_change": float(rows[-1]["relative_energy_change"]),
            "final_edge_energy_fraction": float(rows[-1]["edge_energy_fraction"])}


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


def draw_comparison(runs, output: Path, plt, np, *, labels=None,
                    field="energy", scale="log", stem="energy-comparison") -> tuple[float, float]:
    """Draw two or more runs with one colour scale and the same spatial/time axes.

    The mesh retains every time and every internal bond; no interpolation or
    smoothing is added. Logarithms affect colour assignment, not the data. The same
    colour normalization and map are used for every panel and the colour bar.
    The exported PDF rasterizes the dense meshes, keeping labels as vectors.

    Positive energy uses inferno and LogNorm with a lower limit of 1e-4.
    Signed energy uses a symmetric blue-white-red scale, linear between
    -1e-4 and 1e-4 and logarithmic beyond them. Spin uses that diverging map
    on a symmetric linear scale. No absolute values replace the plotted data.
    Absolute values are used only to choose symmetric colour limits.
    When labels are omitted, each panel gives its fields, sigma and kappa.
    """
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    from matplotlib.ticker import LogFormatterMathtext, MaxNLocator

    if scale == "log":
        vmin, vmax = 1e-4, max(float(run[field].max()) for run in runs)
        if vmax <= vmin:
            raise ValueError("No excess energy exceeds the requested lower colour limit.")
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("black")
        cmap.set_under("black")
    elif scale in ("symlog", "linear"):
        vmax = max(float(np.abs(run[field]).max()) for run in runs)
        if vmax == 0:
            raise ValueError(f"All {field} samples are zero.")
        vmin = -vmax
        norm = (SymLogNorm(linthresh=1e-4, vmin=vmin, vmax=vmax, base=10)
                if scale == "symlog" else Normalize(vmin=vmin, vmax=vmax))
        cmap = plt.get_cmap("RdBu_r")
    else:
        raise ValueError(f"Unknown colour scale: {scale}")
    fig = plt.figure(figsize=(max(7.0, 3.0 * len(runs)), 3.65), layout="constrained")
    grid = fig.add_gridspec(1, len(runs) + 1, width_ratios=(*([1] * len(runs)), .045))
    left = fig.add_subplot(grid[0, 0])
    axes = [left] + [fig.add_subplot(grid[0, index], sharex=left, sharey=left)
                     for index in range(1, len(runs))]
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    colour_axis = fig.add_subplot(grid[0, -1])
    for index, (ax, run) in enumerate(zip(axes, runs)):
        values, times = run[field], run["times"]
        positions = np.arange(1, values.shape[1] + 1)
        shown = np.ma.masked_less_equal(values, 0) if scale == "log" else values
        mesh = ax.pcolormesh(positions, times, shown,
                             cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        p = run["parameters"]
        fields = rf"$h_x={p['h_x']:g},\quad h_z={p['h_z']:g}$"
        title = (f"({chr(97 + index)}) {labels[index]}\n" + fields if labels is not None
                 else f"({chr(97 + index)}) " + fields + "\n" +
                 rf"$\sigma={p['sigma']:g},\quad\kappa={p['kappa']:.4g}$")
        ax.set_title(title, loc="left", pad=8)
        ax.set(xlabel=r"Bond $n$" if field == "energy" else r"Site $n$",
               xlim=(0, p["length"]),
               ylim=(times[0], times[-1]))
        ax.set_facecolor("black" if scale == "log" else "white")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.minorticks_on()
    left.set_ylabel(r"Lattice time $t$")
    bar = fig.colorbar(mesh, cax=colour_axis, extend="min" if scale == "log" else "neither")
    bar.set_label(r"$\delta e_n(t)$" if field == "energy" else r"$\delta\langle\sigma_n^z(t)\rangle$")
    if scale in ("log", "symlog"):
        ticks = [10. ** exponent for exponent in range(-4, math.floor(math.log10(vmax)) + 1)]
        bar.set_ticks(ticks if scale == "log" else [-x for x in reversed(ticks)] + [0.] + ticks)
        bar.formatter = LogFormatterMathtext(base=10)
        bar.update_ticks()
    export(fig, stem, output, plt)
    return vmin, vmax


def main() -> None:
    """Read runs, compare their saved arrays, and export figures with input hashes."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrable-directory", type=Path,
                        help="Completed h_x=1.06, h_z=0 run; native JLD2 needs h5py, or use CSV/summary.")
    parser.add_argument("--nonintegrable-directory", type=Path,
                        help="Completed h_x=1.06, h_z=0.006 run; native JLD2 needs h5py, or use CSV/summary.")
    parser.add_argument("--run-directory", type=Path, action="append",
                        help="Repeat for each completed run in a width or coupling comparison.")
    parser.add_argument("--signed", action="store_true", help="Also draw signed bond energies.")
    parser.add_argument("--spin", action="store_true", help="Also draw the saved spin differences.")
    parser.add_argument("--output-directory", type=Path,
                        help="Directory for PDF, PNG, and JSON; required with --run-directory.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Use Matplotlib mathematical lettering instead of external LaTeX.")
    args = parser.parse_args()
    if args.run_directory:
        if args.integrable_directory or args.nonintegrable_directory:
            parser.error("Use --run-directory or the integrable/nonintegrable pair, not both.")
        if not 2 <= len(args.run_directory) <= 4:
            parser.error("Supply two to four --run-directory arguments.")
        if args.output_directory is None:
            parser.error("Supply --output-directory to keep these figures separate from the published pair.")
    elif not (args.integrable_directory and args.nonintegrable_directory):
        parser.error("Supply both integrable/nonintegrable directories, or repeat --run-directory.")
    if not args.no_tex and not all(shutil.which(command) for command in ("latex", "dvipng")):
        parser.error("LaTeX and dvipng are required; use --no-tex for built-in lettering.")
    output = (args.output_directory or here).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.run_directory:
        runs = tuple(read_run(path.resolve(), None, np) for path in args.run_directory)
        labels, required_equal = None, ("length",)
    else:
        runs = (read_run(args.integrable_directory.resolve(), 0, np),
                read_run(args.nonintegrable_directory.resolve(), .006, np))
        labels, required_equal = ("Integrable", "Non-integrable"), COMMON_PARAMETERS
    for other in runs[1:]:
        for name in required_equal:
            if runs[0]["parameters"][name] != other["parameters"][name]:
                raise ValueError(f"The runs have different {name} parameters.")
        if not np.array_equal(runs[0]["times"], other["times"]):
            raise ValueError("The runs have different saved times.")
    checks = [check_saved_energies(run, np) for run in runs]
    if args.spin:
        for run in runs:
            read_spin(run, np)
    apply_style(plt, not args.no_tex)
    vmin, vmax = draw_comparison(runs, output, plt, np, labels=labels)
    repository = here.parents[3]
    metadata = {
        "observable": "Internal bond energy minus its value in the vacuum, delta_e_n(t).",
        "orientation": "Rows are times, columns are internal bonds 1 through L-1.",
        "panels": {
            chr(97 + index): {"label": labels[index] if labels else run["directory"].name,
                             **input_record(run, repository, np), "checkpoint_comparison": checks[index]}
            for index, run in enumerate(runs)
        },
        "colour_scale": {"normalization": "LogNorm", "colormap": "inferno",
                         "vmin": vmin, "vmax": vmax, "shared_between_panels": True,
                         "nonpositive_and_under_range_colour": "black"},
        "units": "J = a = hbar = 1",
        "text_rendering": "Matplotlib mathtext (--no-tex)" if args.no_tex else "LaTeX",
        "data_operations": "Omit verified unused zero column; mask nonpositive values for LogNorm."
                           " No smoothing, interpolation, rescaling, or extra vacuum subtraction.",
    }
    if args.signed:
        low, high = draw_comparison(runs, output, plt, np, labels=labels,
                                   scale="symlog", stem="energy-signed-comparison")
        metadata["signed_energy_colour_scale"] = {
            "normalization": "SymLogNorm", "linthresh": 1e-4, "colormap": "RdBu_r",
            "vmin": low, "vmax": high, "shared_between_panels": True}
    if args.spin:
        low, high = draw_comparison(runs, output, plt, np, labels=labels, field="spin",
                                   scale="linear", stem="spin-comparison")
        metadata["spin"] = {
            "observable": "Local expectation of sigma_z minus its value in the vacuum.",
            "orientation": "Rows are times, columns are all sites 1 through L.",
            "colour_scale": {"normalization": "Normalize", "colormap": "RdBu_r",
                             "vmin": low, "vmax": high, "shared_between_panels": True}}
    with (output / "energy-comparison.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Shared colour limits: {vmin:.17g} to {vmax:.17g}")
    print(f"Wrote energy-comparison.png, .pdf, and .json to {output}")


if __name__ == "__main__":
    main()
