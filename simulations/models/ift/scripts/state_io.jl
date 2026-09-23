"""
    IFTStateIO

Save and reload the finite MPS window used in the fixed-momentum Ising
collision. Each file contains the state, both infinite boundaries, the
vacuum and excitation tensors used in its preparation, and the physical
time. These objects allow subsequent overlaps and correlation functions to
be evaluated without repeating the collision.
"""
module IFTStateIO

using JLD2
using LinearAlgebra
using MPSKit
using TensorKit

export save_ift_state, load_ift_state, should_save_state, state_provenance

"""
    should_save_state(step, final_step, interval)

Return whether to save a state after `step` completed time steps. Step zero
is the initial state. The initial and final states are always saved; a
positive `interval` also saves every multiple of that number. With interval
zero, only the initial and final states are saved. If `final_step == 0`,
one file represents both. Negative intervals and steps outside
`0:final_step` throw `ArgumentError`.
"""
function should_save_state(step::Integer, final_step::Integer, interval::Integer)
    final_step >= 0 || throw(ArgumentError("final_step must be nonnegative"))
    0 <= step <= final_step || throw(ArgumentError("step must lie in 0:final_step"))
    interval >= 0 || throw(ArgumentError("save_every must be nonnegative"))
    return step == 0 || step == final_step || (interval > 0 && step % interval == 0)
end

"""
    state_provenance(project_directory, source_paths)

Record the Julia, MPSKit, TensorKit, and JLD2 versions, the contents of the
simulation Project and Manifest, and copies of the source files in
`source_paths`. Source copies describe the calculation even when it was
run with uncommitted edits. When Git is available, also record the current
commit and whether tracked files differ from it. Outside a Git checkout,
these two entries are `nothing`. This function reads files and Git metadata
without changing the environment or repository.
"""
function state_provenance(project_directory, source_paths)
    project_directory = abspath(project_directory)
    repository = dirname(project_directory)
    git_arguments = ["git", "-c", "safe.directory=$repository", "-C", repository]
    git_commit, git_dirty = try
        commit = readchomp(pipeline(Cmd(vcat(git_arguments, ["rev-parse", "HEAD"])); stderr=devnull))
        status = read(pipeline(Cmd(vcat(git_arguments, ["status", "--porcelain", "--untracked-files=no"])); stderr=devnull), String)
        (commit, !isempty(status))
    catch
        (nothing, nothing)
    end
    return Dict{String, Any}(
        "julia_version" => string(VERSION),
        "package_versions" => Dict(
            "MPSKit" => string(pkgversion(MPSKit)),
            "TensorKit" => string(pkgversion(TensorKit)),
            "JLD2" => string(pkgversion(JLD2)),
        ),
        "git_commit" => git_commit,
        "git_dirty" => git_dirty,
        "project_toml" => read(joinpath(project_directory, "Project.toml"), String),
        "manifest_toml" => isfile(joinpath(project_directory, "Manifest.toml")) ?
                           read(joinpath(project_directory, "Manifest.toml"), String) : nothing,
        "source_files" => Dict(relpath(path, repository) => read(path, String) for path in source_paths),
    )
end

"""
    save_ift_state(path, state; step, dt, parameters, reference,
                   energy_density, spin_density, provenance)

Write one `WindowMPS` and its preparation data to a new JLD2 file. Existing
files are never replaced. The file is first written beside its destination
under a temporary name, then renamed after JLD2 closes it. An interrupted
write therefore does not masquerade as a completed state file.

The stored keys are `format_version` (currently 1), `state`, `step`, `time`,
`state_norm`, `parameters`, `reference`, `energy_density`, `spin_density`,
and `provenance`. `step` counts completed TDVP steps and `time = step * dt`.
`state` retains its left and right infinite MPS boundaries. The state is
saved with its current norm, without another normalization or truncation.

The caller supplies `reference` with the vacuum, Hamiltonian, the dense
excitation tensors actually used to prepare the packets, their momenta and
energies, and the vacuum expectation values subtracted from the observables.
These preparation tensors cover only the incoming branch at the two central
momenta; a larger outgoing-particle basis can be calculated from the saved
vacuum and Hamiltonian. Keep the step-zero file for comparisons with the
initial two-particle sector.

`energy_density` contains the current `L-1` internal-bond values and
`spin_density` the current `L` site values; neither is a full time history.
The function checks these lengths, the nonnegative step, and finite positive
`dt`. It returns the absolute path of the completed file.
"""
function save_ift_state(
    path, state::WindowMPS;
    step::Integer, dt::Real, parameters, reference, energy_density, spin_density, provenance,
)
    step >= 0 || throw(ArgumentError("step must be nonnegative"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    L = length(state)
    length(energy_density) == L - 1 || throw(DimensionMismatch("expected L-1 internal bond energies"))
    length(spin_density) == L || throw(DimensionMismatch("expected L spin expectation values"))
    destination = abspath(path)
    ispath(destination) && throw(ArgumentError("state file already exists: $destination"))
    mkpath(dirname(destination))
    temporary, io = mktemp(dirname(destination); cleanup=false)
    close(io)
    try
        jldopen(temporary, "w") do file
            file["format_version"] = 1
            file["state"] = state
            file["step"] = step
            file["time"] = step * dt
            file["state_norm"] = norm(state)
            file["parameters"] = parameters
            file["reference"] = reference
            file["energy_density"] = collect(energy_density)
            file["spin_density"] = collect(spin_density)
            file["provenance"] = provenance
        end
        mv(temporary, destination; force=false)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end

"""
    load_ift_state(path)

Read a saved Ising collision state and return its string-keyed dictionary.
For example, `saved["state"]` is the `WindowMPS`, `saved["time"]` its physical
time, and `saved["reference"]["vacuum"]` the preparation vacuum. Checkpoint
format versions other than 1 and objects that do not reload as a `WindowMPS`
throw `ArgumentError`.

Use the Julia and package versions recorded in `saved["provenance"]` when
reloading these Julia objects; compatibility across future package versions
is not guaranteed. The function neither resumes time evolution nor evaluates
particle-sector probabilities. Only load files from a trusted calculation.
"""
function load_ift_state(path)
    saved = load(path)
    get(saved, "format_version", nothing) == 1 ||
        throw(ArgumentError("unrecognized Ising state file format"))
    get(saved, "state", nothing) isa WindowMPS ||
        throw(ArgumentError("the saved object did not load as a WindowMPS"))
    return saved
end

end
