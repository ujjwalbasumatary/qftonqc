using Documenter

# The shared module uses only Base Julia. Loading its source here lets the
# documentation include its docstrings without installing the simulation packages.
include(joinpath(@__DIR__, "..", "simulations", "src", "QFTSimulations.jl"))
using .QFTSimulations

makedocs(
    sitename = "Quantum field theory on a quantum computer",
    authors = "Ujjwal Basumatary and contributors",
    modules = [QFTSimulations],
    checkdocs = :all,
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://ujjwalbasumatary.github.io/qftonqc/",
        edit_link = "main",
        inventory_version = "main",
    ),
    pages = [
        "Introduction" => "index.md",
        "Repository" => "repository.md",
        "Running the calculations" => "running.md",
        "Physics" => [
            "Ising field theory" => "physics/ising.md",
            "Bosonized Schwinger model" => "physics/schwinger.md",
            "Lattice ``\\phi^4`` theory" => "physics/phi4.md",
            "Vacua and wave packets" => "physics/wave-packets.md",
            "Particle production" => "physics/particle-production.md",
        ],
        "Numerical comparisons" => "comparisons.md",
        "Data and figures" => "data.md",
        "Working on the calculations" => "continuing.md",
        "Julia reference" => [
            "Shared functions" => "reference/functions.md",
            "Model programs" => "reference/programs.md",
        ],
        "Course material" => "course.md",
    ],
)
