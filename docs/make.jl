push!(LOAD_PATH,"../src/")

using Documenter, KAPseudospectra

makedocs(
    sitename = "KAPseudospectra.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://dan123222123.github.io/KAPseudospectra.jl",
        assets = String[],
    ),
    modules = [KAPseudospectra],
    pages = [
        "Home" => "index.md",
        "Pseudospectra Theory" => "theory.md",
        "Examples" => [
            "Standard Usage" => "standard_example.md",
            "Multi-GPU Computation" => "multigpu_example.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/dan123222123/KAPseudospectra.jl.git",
    devbranch = "main",
)
