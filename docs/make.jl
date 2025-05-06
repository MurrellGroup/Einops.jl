using Einops
using Documenter

DocMeta.setdocmeta!(Einops, :DocTestSetup, :(using Einops); recursive=true)

makedocs(;
    modules=[Einops],
    authors="Anton Oresten <antonoresten@gmail.com> and contributors",
    sitename="Einops.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Einops.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Einops.jl",
    devbranch="main",
)
