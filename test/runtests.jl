using Einops
using Test

using Pkg

const EINOPS_TEST_ZYGOTE = get(ENV, "EINOPS_TEST_ZYGOTE", "false") == "true"
EINOPS_TEST_ZYGOTE && Pkg.add("Zygote")

const EINOPS_TEST_REACTANT = VERSION >= v"1.10" && get(ENV, "EINOPS_TEST_REACTANT", "false") == "true"
EINOPS_TEST_REACTANT && Pkg.add("Reactant")

@testset "Einops.jl" begin

    include("patterns.jl")
    include("parse_shape.jl")
    include("rearrange.jl")
    include("reduce.jl")
    include("repeat.jl")
    include("pack_unpack.jl")
    include("robustness.jl")

    EINOPS_TEST_ZYGOTE && include("ext_zygote/runtests.jl")
    EINOPS_TEST_REACTANT && include("ext_reactant/runtests.jl")

end
