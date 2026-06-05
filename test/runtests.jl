using Einops
using Test

using Pkg

const EINOPS_TEST_ZYGOTE = v"1.11" <= VERSION <= v"1.12" && get(ENV, "EINOPS_TEST_ZYGOTE", "false") == "true"
EINOPS_TEST_ZYGOTE && Pkg.add("Zygote")

@testset "Einops.jl" begin

    include("patterns.jl")
    include("parse_shape.jl")
    include("reshape.jl")
    include("rearrange.jl")
    include("reduce.jl")
    include("repeat.jl")
    include("einsum.jl")
    include("pack_unpack.jl")
    include("macros.jl")
    include("builders.jl")
    include("robustness.jl")

    EINOPS_TEST_ZYGOTE && include("ext_zygote/runtests.jl")

end
