using Einops
using Test

@testset "Einops.jl" begin
    include("patterns.jl")
    include("parse_shape.jl")
    include("rearrange.jl")
    include("reduce.jl")
    include("repeat.jl")
    include("einsum.jl")
    include("pack_unpack.jl")
    include("robustness.jl")
    include("test_zygote.jl")
end