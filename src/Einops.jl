module Einops

using ChainRulesCore

# currently unsupported
using EllipsisNotation
export ..

include("utils.jl")

include("patterns.jl")
export ArrowPattern, -->
export @einops_str

include("parse_shape.jl")
export parse_shape

include("rearrange.jl")
export rearrange

include("reduce.jl")
export reduce

include("repeat.jl")
export repeat

include("pack_unpack.jl")
export pack, unpack

end
