module Einops

# TODO: support ellipses
using EllipsisNotation
export ..

# TODO: use TransmuteDims.jl

include("utils.jl")
export -->
export parse_shape

include("einops_str.jl")
export @einops_str

include("rearrange.jl")
export rearrange

include("pack_unpack.jl")
export pack, unpack

include("reduce.jl")
export reduce

include("repeat.jl")
export repeat

end
