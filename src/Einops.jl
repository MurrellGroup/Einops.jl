module Einops

# TODO: support ellipses
using EllipsisNotation
export ..

# TODO: use TransmuteDims.jl

include("utils.jl")
export -->

include("einops_str.jl")
export @einops_str

include("rearrange.jl")
export rearrange

include("pack_unpack.jl")
export pack, unpack

# TODO: implement reduce, repeat
Base.reduce(f, x::AbstractArray, pattern::Pattern; context...) = error("Not implemented")
Base.repeat(x::AbstractArray, pattern::Pattern; context...) = error("Not implemented")

# TODO: einsum

end
