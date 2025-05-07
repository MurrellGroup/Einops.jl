module Einops

# TODO: support ellipses
using EllipsisNotation
export ..

# TODO: use TransmuteDims.jl

struct Pattern{L,R} end
(-->)(left, right) = Pattern{left, right}()
Base.show(io::IO, ::Pattern{L,R}) where {L,R} = print(io, "$L --> $R")
Base.iterate(::Pattern{L}) where L = (L, Val(:right))
Base.iterate(::Pattern{<:Any,R}, ::Val{:right}) where R = (R, nothing)
Base.iterate(::Pattern, ::Nothing) = nothing

include("utils.jl")
export -->

include("einops_str.jl")
export @einops_str

include("rearrange.jl")
export rearrange

# TODO: implement reduce, repeat
Base.reduce(f, x::AbstractArray, pattern::Pattern; context...) = error("Not implemented")
Base.repeat(x, pattern::Pattern; context...) = error("Not implemented")

# TODO: einsum, pack, unpack

end
