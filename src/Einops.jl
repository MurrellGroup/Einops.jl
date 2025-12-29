module Einops

import Base: reshape, reduce, repeat

using EllipsisNotation; export ..
using TupleTools: flatten, insertat
using ChainRulesCore: @ignore_derivatives
using OMEinsum: OMEinsum
using Rewrap
using Rewrap: Permute, Reduce, Repeat

include("utils.jl")

include("patterns/patterns.jl")
export ArrowPattern, -->
export @einops_str

include("parse_shape.jl")
export parse_shape

include("rearrange.jl")
export rearrange

include("reshape.jl")
export reshape

include("reduce.jl")
export reduce

include("repeat.jl")
export repeat

include("einsum.jl")
export einsum

include("pack_unpack.jl")
export pack, unpack

end
