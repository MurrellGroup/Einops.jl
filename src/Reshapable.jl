module Reshapable

using Base: ReshapedArray

export reshapable, hardreshape

reshapable(x::DenseArray) = x
reshapable(x::ReshapedArray) = x # XXX: should `hardreshape` simply never return ReshapedArray? 
reshapable(x::AbstractArray) = copy(x) # XXX: copy doesn't necessarily return a DenseArray...

hardreshape(x, args...) = reshape(reshapable(x), args...)

"""
    reshapable(x::AbstractArray)

Return a representation of `x` that can be reshaped without stacking an extra
`Base.ReshapedArray` layer. This may drop wrappers and may allocate. Similar
in spirit to `copy(x)`.

Array types that do not implement `reshape` should implement `reshapable`
if they can be reshaped without copying.

# Examples

```julia
julia> struct MyArray{T,N} <: AbstractArray{T,N}
           parent::Array{T,N}
       end

julia> Base.size(x::MyArray) = size(x.parent)

julia> Base.getindex(x::MyArray, i...) = getindex(x.parent, i...)

julia> x = MyArray(ones(1, 2))
1×2 MyArray{Float64, 2}:
 1.0  1.0

julia> reshape(x, :)
2-element reshape(::MyArray{Float64, 2}, 2) with eltype Float64:
 1.0
 1.0

julia> hardreshape(x, :)
2-element Vector{Float64}:
 1.0
 1.0

julia> pointer(ans) == pointer(x.parent)
true

julia> Reshapable.reshapable(x::MyArray) = x.parent

julia> hardreshape(x, :)
2-element Vector{Float64}:
 1.0
 1.0

julia> pointer(ans) == pointer(x.parent)
true
```
"""
reshapable

"""
    hardreshape(x, args...)

Like `reshape`, but can copy the data to avoid extra layer of wrapping.

Equal to `reshape(reshapable(x), args...)`.
For customizing behavior, see [`reshapable`](@ref).
"""
hardreshape

end
