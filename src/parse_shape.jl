@generated function findtype(::Type{T}, xs::Tuple) where T
    return Expr(:tuple, (i for (i, el_type) in enumerate(xs.parameters) if el_type <: T)...)
end

const Ignored = typeof(-)
const ParseShapePattern{N} = NTuple{N,Union{Symbol,Ignored}}
const ParseShapePatternEllipsis{N} = NTuple{N,Union{Symbol,Ignored,EllipsisNotation.Ellipsis}}

@generated function replace_ellipses_parse_shape(::Val{pattern}, ::AbstractArray{<:Any,N}) where {pattern, N}
    (..) in pattern || return :($pattern)
    ellipsis_index = findfirst(==(..), pattern)
    new_pattern = insertat(pattern, ellipsis_index, ntuple(i -> (-), N - length(pattern) + 1))
    return :($new_pattern)
end

"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions using `Symbol`s,
and `-` to ignore dimensions, and `...` to ignore any number of dimensions.

!!! note
    For proper type inference, the pattern needs to be passed as `Val(pattern)`
    when an ellipsis is present. This is done automatically when using [`@einops_str`](@ref).

# Examples

```jldoctest
julia> parse_shape(rand(2,3,4), (:a, :b, -))
(a = 2, b = 3)

julia> parse_shape(rand(2,3), (-, -))
NamedTuple()

julia> parse_shape(rand(2,3,4,5), einops"first second third fourth")
(first = 2, second = 3, third = 4, fourth = 5)

julia> parse_shape(rand(2,3,4), Val((:a, :b, ..)))
(a = 2, b = 3)
```
"""
function parse_shape(x::AbstractArray{<:Any,N}, pattern::ParseShapePattern{N}) where N
    names = extract(Symbol, pattern)
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    inds = findtype(Symbol, pattern)
    shape_info = @ignore_derivatives NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
    return shape_info
end

function parse_shape(x::AbstractArray, ::Val{pattern_ellipsis}) where pattern_ellipsis
    pattern = @ignore_derivatives replace_ellipses_parse_shape(Val(pattern_ellipsis), x)
    names = extract(Symbol, pattern)
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    inds = findtype(Symbol, pattern)
    shape_info = @ignore_derivatives NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
    return shape_info
end

parse_shape(x::AbstractArray, pattern::ParseShapePatternEllipsis) = parse_shape(x, Val(pattern))
