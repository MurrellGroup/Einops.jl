@generated function findtype(::Type{T}, xs::Tuple) where T
    return Expr(:tuple, (i for (i, el_type) in enumerate(xs.parameters) if el_type <: T)...)
end

const Ignored = typeof(-)
const ParseShapePattern{N} = NTuple{N,Union{Symbol,Ignored}}

"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions using `Symbol`s,
and `-` to ignore dimensions.

# Examples

```jldoctest
julia> parse_shape(rand(2,3,4), (:a, :b, -))
(a = 2, b = 3)

julia> parse_shape(rand(2,3), (-, -))
NamedTuple()

julia> parse_shape(rand(2,3,4,5), (:first, :second, :third, :fourth))
(first = 2, second = 3, third = 4, fourth = 5)
```
"""
function parse_shape(x::AbstractArray{<:Any,N}, pattern::ParseShapePattern{N}) where N
    names = extract(Symbol, pattern)
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    inds = findtype(Symbol, pattern)
    return NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
end
