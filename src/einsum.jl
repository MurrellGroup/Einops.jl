using OMEinsum: StaticEinCode

nested(x::Tuple) = (x,)
nested(x::Tuple{Vararg{Tuple}}) = x

"""
    einsum(arrays..., (left --> right))

Compute the einsum operation specified by the pattern.

# Examples

```jldoctest
julia> x, y = rand(2,3), rand(3,4);

julia> einsum(x, y, ((:i, :j), (:j, :k)) --> (:i, :k)) == x * y
true
```
"""
function einsum(args::Vararg{Union{AbstractArray,ArrowPattern{L,R}}}) where {L,R}
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    return StaticEinCode{Symbol,nested(L),R}()(arrays...)
end
