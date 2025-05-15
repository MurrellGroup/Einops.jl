using OMEinsum: StaticEinCode

nested(x::Tuple) = (x,)
nested(x::Tuple{Vararg{Tuple}}) = x

"""
    einsum(args::Vararg{Union{AbstractArray,ArrowPattern{L,R}}}) where {L,R}

Compute the einsum operation specified by the pattern.

# Examples

```jldoctest
julia> a = rand(2,3,4);

julia> b = rand(2,4,5);

julia> einsum(a, b, )
```
"""
function einsum(args::Vararg{Union{AbstractArray,ArrowPattern{L,R}}}) where {L,R}
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    return StaticEinCode{Symbol,nested(L),R}()(arrays...)
end
