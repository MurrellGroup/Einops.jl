nested(x::Tuple) = (x,)
nested(x::Tuple{Vararg{Tuple}}) = x

function get_size_dict(arrays, indices::Tuple{Vararg{Tuple{Vararg{Symbol}}}})
    size_named_tuple = merge(parse_shape.(arrays, indices)...)
    return Dict(zip(keys(size_named_tuple), values(size_named_tuple)))
end

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
function einsum(args::Vararg{Union{AbstractArray,ArrowPattern{L,R}}}; optimizer=TreeSA()) where {L,R}
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    last(args)::ArrowPattern
    optimized_code = @ignore_derivatives begin
        L′, R′ = replace_ellipses_einsum(nested(L) --> R, Val(ndims.(arrays)))
        code = StaticEinCode{Symbol,L′,R′}()
        optimized_code = optimize_code(code, get_size_dict(arrays, L′), optimizer)
    end
    return optimized_code(arrays...)
end
