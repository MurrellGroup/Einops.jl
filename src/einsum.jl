nested(x::Tuple) = (x,)
nested(x::Tuple{Vararg{Tuple}}) = x

function get_size_dict(arrays, indices)
    size_named_tuple = merge(parse_shape.(arrays, Val.(indices))...)
    return Dict(zip(keys(size_named_tuple), values(size_named_tuple)))
end

function _einsum(
    ::ArrowPattern{L,R}, arrays::Vararg{AbstractArray};
    optimizer::OMEinsum.CodeOptimizer = OMEinsum.GreedyMethod()
) where {L,R}
    optimized_code = @ignore_derivatives begin
        L′, R′ = replace_ellipses_einsum(nested(L) --> R, Val(ndims.(arrays)))
        code = OMEinsum.StaticEinCode{Symbol,L′,R′}()
        optimized_code = OMEinsum.optimize_code(code, get_size_dict(arrays, L′), optimizer)
    end
    return optimized_code(arrays...)
end

"""
    einsum(arrays..., pattern; optimizer=OMEinsum.GreedyMethod())

Compute the einsum operation specified by the pattern.

# Examples

```jldoctest
julia> x, y = rand(2,3), rand(3,4);

julia> einsum(x, y, ((:i, :j), (:j, :k)) --> (:i, :k)) == x * y
true
```
"""
function einsum(args::Vararg{Union{AbstractArray,ArrowPattern}}; kws...)
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    pattern = last(args)::ArrowPattern
    return _einsum(pattern, arrays...; kws...)
end
