nested(x::Tuple) = (x,)
nested(x::Tuple{Vararg{Tuple}}) = x
nested_val(::Val{x}) where x = Val.(nested(x))

function get_size_dict(arrays, indices)
    size_named_tuple = merge(parse_shape.(arrays, Val.(indices))...)
    return Dict(zip(keys(size_named_tuple), values(size_named_tuple)))
end

function omeinsum_indices(L′, R′)
    L_flat = Tuple(map(flatten, L′))
    R_flat = flatten(R′)
    L_ome = Tuple(begin
        any(x -> x isa Int && x != 1, li) && throw(ArgumentError("Only singleton integer dimensions (1) are allowed in left indices: $li"))
        Tuple(e for e in li if e isa Symbol)
    end for li in L_flat)
    any(x -> x isa Int && x != 1, R_flat) && throw(ArgumentError("Only singleton integer dimensions (1) are allowed in right indices: $R_flat"))
    R_ome = Tuple(e for e in R_flat if e isa Symbol)
    return L_ome, R_ome
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
function einsum(
    args::Vararg{Union{AbstractArray,ArrowPattern{L,R}}};
    optimizer::OMEinsum.CodeOptimizer = OMEinsum.GreedyMethod(),
    context...
) where {L,R}
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    last(args)::ArrowPattern
    # Replace ellipses in nested patterns using per-array ranks
    L′, R′ = @ignore_derivatives replace_ellipses_einsum(nested(L) --> R, Val(ndims.(arrays)))
    # Expand arrays according to possibly nested left indices
    arrays = expand.(arrays, nested_val(Val(L)); context...)
    # Prepare OMEinsum indices (flat symbols only)
    L_ome, R_ome = omeinsum_indices(L′, R′)
    optimized_code = @ignore_derivatives begin
        code = OMEinsum.StaticEinCode{Symbol,L_ome,R_ome}()
        OMEinsum.optimize_code(code, get_size_dict(arrays, L_ome), optimizer)
    end
    output = optimized_code(arrays...)
    return collapse(output, Val(R); context...)
end
