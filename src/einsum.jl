function get_size_dict(arrays, indices)
    size_named_tuple = merge(parse_shape.(arrays, Val.(indices))...)
    # Preserve the label type even when all operands are zero-dimensional.
    return Dict{Symbol,Int}(zip(keys(size_named_tuple), values(size_named_tuple)))
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

# One operand with unique labels and nothing new on the right is a reduction, which
# Einops' own `reduce` lowers to a reshape, a sum and a permute with static shapes.
# OMEinsum keeps the rest (traces and diagonals).
function _einsum(pattern::ArrowPattern{L,R}, x::AbstractArray; optimizer=nothing, context...) where {L,R}
    left, right = extract(Symbol, L), extract(Symbol, R)
    if allunique(left) && issubset(right, left)
        return reduce(sum, x, pattern; context...)
    end
    return _contract_einsum(pattern, x; optimizer, context...)
end

function _einsum(pattern::ArrowPattern, arrays::Vararg{AbstractArray}; optimizer=nothing, context...)
    return _contract_einsum(pattern, arrays...; optimizer, context...)
end

function _contract_einsum(
    ::ArrowPattern{L,R}, arrays::Vararg{AbstractArray};
    optimizer::Union{Nothing,OMEinsum.CodeOptimizer} = nothing,
    context...
) where {L,R}
    # Arity distinguishes one array's grouped axes from multiple operand patterns.
    operands = length(arrays) == 1 ? (L,) : L
    operands isa Tuple{Vararg{Tuple}} || throw(ArgumentError("Multiple arrays require one axis-pattern tuple per operand"))
    length(operands) == length(arrays) || throw(ArgumentError("Expected $(length(operands)) operands, got $(length(arrays))"))
    # Replace ellipses in operand patterns using per-array ranks
    L′, R′ = @ignore_derivatives replace_ellipses_einsum(operands --> R, Val(ndims.(arrays)))

    # Infer dimension sizes from patterns and merge with user context
    inferred = merge(parse_shape.(arrays, Val.(operands))...)
    merged_context = merge(inferred, NamedTuple(context))

    # Expand arrays according to possibly nested left indices
    arrays = expand.(arrays, Val.(operands); merged_context...)

    # Hand the flat symbol labels to the backend
    L_ome, R_ome = omeinsum_indices(L′, R′)
    output = isnothing(optimizer) ? contract(L_ome, R_ome, arrays...) : contract(L_ome, R_ome, arrays...; optimizer)
    return collapse(output, Val(R); merged_context...)
end

"""
    einsum(arrays..., pattern; optimizer=OMEinsum.GreedyMethod())

Compute the einsum operation specified by the pattern.

This function supports rearrange-style grouped axes, and can in some cases
automatically resolve size ambiguities.
Note that upstream Python `einops.einsum` does not yet do this (as of 0.8.1).

Commas in string patterns separate operands, including empty patterns for scalar
arrays. In the tuple API, one array uses the entire left tuple as its axis pattern;
multiple arrays use one axis-pattern tuple per array. There is no one-element
operand-list wrapper: `((:a, :b),)` means a grouped axis for a single array.

!!! note
    This function is not type stable.

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

# === Backend seam ===

"""
    contract(ixs, iout, xs...; optimizer=OMEinsum.GreedyMethod())

Backend entry point of [`einsum`](@ref): `xs[k]` carries the flat symbol labels `ixs[k]`,
and the result carries the labels `iout`. Ellipses and grouped axes are already resolved
by the time this is called. The default method contracts through OMEinsum; array types
can add methods to take over, as the Reactant extension does for two operands.

# Examples

```jldoctest
julia> x, y = rand(2, 3), rand(3, 4);

julia> Einops.contract(((:i, :j), (:j, :k)), (:i, :k), x, y) ≈ x * y
true
```
"""
function contract(
    ixs::Tuple, iout::Tuple, xs::AbstractArray...;
    optimizer::OMEinsum.CodeOptimizer = OMEinsum.GreedyMethod()
)
    optimized_code = @ignore_derivatives begin
        code = OMEinsum.StaticEinCode{Symbol,ixs,iout}()
        OMEinsum.optimize_code(code, get_size_dict(xs, ixs), optimizer)
    end
    return optimized_code(xs...)
end
