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
    if !(L isa List)
        left, right = extract(Symbol, L), extract(Symbol, R)
        allunique(left) && issubset(right, left) && return reduce(sum, x, pattern; context...)
    end
    return _contract_einsum(pattern, x; optimizer, context...)
end

function _einsum(pattern::ArrowPattern, arrays::Vararg{AbstractArray}; optimizer=nothing, context...)
    return _contract_einsum(pattern, arrays...; optimizer, context...)
end

_einsum(::BoundPattern, ::Vararg{AbstractArray}; context...) = throw_interpolated_einsum()

# An axis given only by its length could only be summed out, which a named axis does too.
throw_interpolated_einsum() = throw(ArgumentError(
    "`einsum` takes no interpolated lengths: an axis given only by its length could only be summed out, " *
    "which a named axis does too. Name it, and pass its length as a keyword."))

function _contract_einsum(
    ::ArrowPattern{L,R}, arrays::Vararg{AbstractArray};
    optimizer::Union{Nothing,OMEinsum.CodeOptimizer} = nothing,
    context...
) where {L,R}
    # Commas make a list of operands; without them, the left side is the one operand.
    operands = L isa List ? Tuple(L) : (L,)
    length(operands) == length(arrays) || throw(ArgumentError(
        "The pattern has $(length(operands)) operand$(length(operands) == 1 ? "" : "s"), " *
        "but $(length(arrays)) array$(length(arrays) == 1 ? " was" : "s were") given"))
    # Replace ellipses in operand patterns using per-array ranks
    L′, R′ = @ignore_derivatives replace_ellipses_einsum(Val(operands), Val(R), Val(ndims.(arrays)))

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

Commas in patterns separate operands, including empty patterns for scalar arrays.

!!! note
    This function is not type stable.

# Examples

```jldoctest
julia> x, y = rand(2,3), rand(3,4);

julia> einsum(x, y, einops"i j, j k -> i k") == x * y
true
```
"""
function einsum(args::Vararg{Union{AbstractArray,ArrowPattern,BoundPattern}}; kws...)
    arrays::Tuple{Vararg{AbstractArray}} = Base.front(args)
    pattern = last(args)::Union{ArrowPattern,BoundPattern}
    return _einsum(pattern, arrays...; kws...)
end

# === Backend seam ===

"""
    contract(ixs, iout, xs...; optimizer=OMEinsum.GreedyMethod())

Backend entry point of [`einsum`](@ref): `xs[k]` carries the flat symbol labels `ixs[k]`,
and the result carries the labels `iout`. Ellipses and grouped axes are already resolved
by the time this is called. One or two operands contract through OMEinsum; more contract
pairwise, in the order `optimizer` finds, each pair through the two-operand method. So an
array type takes over every einsum by adding methods for one and two operands, as the
Reactant extension does for two.

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
    code = OMEinsum.StaticEinCode{Symbol,ixs,iout}()
    length(xs) <= 2 && return code(xs...)
    order = @ignore_derivatives contraction_order(OMEinsum.optimize_code(code, get_size_dict(xs, ixs), optimizer))
    return contract_tree(order, xs)
end

# The contraction tree as plain data: an operand's position at a leaf, and at a node
# the labels of its children, of its result, and the children. OMEinsum's own tree
# leaves fields undefined at its leaves, which a tracing compiler cannot look at.
function contraction_order(node)
    OMEinsum.isleaf(node) && return OMEinsum.tensorindex(node)
    eins = OMEinsum.rootcode(node)
    ixs = Tuple(Tuple(ix) for ix in OMEinsum.getixsv(eins))
    iout = Tuple(OMEinsum.getiyv(eins))
    # A loop, not `map`: a tracing compiler inspects what `map` is given.
    children = Any[]
    for child in OMEinsum.siblings(node)
        push!(children, contraction_order(child))
    end
    return (; ixs, iout, children = Tuple(children))
end

function contraction_order(tree::OMEinsum.SlicedEinsum)
    isempty(tree.slicing) || throw(ArgumentError("A sliced contraction order is not supported; choose an optimizer without slicing"))
    return contraction_order(tree.eins)
end

contract_tree(k::Int, xs) = xs[k]
contract_tree(node::NamedTuple, xs) =
    contract(node.ixs, node.iout, map(child -> contract_tree(child, xs), node.children)...)
