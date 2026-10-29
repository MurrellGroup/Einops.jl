function get_size_dict(arrays, indices)
    size_named_tuple = merge(parse_shape.(arrays, Val.(indices))...)
    # Preserve the label type even when all operands are zero-dimensional.
    return Dict{Symbol,Int}(zip(keys(size_named_tuple), values(size_named_tuple)))
end

# `check_einsum` has ruled out integers other than 1, which are units and drop out here.
function omeinsum_indices(L′, R′)
    L_ome = Tuple(Tuple(e for e in flatten(li) if e isa Symbol) for li in L′)
    R_ome = Tuple(e for e in flatten(R′) if e isa Symbol)
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

_einsum(p::BoundPattern, ::Vararg{AbstractArray}; context...) = throw_unnamed_einsum(first(keys(p.lengths)))

# An axis given only by its length could only be summed out, which a named axis does too.
throw_unnamed_einsum(dim) = throw(ArgumentError(
    "Unnamed axis `$(dim isa Symbol ? display_name(dim) : dim)` isn't supported by `einsum`; " *
    "name it and pass its length as a keyword"))

# What only the whole pattern and every operand's rank can show, checked once, as the
# method is generated: `operands` holds one side per operand, and `Ns` their ranks.
@generated function check_einsum(::Val{operands}, ::Val{R}, ::Val{Ns}) where {operands,R,Ns}
    n = length(operands)
    check_ellipses(operands, R)
    for (k, side) in enumerate(operands)
        check_rank(side, Ns[k], input_name(k, n))
    end
    for side in (operands..., R), dim in flatten(side)
        dim isa Integer && dim != 1 && throw_unnamed_einsum(dim)
    end
    allunique(extract(Symbol, R)) || throw_repeated(extract(Symbol, R), "on the right side")
    only_right = setdiff(extract(Symbol, R), extract(Symbol, operands))
    isempty(only_right) || throw(ArgumentError("$(listing(only_right, "is", "are")) on the right side but in no operand"))
    spans = [(k, Ns[k] - length(side) + 1) for (k, side) in enumerate(operands) if Ellipsis() in side]
    for (k, span) in spans
        span == spans[1][2] || throw(DimensionMismatch(
            "`...` spans $(spans[1][2]) dimension$(spans[1][2] == 1 ? "" : "s") in operand $(spans[1][1]) but $span in operand $k"))
    end
    return nothing
end

# Operands that share a name must agree on its length.
function check_shared_lengths(shapes)
    for j in eachindex(shapes), i in 1:j-1, name in keys(shapes[i])
        haskey(shapes[j], name) && shapes[i][name] != shapes[j][name] && throw(DimensionMismatch(
            "`$name` has length $(shapes[i][name]) in operand $i but $(shapes[j][name]) in operand $j"))
    end
    return nothing
end

@non_differentiable check_shared_lengths(::Any)

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
    @ignore_derivatives check_einsum(Val(operands), Val(R), Val(ndims.(arrays)))
    # Replace ellipses in operand patterns using per-array ranks
    L′, R′ = @ignore_derivatives replace_ellipses_einsum(Val(operands), Val(R), Val(ndims.(arrays)))

    # Infer dimension sizes from patterns and merge with user context
    shapes = parse_shape.(arrays, Val.(operands))
    check_shared_lengths(shapes)
    merged_context = merge(merge(shapes...), NamedTuple(context))

    # Expand arrays according to possibly nested left indices
    inputs = ntuple(k -> Val(Symbol(input_name(k, length(operands)))), length(operands))
    arrays = expand.(arrays, Val.(operands), inputs; merged_context...)

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
