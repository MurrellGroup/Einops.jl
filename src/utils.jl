using Rewrap: Keep, Merge, Split, Resqueeze, Squeeze, Unsqueeze, Repeat

pairs_type_to_names(::Type{<:Base.Pairs{Symbol,<:Any,<:Any,<:NamedTuple{names}}}) where names = names

function get_shape_in(N, left, context_names; allow_repeats=false)
    length(left) == N || throw(ArgumentError("Input length $(length(left)) does not match array dimensionality $N"))
    allunique(extract(Symbol, left)) || allow_repeats || throw(ArgumentError("Left names $(left) are not unique"))
    left isa Tuple{Vararg{Symbol}} && return nothing
    new_shape = :()
    ops = new_shape.args
    for (i, input_dim) in enumerate(left)
        if input_dim isa Int
            input_dim == 1 || throw(ArgumentError("Singleton dimension size is not 1: $input_dim"))
            push!(ops, :($Squeeze()))
        elseif input_dim isa Symbol
            push!(ops, :($Keep()))
        elseif input_dim isa Tuple{Symbol}
            push!(ops, :($Keep()))
        elseif input_dim isa Tuple{Vararg{Symbol}}
            known_size = filter(in(context_names), input_dim)
            length(input_dim) - length(known_size) <= 1 || throw(ArgumentError("Unknown dimension sizes: $(filter(∉(context_names), input_dim))"))
            sizes_tuple = Expr(:tuple, (name in known_size ? :(getfield(context, $(QuoteNode(name)))) : :(:) for name in input_dim)...)
            push!(ops, :($Split(1, $sizes_tuple)))
        else
            throw(ArgumentError("Invalid input dimension: $input_dim"))
        end
    end
    return new_shape
end

get_mapping(left, right) = Tuple(findfirst.(isequal.(right), (left,)))

function get_permutation(left, right)
    isempty(setdiff(left, right)) || throw(ArgumentError("Set of left names $(left) does not match set of right names $(right)"))
    return get_mapping(left, right)
end

function get_shape_out(right)
    allunique(extract(Symbol, right)) || throw(ArgumentError("Right names $(right) are not unique"))
    right isa Tuple{Vararg{Symbol}} && return nothing
    new_shape = :()
    sizes = new_shape.args
    i = 1
    for dim in right
        if dim isa Int
            dim == 1 || throw(ArgumentError("Singleton dimension size is not 1: $dim"))
            push!(sizes, :($Unsqueeze()))
        elseif dim isa Symbol
            push!(sizes, :($Keep()))
            i += 1
        elseif dim isa Tuple{Vararg{Symbol}}
            N = length(dim)
            push!(sizes, :($Merge($N)))
            i += N
        end
    end
    return new_shape
end

function get_dropdims_shape(N::Int, dims::Tuple{Vararg{Int}})
    new_shape = :()
    ops = new_shape.args
    for i in 1:N
        push!(ops, i in dims ? :($Squeeze()) : :($Keep()))
    end
    return new_shape
end

# === Ellipsis inlining for the shorthand macros ===
#
# With an ellipsis the rank N = ndims(x) is unknown at expansion, so the span it covers,
# `m = N - C`, rides in as the folded constant `ndims(x) - C`. The structure is still
# built at expansion: the ellipsis becomes one placeholder symbol, the ordinary builders
# run, then each N-dependent piece is rewritten in terms of `m` — a kept run folds to a
# single `Keep(m)`, a fixed+ellipsis merge to `Merge((k-1)+m)`, reduced indices past it
# shift by `m-1`, and the permutation expands its slot via `ntuple(_, Val(m))`.

const ELLIPSIS_PLACEHOLDER = Symbol("__ellipsis_placeholder__")

# Replace the ellipsis (`..`, possibly nested on the right) with the placeholder.
replace_ellipsis_placeholder(side) =
    map(el -> el == (..) ? ELLIPSIS_PLACEHOLDER :
              el isa Tuple ? replace_ellipsis_placeholder(el) : el, side)

# `m` is captured once at the top of the body, since `x` is reassigned as the plan runs
# and `ndims(x)` would otherwise drift. C = length(L) - 1 (every left entry but the slot).
const ELLIPSIS_M = Symbol("__ellipsis_m__")
ellipsis_m_binding(L) = :($ELLIPSIS_M = ndims(x) - $(length(L) - 1))

keep_run(m) = Expr(:call, Keep, m)                        # Keep(m)
merge_span(k, m) = Expr(:call, Merge, :($(k - 1) + $m))   # Merge((k-1) + m)

@inline function ellipsis_perm(::Val{skeleton}, ::Val{pli}, ::Val{N}) where {skeleton,pli,N}
    m = N - (length(skeleton) - 1)
    q = something(findfirst(==(pli), skeleton))   # placeholder's slot in the skeleton
    shift(e) = e < pli ? e : e - 1 + m            # e ≠ pli here ⇒ always an Int
    before = ntuple(i -> shift(skeleton[i]),     Val(q - 1))
    run    = ntuple(i -> pli + i - 1,            Val(m))
    after  = ntuple(i -> shift(skeleton[q + i]), Val(length(skeleton) - q))
    return (before..., run..., after...)
end

@inline _permute(x, perm::NTuple{N,Int}) where {N} = Permute{perm,N}()(x)

permute_run_expr(permutation, pli) =
    :(_permute(x, ellipsis_perm($(Val(permutation)), $(Val(pli)), Val(ndims(x)))))

# Swap the op at top-level entry `i` (the placeholder's `Keep()`) for `Keep(m)`.
expand_keep!(ops::Expr, i, m) = (ops.args[i] = keep_run(m); ops)

# Placeholder is either a top-level `Keep` (→ `Keep(m)`) or inside a merged group (→ bump).
function expand_shape_out!(shape_out::Expr, right, m)
    oidx = findfirst(==(ELLIPSIS_PLACEHOLDER), right)
    if oidx !== nothing
        shape_out.args[oidx] = keep_run(m)
    else
        gidx = findfirst(t -> t isa Tuple && ELLIPSIS_PLACEHOLDER in t, right)
        shape_out.args[gidx] = merge_span(length(right[gidx]), m)
    end
    return shape_out
end

# Index of the `n`-th `Keep` op (locates the placeholder keep among interleaved `Unsqueeze`s).
function nth_keep_index(ops::Expr, n)
    count = 0
    for (i, op) in enumerate(ops.args)
        if Meta.isexpr(op, :call) && op.args[1] === Keep
            count += 1
            count == n && return i
        end
    end
    error("placeholder keep not found")
end
