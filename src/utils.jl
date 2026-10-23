using Rewrap: Keep, Merge, Split, Resqueeze, Squeeze, Unsqueeze, Repeat

# Every reshape the generated code emits goes through this hook, so an array type can
# take over without touching Rewrap: the Reactant extension materializes the lazy
# `ReshapedArray` that `Base.reshape` leaves around a traced array.
_reshape(x, shape) = Rewrap.reshape(x, shape)

pairs_type_to_names(::Type{<:Base.Pairs{Symbol,<:Any,<:Any,<:NamedTuple{names}}}) where names = names

function get_shape_in(N, left, context_names; allow_repeats=false)
    length(left) == N || throw(ArgumentError("Input length $(length(left)) does not match array dimensionality $N"))
    allunique(extract(Symbol, left)) || allow_repeats || throw(ArgumentError("Left names $(left) are not unique"))
    left isa Tuple{Vararg{Symbol}} && return nothing
    new_shape = :()
    ops = new_shape.args
    for input_dim in left
        input_dim isa Symbol && (push!(ops, :($Keep())); continue)
        names = axis_names(input_dim)
        if isempty(names)
            push!(ops, :($Squeeze()))
        elseif length(names) == 1
            push!(ops, :($Keep()))
        else
            known_size = filter(in(context_names), names)
            length(names) - length(known_size) <= 1 || throw(ArgumentError("Unknown dimension sizes: $(filter(∉(context_names), names))"))
            sizes_tuple = Expr(:tuple, (name in known_size ? :(getfield(context, $(QuoteNode(name)))) : :(:) for name in names)...)
            push!(ops, :($Split(1, $sizes_tuple)))
        end
    end
    return new_shape
end

# The names an axis of the pattern splits into or merges from. A `1` is a unit, of length
# one, so it adds no name: `(1 b)` is `(b)`, and `1`, `(1)` and `()` each have none.
function axis_names(dim)
    dim isa Symbol && return (dim,)
    dim == 1 && return ()
    dim isa Integer && throw_anonymous_axis("Integer $dim")
    dim isa Tuple || throw(ArgumentError("Invalid dimension in pattern: $dim"))
    for el in dim
        el isa Symbol || el == 1 || (el isa Integer ? throw_anonymous_axis("Integer $el") : throw(ArgumentError("Invalid dimension in pattern: $dim")))
    end
    return filter(el -> el isa Symbol, dim)
end

throw_anonymous_axis(what) = throw(ArgumentError(
    "$what in pattern: an axis given only by its length can be reduced (left side of `reduce`) " *
    "or created (right side of `repeat`), but not carried through. Name it, and pass its length as a keyword."))

# An axis given only by its length, an integer other than 1 or an interpolated `$x`, on a
# side where it would have to be carried through rather than reduced or created.
function reject_anonymous_axes(side)
    for dim in side
        if dim isa Tuple
            reject_anonymous_axes(dim)
        elseif dim isa Integer && dim != 1
            throw_anonymous_axis("Integer $dim")
        elseif dim isa Symbol && is_interpolated(dim)
            throw_anonymous_axis("`$(interpolated_source(dim))`")
        end
    end
end

# Each input axis whose length the pattern pins down is checked before anything reshapes it,
# so a wrong length is reported against the pattern instead of surfacing from a reshape, or
# not at all. A unit, an integer or a name with a known length fixes the whole axis; in a
# group with one name left to infer, the known part must divide it. `axis(i)` maps the
# position of a left entry to the input axis it covers.
function axis_length_checks(left, known; axis=identity)
    checks = Expr[]
    for (i, dim) in enumerate(left)
        names = axis_names(dim)
        given = filter(in(known), names)
        whole = length(given) == length(names)
        whole || !isempty(given) || continue
        lengths = [:(getfield(context, $(QuoteNode(name)))) for name in given]
        n = isempty(lengths) ? 1 : length(lengths) == 1 ? only(lengths) : Expr(:call, :*, lengths...)
        shown = Tuple(unique(filter(!is_hidden_name, given)))
        shown_lengths = Expr(:tuple, (:(getfield(context, $(QuoteNode(name)))) for name in shown)...)
        ax = axis(i)
        push!(checks, :(check_axis_length(size(x, $ax), $n, $ax, $whole, NamedTuple{$shown}($shown_lengths))))
    end
    return checks
end

# Names Einops makes up for itself (an integer's axis, an ellipsis' axes) stay out of messages.
is_hidden_name(name::Symbol) = startswith(String(name), "__")

@inline function check_axis_length(len, n, i, whole, given)
    (whole ? len == n : n != 0 && rem(len, n) == 0) || throw_axis_length(len, n, i, whole, given)
    return nothing
end

@noinline function throw_axis_length(len, n, i, whole, given)
    expected = whole ? string(n) : "a multiple of $n"
    names = isempty(given) ? "" : string(" (", join(("$k = $v" for (k, v) in pairs(given)), ", "), ")")
    throw(DimensionMismatch("Input axis $i has length $len, but the pattern expects $expected$names"))
end

@non_differentiable check_axis_length(::Any...)

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
    for dim in right
        if dim isa Symbol
            push!(sizes, :($Keep()))
        else
            names = axis_names(dim)
            push!(sizes, isempty(names) ? :($Unsqueeze()) : :($Merge($(length(names)))))
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
    map(el -> el == Ellipsis() ? ELLIPSIS_PLACEHOLDER :
              el isa Tuple ? replace_ellipsis_placeholder(el) : el, side)

# `m` is captured once at the top of the body, since `x` is reassigned as the plan runs
# and `ndims(x)` would otherwise drift. C = length(L) - 1 (every left entry but the slot).
const ELLIPSIS_M = Symbol("__ellipsis_m__")
ellipsis_m_binding(L) = :($ELLIPSIS_M = ndims(x) - $(length(L) - 1))

# The input axis a left entry covers, when the placeholder at `pli` stands for `m` axes.
ellipsis_axis(pli) = i -> i < pli ? i : :($(i - 1) + $ELLIPSIS_M)

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
        shape_out.args[gidx] = merge_span(length(axis_names(right[gidx])), m)
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
