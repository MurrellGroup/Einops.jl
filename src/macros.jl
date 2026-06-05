# Shorthand macros: write a pattern as a plain string instead of `einops"..."`. When
# the pattern can be sized at expansion time the macro inlines the whole plan at the
# call site (a `let`, no `@generated` call and no `Base.Pairs` — both fragile for
# consumers like cuTile); otherwise it falls back to the function call. The pattern may
# sit anywhere among the positional args, so the same helper serves every op.
#
#     @rearrange(x, "a b -> b a")   ==>   let x = x; x = Permute((2, 1))(x); x end

# Inlineable ops → (local var names for `*_body`, static builder, ellipsis builder).
# `einsum` is absent: it routes through OMEinsum, not a reshape/permute plan.
const INLINEABLE = Dict{Symbol,Tuple{Vector{Symbol},Function,Function}}(
    :rearrange => ([:x],     rearrange_body, rearrange_body_ellipsis),
    :reshape   => ([:x],     reshape_body,   reshape_body_ellipsis),
    :reduce    => ([:f, :x], reduce_body,    reduce_body_ellipsis),
    :repeat    => ([:x],     repeat_body,    repeat_body_ellipsis),
)

function einops_macro_call(f::Symbol, args)
    positional = Any[]        # raw (unescaped) positional args, excluding the pattern
    pattern = nothing         # parsed value of the string-literal pattern
    pattern_index = 0         # its slot among the positional args (source order)
    names = Symbol[]          # statically known keyword names
    kwentries = Any[]         # `:kw`/`:...` entries for building the context NamedTuple
    has_splat = false
    for a in args
        if Meta.isexpr(a, :parameters)
            for kw in a.args
                _collect_kw!(kw, names, kwentries) && (has_splat = true)
            end
        elseif Meta.isexpr(a, :(=))
            _collect_kw!(a, names, kwentries)
        elseif a isa AbstractString
            pattern_index == 0 || throw(ArgumentError("@$f expects a single string pattern argument"))
            pattern = parse_pattern(a)
            pattern_index = length(positional) + 1
        else
            push!(positional, a)
        end
    end
    pattern_index == 0 && throw(ArgumentError("@$f expects a string pattern argument"))

    inlined = try_inline(f, pattern, positional, names, kwentries, has_splat)
    inlined === nothing || return inlined

    # Fallback: reconstruct the original `f(positional...; kwargs...)` call, with the
    # pattern restored to the slot it occupied in the source.
    posargs = Any[esc(a) for a in positional]
    insert!(posargs, pattern_index, pattern)
    call = Expr(:call, GlobalRef(@__MODULE__, f))
    isempty(kwentries) || push!(call.args, Expr(:parameters, kwentries...))
    append!(call.args, posargs)
    return call
end

# Record a keyword entry: its statically-known name (if any) and the `:kw`/`:...` entry
# for the context NamedTuple. Returns `true` for a `; kws...` splat (keys unknowable).
function _collect_kw!(kw, names, kwentries)
    if kw isa Symbol                                   # `; k`  ==>  `k = k`
        push!(names, kw); push!(kwentries, Expr(:kw, kw, esc(kw))); return false
    elseif Meta.isexpr(kw, :...)                       # `; kws...`
        push!(kwentries, Expr(:..., esc(kw.args[1]))); return true
    elseif Meta.isexpr(kw, :.)                         # `; obj.k`  ==>  `k = obj.k`
        push!(names, kw.args[2].value); push!(kwentries, Expr(:kw, kw.args[2].value, esc(kw))); return false
    else                                               # `k = v`
        push!(names, kw.args[1]); push!(kwentries, Expr(:kw, kw.args[1], esc(kw.args[2]))); return false
    end
end

# Returns an inlined expression, or `nothing` to signal "fall back to the call".
function try_inline(f, pattern, positional, names, kwentries, has_splat)
    haskey(INLINEABLE, f) || return nothing
    pattern isa ArrowPattern || return nothing
    vars, static_body, ellipsis_body = INLINEABLE[f]
    length(positional) == length(vars) || return nothing

    L, R = arrow_sides(pattern)
    has_ellipsis = (..) in flatten(L) || (..) in flatten(R)
    # A `; kws...` splat hides keys we'd need to resolve a multi-symbol left group's
    # inferred axis; without such a group the splat only supplies values, which stay
    # available through the runtime NamedTuple.
    has_splat && has_multisymbol_group(L) && return nothing

    plan = has_ellipsis ? ellipsis_body(L, R, Tuple(names)) :
                          static_body(length(L), L, R, Tuple(names))
    context = Expr(:tuple, Expr(:parameters, kwentries...))
    bindings = Any[Expr(:(=), v, esc(p)) for (v, p) in zip(vars, positional)]
    push!(bindings, Expr(:(=), :context, context))
    body = Expr(:block, rank_check(L, has_ellipsis), plan)
    return Expr(:let, Expr(:block, bindings...), body)
end

arrow_sides(::ArrowPattern{L,R}) where {L,R} = (L, R)

# Inlining builds the plan with `length(L)`, so `get_shape_in`'s `ndims(x)` rank check
# collapses to a tautology and is lost. Re-emit it: an exact rank without an ellipsis,
# a lower bound (the ellipsis must span ≥ 0 dims) with one. The non-ellipsis message
# matches `get_shape_in`'s.
function rank_check(L, has_ellipsis)
    n = length(L) - has_ellipsis
    has_ellipsis ?
        :(ndims(x) >= $n || throw(ArgumentError(string("Input rank ", ndims(x), " is too small for pattern requiring at least ", $n, " dimensions")))) :
        :(ndims(x) == $n || throw(ArgumentError(string("Input length ", $n, " does not match array dimensionality ", ndims(x)))))
end

has_multisymbol_group(side) = any(el -> el isa Tuple && length(el) >= 2, side)

"""
    @rearrange(x, "pattern", context...)

Shorthand for [`rearrange`](@ref) that takes the pattern as a string literal
instead of `einops"..."`. The reshape/permute plan is inlined at the call site
(no `@generated` function, no `Base.Pairs` keyword object) whenever the pattern
can be sized at macro-expansion time, otherwise it falls back to a `rearrange` call.

# Examples

```jldoctest
julia> using Einops: @rearrange

julia> x = rand(2, 3, 5);

julia> @rearrange(x, "a b c -> c b a") == rearrange(x, einops"a b c -> c b a")
true
```
"""
macro rearrange(args...)
    einops_macro_call(:rearrange, args)
end

"""
    @reshape(x, "pattern", context...)

Shorthand for [`reshape`](@ref) that takes the pattern as a string literal
instead of `einops"..."`. See [`@rearrange`](@ref) for inlining behaviour.

# Examples

```jldoctest
julia> using Einops: @reshape

julia> x = rand(2, 3);

julia> @reshape(x, "a b -> (a b)") == reshape(x, einops"a b -> (a b)")
true
```
"""
macro reshape(args...)
    einops_macro_call(:reshape, args)
end

"""
    @reduce(f, x, "pattern", context...)

Shorthand for [`reduce`](@ref) that takes the pattern as a string literal
instead of `einops"..."`. See [`@rearrange`](@ref) for inlining behaviour.

# Examples

```jldoctest
julia> using Einops: @reduce

julia> x = rand(2, 3);

julia> @reduce(sum, x, "a b -> a") == reduce(sum, x, einops"a b -> a")
true
```
"""
macro reduce(args...)
    einops_macro_call(:reduce, args)
end

"""
    @repeat(x, "pattern", context...)

Shorthand for [`repeat`](@ref) that takes the pattern as a string literal
instead of `einops"..."`. See [`@rearrange`](@ref) for inlining behaviour.

# Examples

```jldoctest
julia> using Einops: @repeat

julia> x = rand(2, 3);

julia> @repeat(x, "a b -> a b c", c=4) == repeat(x, einops"a b -> a b c", c=4)
true
```
"""
macro repeat(args...)
    einops_macro_call(:repeat, args)
end

"""
    @einsum(arrays..., "pattern", context...)

Shorthand for [`einsum`](@ref) that takes the pattern as a string literal
instead of `einops"..."`. Always expands to an `einsum` call (it is not type
stable and routes through OMEinsum rather than a reshape/permute plan).

# Examples

```jldoctest
julia> using Einops: @einsum

julia> x, y = rand(2, 3), rand(3, 4);

julia> @einsum(x, y, "i j, j k -> i k") == einsum(x, y, einops"i j, j k -> i k")
true
```
"""
macro einsum(args...)
    einops_macro_call(:einsum, args)
end
