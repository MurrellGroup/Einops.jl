# Shorthand macros that let a pattern be written as a plain string literal
# instead of `einops"..."`. Each `@f(args...)` expands to `f(args...)` with the
# single string-literal argument replaced by its parsed pattern, so e.g.
#
#     @rearrange(x, "a b -> b a", k=1)   ==>   rearrange(x, (:a, :b) --> (:b, :a); k=1)
#
# The string can sit anywhere among the positional arguments, which is why the
# same helper works for `einsum` (pattern last) as for `rearrange` (pattern
# second) or `reduce` (pattern third).

function einops_macro_call(f::Symbol, args)
    positional = Any[]
    kwargs = Any[]
    found = false
    for a in args
        if Meta.isexpr(a, :parameters)
            # keyword arguments passed after `;`
            for kw in a.args
                push!(kwargs, _to_kw(kw))
            end
        elseif Meta.isexpr(a, :(=))
            push!(kwargs, _to_kw(a))
        elseif a isa AbstractString
            found && throw(ArgumentError("@$f expects a single string pattern argument"))
            found = true
            push!(positional, parse_pattern(a))
        else
            push!(positional, esc(a))
        end
    end
    found || throw(ArgumentError("@$f expects a string pattern argument"))
    call = Expr(:call, GlobalRef(@__MODULE__, f))
    isempty(kwargs) || push!(call.args, Expr(:parameters, kwargs...))
    append!(call.args, positional)
    return call
end

# Normalize a single keyword entry for forwarding into the generated call's
# `:parameters` block. The parser only ever emits one of three forms here:
#   the `; k` shorthand (a bare `Symbol`), a `; kws...` splat (`:...`), or an
#   explicit `k = v` (`:kw` from `; k=v`, or `:(=)` from a trailing `, k=v`).
# Only values are escaped; key names are not.
function _to_kw(kw)
    kw isa Symbol && return Expr(:kw, kw, esc(kw))          # `; k`  ==>  `k = k`
    Meta.isexpr(kw, :...) && return Expr(:..., esc(kw.args[1]))  # `; kws...`
    return Expr(:kw, kw.args[1], esc(kw.args[2]))           # `k = v`
end

"""
    @rearrange(x, "pattern", context...)

Shorthand for [`rearrange`](@ref) that takes the pattern as a string literal
instead of `einops"..."`.

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
instead of `einops"..."`.

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
instead of `einops"..."`.

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
instead of `einops"..."`.

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
instead of `einops"..."`.

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
