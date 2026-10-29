include("utils.jl")

# The `...` of a pattern, standing for any number of axes.
struct Ellipsis end

# The operands of an einsum pattern, which commas separate: the left side of
# `einops"i j, j k -> i k"` is `List{((:i, :j), (:j, :k))}()`. A pattern without commas has
# one operand, whose tuple is the left side itself, so a list always holds at least two.
# It iterates and indexes like the tuple it holds.
struct List{items}
    function List{items}() where {items}
        items isa Tuple && length(items) >= 2 && all(is_valid_side, items) ||
            throw(ArgumentError("A list of operands holds two or more pattern sides, not $items"))
        new{items}()
    end
end

Base.Tuple(::List{items}) where {items} = items
Base.length(::List{items}) where {items} = length(items)
Base.eltype(::Type{<:List}) = Tuple
Base.getindex(::List{items}, i) where {items} = items[i]
Base.iterate(l::List, i=1) = i > length(l) ? nothing : (l[i], i + 1)
Base.map(f, l::List) = map(f, Tuple(l))

throw_operand_list(L::List) = throw(ArgumentError(
    "The pattern lists $(length(L)) operands, but only `einsum` takes more than one"))

function is_valid_side(x)
    x isa Tuple || return false
    for el in x
        if el isa Tuple
            # allow arbitrarily nested tuples by recursive check
            is_valid_side(el) || return false
        elseif !(el isa Symbol || el isa Int || el isa Ellipsis)
            return false
        end
    end
    return true
end

check_side(x) = is_valid_side(x) || throw(ArgumentError("Invalid pattern: $x. Tuple elements must be Symbols, Ints, Ellipsis, or nested tuples thereof."))

"""
    ArrowPattern{L,R}

The left and right sides of a pattern, made by [`@einops_str`](@ref) and stored as type
parameters, so the pattern is known at compile time. A side is a tuple: a name is a
`Symbol`, a group a nested tuple, and `...` an `Einops.Ellipsis()`. When commas separate
the operands of an einsum, the left side is an `Einops.List` of one such tuple per operand.

A pattern shows as the `einops"..."` string that makes it.

# Examples

```jldoctest
julia> p = einops"a b c -> c (b a)"
einops"a b c -> c (b a)"

julia> typeof(p)
Einops.ArrowPattern{(:a, :b, :c), (:c, (:b, :a))}
```

!!! note
    A pattern lives in its type, integers included. To give an axis a length computed at
    run time, name the axis and pass its length as a keyword, or interpolate the length
    with `\$` (see [`@einops_str`](@ref)), which keeps the value out of the type.
"""
struct ArrowPattern{L,R}
    function ArrowPattern{L,R}() where {L,R}
        L isa List || check_side(L)
        check_side(R)
        new{L,R}()
    end
end

Base.iterate(::ArrowPattern{L,R}) where {L,R} = (L, Val(:L))
Base.iterate(::ArrowPattern{L,R}, ::Val{:L}) where {L,R} = (R, Val(:R))
Base.iterate(::ArrowPattern{L,R}, ::Val{:R}) where {L,R} = nothing

# `q, k = rearrange.((q, k), einops"(d h) ... -> d h ..."; h)`
Base.broadcastable(p::ArrowPattern) = Ref(p)

Base.show(io::IO, ::ArrowPattern{L,R}) where {L,R} = print(io, pattern_string(L, R))

# The `einops"..."` string that makes a pattern with these sides, with each name spelled as
# `f` turns it.
function pattern_string(L, R, f=identity)
    left = L isa List ? rstrip(join(map(op -> spell_side(op, f), L), ", ")) : spell_side(L, f)
    return string("einops\"", left, isempty(left) ? "" : " ", "->", isempty(R) ? "" : " ", spell_side(R, f), "\"")
end

spell_side(side, f) = join(map(dim -> spell_axis(dim, f), side), " ")
spell_axis(dim, f) = dim isa Tuple ? string("(", spell_side(dim, f), ")") : dim isa Ellipsis ? "..." : string(dim isa Symbol ? f(dim) : dim)

include("ellipses.jl")
include("parse.jl")

"""
    @einops_str -> Union{ArrowPattern,BoundPattern,Val}

For parity with Python implementation.

In a pattern, `1` is an axis of length one, and inside parentheses it adds nothing:
`(1 b)` is `(b)`. Any other integer is an axis given only by its length, which can be
reduced (on the left of [`reduce`](@ref)) or created (on the right of [`repeat`](@ref)).
Integers and names given a length by a keyword have that length checked against the input.

`\$x` interpolates an axis of length `x`, as if the value were written there: it too can
only be reduced or created, and each occurrence is an axis of its own. The value is not
part of the pattern's type, so it may be computed at run time without making the pattern's
type unknown; the pattern then carries it along as a [`BoundPattern`](@ref).

# Examples

```jldoctest
julia> einops"embed token (head batch) -> (embed head) token batch"
einops"embed token (head batch) -> (embed head) token batch"

julia> typeof(einops"a 1 b ... -> (b a) ...")
Einops.ArrowPattern{(:a, 1, :b, Einops.Ellipsis()), ((:b, :a), Einops.Ellipsis())}

julia> einops"i j, j k -> i k" # for einsum
einops"i j, j k -> i k"

julia> einops"a b _ d" # for parse_shape
Val{(:a, :b, -, :d)}()

julia> einops"i j * k" # for pack/unpack
Val{(:i, :j, *, :k)}()

julia> k = 2;

julia> reduce(sum, rand(4, 6), einops"(h \$k) (w \$k) -> h w") |> size
(2, 3)
```
"""
macro einops_str(pattern)
    occursin('$', pattern) || return parse_pattern(pattern)
    pattern, names, exprs = parse_interpolated_pattern(interpolation_parts(pattern))
    return :(BoundPattern($pattern, NamedTuple{$(Tuple(names))}(($(map(esc, exprs)...),))))
end

"""
    BoundPattern

An [`ArrowPattern`](@ref) together with the lengths of some of its axes, which
[`@einops_str`](@ref) makes when values are interpolated into a pattern with `\$`.
The ops pass the lengths on as keywords, so the values stay out of the type.

# Examples

```jldoctest
julia> n = 3;

julia> einops"(\$n b) -> b"
einops"(3 b) -> b"
```
"""
struct BoundPattern{P<:ArrowPattern,C<:NamedTuple}
    pattern::P
    lengths::C
end

# Shown with the values where they were interpolated, which is what the pattern means.
function Base.show(io::IO, p::BoundPattern)
    L, R = p.pattern
    print(io, pattern_string(L, R, name -> get(p.lengths, name, name)))
end

Base.broadcastable(p::BoundPattern) = Ref(p)
