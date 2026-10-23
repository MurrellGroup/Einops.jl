include("utils.jl")

const ArrowPatternSideNestedTuple = Tuple{Vararg{Union{Symbol,Int,EllipsisNotation.Ellipsis}}}
const ArrowPatternSide = Tuple{Vararg{Union{Symbol,Int,EllipsisNotation.Ellipsis,ArrowPatternSideNestedTuple}}}

function is_valid_side(x)
    x isa Tuple || return false
    for el in x
        if el isa Tuple
            # allow arbitrarily nested tuples by recursive check
            is_valid_side(el) || return false
        elseif !(el isa Symbol || el isa Int || el isa EllipsisNotation.Ellipsis)
            return false
        end
    end
    return true
end

check_side(x) = is_valid_side(x) || throw(ArgumentError("Invalid pattern: $x. Tuple elements must be Symbols, Ints, Ellipsis, or nested tuples thereof."))

"""
    ArrowPattern{L,R}

The left and right tuples of a pattern, stored as type parameters so the pattern
is known at compile time. For einsum with one array, the left tuple describes its
axes; with multiple arrays, it contains one axis-pattern tuple per array.

An instance `ArrowPattern{L,R}()` gets shown as `L --> R`, as [`-->`](@ref) is used for construction.
"""
struct ArrowPattern{L,R}
    function ArrowPattern{L,R}() where {L,R}
        check_side(L)
        check_side(R)
        new{L,R}()
    end
end

Base.show(io::IO, ::ArrowPattern{L,R}) where {L,R} = print(io, "$L --> $R")
Base.iterate(::ArrowPattern{L,R}) where {L,R} = (L, Val(:L))
Base.iterate(::ArrowPattern{L,R}, ::Val{:L}) where {L,R} = (R, Val(:R))
Base.iterate(::ArrowPattern{L,R}, ::Val{:R}) where {L,R} = nothing

# `q, k = rearrange.((q, k), ((:d, :h) ..) --> (:d, :h, ..))`
Base.broadcastable(p::ArrowPattern) = Ref(p)

"""
    -->

Create an [`ArrowPattern`](@ref) from a left and right tuple.
Non-tuple elements are automatically wrapped in a single-element tuple.
For einsum with multiple arrays, the left tuple contains one pattern per array.
With one array, the entire left tuple describes its axes: nested tuples group axes,
and there is no additional one-element operand-list wrapper.

# Examples

```jldoctest
julia> pattern1 = (:a, :b, :c) --> (:c, (:b, :a)) # nested tuple
(:a, :b, :c) --> (:c, (:b, :a))

julia> typeof(pattern1)
Einops.ArrowPattern{(:a, :b, :c), (:c, (:b, :a))}

julia> pattern2 = :a --> (1, :a) # single-element autoconversion
(:a,) --> (1, :a)

julia> typeof(pattern2)
Einops.ArrowPattern{(:a,), (1, :a)}

julia> (:a, ..) --> :a # exported ellipsis notation
(:a, EllipsisNotation.Ellipsis()) --> (:a,)
```

!!! note
    A pattern lives in its type, integers included, so an integer computed at run time
    gives a pattern whose type is only known at run time. Name the axis and pass its
    length as a keyword, or interpolate it into a string pattern (see [`@einops_str`](@ref)),
    which keeps the value out of the type.
"""
function --> end

(-->)(L::Tuple, R::Tuple) = ArrowPattern{L, R}()
(-->)(L::Tuple, R) = L --> (R,)
(-->)(L, R::Tuple) = (L,) --> R
(-->)(L, R) = (L,) --> (R,)


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
julia> einops"a 1 b c -> (c b) a"
(:a, 1, :b, :c) --> ((:c, :b), :a)

julia> einops"embed token (head batch) -> (embed head) token batch"
(:embed, :token, (:head, :batch)) --> ((:embed, :head), :token, :batch)

julia> einops"i j, j k -> i k" # for einsum
((:i, :j), (:j, :k)) --> (:i, :k)

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
((3, :b),) --> (:b,)
```
"""
struct BoundPattern{P<:ArrowPattern,C<:NamedTuple}
    pattern::P
    lengths::C
end

# Shown with the values where they were interpolated, which is what the pattern means.
function Base.show(io::IO, p::BoundPattern)
    L, R = p.pattern
    substitute(side) = map(el -> el isa Tuple ? substitute(el) : el isa Symbol ? get(p.lengths, el, el) : el, side)
    print(io, substitute(L), " --> ", substitute(R))
end

Base.broadcastable(p::BoundPattern) = Ref(p)
