include("utils.jl")

const ArrowPatternSideNestedTuple = Tuple{Vararg{Union{Symbol,Int,EllipsisNotation.Ellipsis}}}
const ArrowPatternSide = Tuple{Vararg{Union{Symbol,Int,EllipsisNotation.Ellipsis,ArrowPatternSideNestedTuple}}}

check_side(x) = x isa ArrowPatternSide || throw(ArgumentError("Invalid pattern: $x. Expected instance of type $ArrowPatternSide"))

"""
    ArrowPattern{L,R}

A pair of tuples representing the left and right sides of a `rearrange`/`reduce`/`repeat` pattern.
These tuples are stored as type parameters, such that the pattern is known at compile time.

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


"""
    -->

Create an [`ArrowPattern`](@ref) from a left and right tuple.
Non-tuple elements are automatically wrapped in a single-element tuple.

# Examples

```jldoctest
julia> pattern1 = (:a, :b, :c) --> (:c, (:b, :a)) # nested tuple
(:a, :b, :c) --> (:c, (:b, :a))

julia> typeof(pattern1)
ArrowPattern{(:a, :b, :c), (:c, (:b, :a))}

julia> pattern2 = :a --> (1, :a) # single-element autoconversion
(:a,) --> (1, :a)

julia> typeof(pattern2)
ArrowPattern{(:a,), (1, :a)}

julia> (:a, ..) --> :a # exported ellipsis notation
(:a, EllipsisNotation.Ellipsis()) --> (:a,)
```
"""
function --> end

(-->)(L::Tuple, R::Tuple) = ArrowPattern{L, R}()
(-->)(L::Tuple, R) = L --> (R,)
(-->)(L, R::Tuple) = (L,) --> R
(-->)(L, R) = (L,) --> (R,)


include("ellipses.jl")
include("parse.jl")

"""
    @einops_str -> Union{ArrowPattern,Tuple}

For parity with Python implementation.

# Examples

```jldoctest
julia> einops"a 1 b c -> (c b) a"
(:a, 1, :b, :c) --> ((:c, :b), :a)

julia> einops"embed token (head batch) -> (embed head) token batch"
(:embed, :token, (:head, :batch)) --> ((:embed, :head), :token, :batch)

julia> einops"i j, j k -> i k" # for einsum
((:i, :j), (:j, :k)) --> (:i, :k)

julia> einops"a b _ d" # for parse_shape
(:a, :b, -, :d)

julia> einops"i j * k" # for pack/unpack
(:i, :j, *, :k)
```
"""
macro einops_str(pattern)
    return parse_pattern(pattern)
end
