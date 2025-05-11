const ArrowPatternNestedTuple = Tuple{Vararg{Union{Symbol,Int}}}
const ArrowPatternSide = Tuple{Vararg{Union{Symbol,Int,EllipsisNotation.Ellipsis,ArrowPatternNestedTuple}}}

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


function parse_pattern(pattern::AbstractString)
    occursin("->", pattern) || return tokenize_generic(pattern)
    lhs, rhs = strip.(split(pattern, "->"; limit = 2))
    occursin("->", rhs) && throw(ArgumentError("multiple \"->\" in pattern"))
    return tokenize_side(lhs) --> tokenize_side(rhs)
end

function tokenize_side(side::AbstractString)

    function parse_token!(buf::IOBuffer, tokens::Vector)
        if position(buf) > 0
            s = String(take!(buf))
            token = tryparse(Int, s)
            isnothing(token) && (token = Symbol(s))
            push!(tokens, token)
        end
    end

    tokens = Any[]
    buf = IOBuffer()
    stack = Vector{Any}[]
    i = firstindex(side)
    while i <= lastindex(side)
        c = side[i]
        if c == ' '
            parse_token!(buf, tokens)
            i += 1
        elseif c == '('
            parse_token!(buf, tokens)
            push!(stack, tokens)
            tokens = Any[]
            i += 1
        elseif c == ')'
            parse_token!(buf, tokens)
            isempty(stack) && throw(ArgumentError("unmatched ')' in pattern"))
            sub = tokens
            tokens = pop!(stack)
            push!(tokens, Tuple(sub))
            i += 1
        elseif c == '.'
            # Expect literal "..."
            (i + 2 ≤ lastindex(side) && side[i:i+2] == "...") ||
                throw(ArgumentError("single '.' not allowed in pattern"))
            push!(tokens, ..)
            i += 3
        else
            write(buf, c)
            i += 1
        end
    end
    parse_token!(buf, tokens)

    !isempty(stack) && throw(ArgumentError("unmatched '(' in pattern"))
    return Tuple(tokens)
end

const SpecialToken = Dict(:* => (*), :_ => (-))
get_special_token(symbol) = get(SpecialToken, symbol, symbol)
mapfilter(f, pred, xs) = map(f, filter(pred, xs))
tokenize_generic(pattern) = Tuple(mapfilter(get_special_token ∘ Symbol, !isempty, split(pattern, ' ')))

"""
    @einops_str -> Union{ArrowPattern,Tuple}

For parity with Python implementation.

# Examples

```jldoctest
julia> einops"a 1 b c -> (c b) a"
(:a, 1, :b, :c) --> ((:c, :b), :a)

julia> einops"embed token (head batch) -> (embed head) token batch"
(:embed, :token, (:head, :batch)) --> ((:embed, :head), :token, :batch)

julia> einops"i j * k" # for pack/unpack
(:i, :j, *, :k)

julia> einops"a b _ d" # for parse_shape
(:a, :b, -, :d)
```
"""
macro einops_str(pattern)
    return parse_pattern(pattern)
end
