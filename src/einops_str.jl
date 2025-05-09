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
    buf    = IOBuffer()
    stack  = Vector{Any}[]
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
    @einops_str -> Union{Pattern,Tuple}

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
