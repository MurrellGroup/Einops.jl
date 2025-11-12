function parse_pattern(pattern::AbstractString)
    occursin("->", pattern) || return tokenize_generic(pattern)
    lhs, rhs = strip.(split(pattern, "->"; limit = 2))
    occursin("->", rhs) && throw(ArgumentError("multiple \"->\" in pattern"))
    return tokenize_side(lhs) --> tokenize_side(rhs)
end

function tokenize_side(side::AbstractString)
    # Check if there are any commas
    if occursin(",", side)
        # Split by comma and process each part separately
        parts = split(side, ",")
        return Tuple(tokenize_side(strip(part)) for part in parts)
    end
    
    # Original tokenization logic for parts without commas
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

const SpecialToken = Dict(:* => (*), :_ => (-), :... => (..))
get_special_token(symbol) = get(SpecialToken, symbol, symbol)
mapfilter(f, pred, xs) = map(f, filter(pred, xs))

# Recursively map special tokens inside possibly nested tuples
map_special_tokens(x) = x
map_special_tokens(x::Symbol) = get_special_token(x)
map_special_tokens(x::Tuple) = Tuple(map(map_special_tokens, x))

# Generic patterns (no arrow) should still honor parentheses and commas,
# producing nested tuples where appropriate. Reuse the side tokenizer and
# then map special tokens like `_`, `*`, and `...`.
tokenize_generic(pattern) = Val(map_special_tokens(tokenize_side(pattern)))
