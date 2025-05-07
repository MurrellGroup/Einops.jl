function parse_pattern(pattern::AbstractString)
    occursin("->", pattern) ||
        throw(ArgumentError("pattern must contain \"->\" (got \"$pattern\")"))

    lhs, rhs = strip.(split(pattern, "->"; limit = 2))
    lhs_axes = tokenise_side(lhs)
    rhs_axes = tokenise_side(rhs)
    return Tuple(lhs_axes) --> Tuple(rhs_axes)
end

function tokenise_side(side::AbstractString)
    tokens = Any[]
    stack  = Vector{Any}[]
    buf    = IOBuffer()
    i = firstindex(side)

    while i <= lastindex(side)
        c = side[i]

        if c == ' '
            if position(buf) > 0
                s = String(take!(buf))
                token = tryparse(Int, s)
                isnothing(token) && (token = Symbol(s))
                push!(tokens, token)
            end
            i += 1

        elseif c == '('
            if position(buf) > 0
                s = String(take!(buf))
                token = tryparse(Int, s)
                isnothing(token) && (token = Symbol(s))
                push!(tokens, token)
            end
            push!(stack, tokens)
            tokens = Any[]
            i += 1

        elseif c == ')'
            if position(buf) > 0
                s = String(take!(buf))
                token = tryparse(Int, s)
                isnothing(token) && (token = Symbol(s))
                push!(tokens, token)
            end
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

    if position(buf) > 0
        s = String(take!(buf))
        token = tryparse(Int, s)
        isnothing(token) && (token = Symbol(s))
        push!(tokens, token)
    end
    !isempty(stack) && throw(ArgumentError("unmatched '(' in pattern"))
    return Tuple(tokens)
end

macro einops_str(pattern)
    return parse_pattern(pattern)
end
