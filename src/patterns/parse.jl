function parse_pattern(pattern::AbstractString)
    occursin("->", pattern) || return tokenize_generic(pattern)
    lhs, rhs = strip.(split(pattern, "->"; limit = 2))
    occursin("->", rhs) && throw(ArgumentError("multiple \"->\" in pattern"))
    occursin(",", rhs) && throw(ArgumentError("Commas are only allowed on the input side of a pattern"))
    return tokenize_side(lhs) --> tokenize_side(rhs)
end

function tokenize_side(side::AbstractString)
    # Check if there are any commas
    if occursin(",", side)
        # Split by comma and process each part separately
        parts = split(side, ","; keepempty=true)
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
            i = nextind(side, i)
        elseif c == '('
            parse_token!(buf, tokens)
            push!(stack, tokens)
            tokens = Any[]
            i = nextind(side, i)
        elseif c == ')'
            parse_token!(buf, tokens)
            isempty(stack) && throw(ArgumentError("unmatched ')' in pattern"))
            sub = tokens
            tokens = pop!(stack)
            push!(tokens, Tuple(sub))
            i = nextind(side, i)
        elseif c == '.'
            # Expect literal "..."
            i2 = nextind(side, i)
            i3 = i2 ≤ lastindex(side) ? nextind(side, i2) : i2
            (i3 ≤ lastindex(side) && side[i2] == '.' && side[i3] == '.') ||
                throw(ArgumentError("single '.' not allowed in pattern"))
            push!(tokens, ..)
            i = nextind(side, i3)
        else
            write(buf, c)
            i = nextind(side, i)
        end
    end
    parse_token!(buf, tokens)

    !isempty(stack) && throw(ArgumentError("unmatched '(' in pattern"))
    return Tuple(tokens)
end

const SpecialToken = Dict(:* => (*), :_ => (-), :... => (..))
get_special_token(symbol) = get(SpecialToken, symbol, symbol)

# Recursively map special tokens inside possibly nested tuples
map_special_tokens(x) = x
map_special_tokens(x::Symbol) = get_special_token(x)
map_special_tokens(x::Tuple) = Tuple(map(map_special_tokens, x))

# Generic patterns (no arrow) should still honor parentheses and commas,
# producing nested tuples where appropriate. Reuse the side tokenizer and
# then map special tokens like `_`, `*`, and `...`.
tokenize_generic(pattern) = Val(map_special_tokens(tokenize_side(pattern)))

# === Interpolation ===
#
# `$x` in a pattern is an axis of length `x` and nothing more, as if the value were written
# there: like an integer other than 1, it can be reduced or created but not carried through,
# and each occurrence is an axis of its own. The value stays out of the pattern's type: the
# axis gets a hidden name, and the value goes along as that name's length, the way a keyword
# would.

# A raw string's literal parts and the expressions it interpolates with `$`, the way Julia's
# parser splits a string literal in code (which hands the macros the same parts).
function interpolation_parts(str::AbstractString)
    parts = Any[]
    buf = IOBuffer()
    i = firstindex(str)
    while i <= lastindex(str)
        if str[i] == '$'
            j = nextind(str, i)
            j <= lastindex(str) && !isspace(str[j]) ||
                throw(ArgumentError("`\$` in a pattern must be followed by a name or a parenthesized expression: $str"))
            push!(parts, String(take!(buf)))
            ex, i = Meta.parseatom(str, j)
            push!(parts, ex)
        else
            write(buf, str[i])
            i = nextind(str, i)
        end
    end
    push!(parts, String(take!(buf)))
    return parts
end

interpolation_placeholder(k) = Symbol("__interp_", k, "__")

# The hidden name keeps the source, `$x`, for messages, and a count that makes it unique.
interpolated_name(ex, k) = Symbol("__\$", ex isa Symbol ? ex : "($ex)", "#", k)
is_interpolated(name::Symbol) = startswith(String(name), "__\$")
interpolated_source(name::Symbol) = (s = String(name); s[3:prevind(s, findlast('#', s))])

# Returns the pattern, and the name and expression of each interpolation.
function parse_interpolated_pattern(parts)
    names, exprs = Symbol[], Any[]
    str = sprint() do io
        for part in parts
            part isa AbstractString && (print(io, part); continue)
            push!(exprs, part)
            push!(names, interpolated_name(part, length(exprs)))
            print(io, ' ', interpolation_placeholder(length(exprs)), ' ')
        end
    end
    pattern = parse_pattern(str)
    pattern isa ArrowPattern || throw(ArgumentError("Interpolation is only supported in patterns with `->`"))
    renames = Dict(interpolation_placeholder(k) => name for (k, name) in enumerate(names))
    L, R = pattern
    return rename_axes(renames, L) --> rename_axes(renames, R), names, exprs
end

rename_axes(renames, side::Tuple) = map(el -> el isa Tuple ? rename_axes(renames, el) : get(renames, el, el), side)
