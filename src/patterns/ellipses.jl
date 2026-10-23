function ellipsis_replacement(side, N)
    count(==(Ellipsis()), side) <= 1 || throw("At most one ellipsis is allowed: $side")
    return anonymous_symbols(:__ellipsis, N - length(side) + 1)
end

function replace_ellipses_left(left, N)
    n = count(==(Ellipsis()), left)
    n == 0 && return left
    n == 1 || throw("Only one ellipsis is allowed on left side: $left")
    replacement = ellipsis_replacement(left, N)
    new_left = insertat(left, only(findfirst(==(Ellipsis()), left)), replacement)
    return new_left
end

function replace_ellipses_right(left, right, N)
    replacement = ellipsis_replacement(left, N)
    if Ellipsis() in flatten(right)
        if Ellipsis() in right
            insertat(right, only(findfirst(==(Ellipsis()), right)), replacement)
        else
            # ellipsis is in a nested tuple
            i = findfirst(t -> t isa Tuple && Ellipsis() in t, right)
            t = right[i]
            j = findfirst(==(Ellipsis()), t)
            insertat(right, i, (insertat(t, j, replacement),))
        end
    else
        right
    end
end

function replace_ellipses(left, right, N)
    left isa List && throw_operand_list(left)
    if Ellipsis() ∈ flatten(left)
        Ellipsis() ∉ left && throw("Ellipsis is not allowed to be nested on left side.")
    else
        Ellipsis() ∈ flatten(right) && throw("Ellipsis found on right side but not on left side: $(ArrowPattern{left,right}())")
        return left, right
    end
    new_left = replace_ellipses_left(left, N)
    new_right = replace_ellipses_right(left, right, N)
    new_left, new_right
end

function replace_ellipses_parse_shape(pattern, N)
    Ellipsis() in pattern || return pattern
    count(==(Ellipsis()), pattern) == 1 || throw("Only one ellipsis is allowed: $pattern")
    ellipsis_index = findfirst(==(Ellipsis()), pattern)
    new_pattern = insertat(pattern, ellipsis_index, ntuple(Returns(-), N - length(pattern) + 1))
    return new_pattern
end

function replace_ellipses_collapse(right, N)
    Ellipsis() ∈ flatten(right) || return right
    count(==(Ellipsis()), flatten(right)) == 1 || throw("Only one ellipsis is allowed: $right")
    symbol_count = length(extract(Symbol, right))
    remaining = N - symbol_count
    remaining >= 0 || throw("Ellipsis represents a negative number of dimensions: $right with N=$N")
    replacement = anonymous_symbols(:__ellipsis, remaining)
    if Ellipsis() in right
        return insertat(right, only(findfirst(==(Ellipsis()), right)), replacement)
    else
        # ellipsis is in a nested tuple
        i = findfirst(t -> t isa Tuple && Ellipsis() in t, right)
        t = right[i]
        j = findfirst(==(Ellipsis()), t)
        return insertat(right, i, (insertat(t, j, replacement),))
    end
end

# `left` holds one side per operand, and `Ns` their ranks. Returns the new `(left, right)`.
@generated function replace_ellipses_einsum(::Val{left}, ::Val{right}, ::Val{Ns}) where {left,right,Ns}
    Ellipsis() ∉ flatten(left) && Ellipsis() ∉ flatten(right) && return QuoteNode((left, right))
    Ellipsis() ∉ flatten(left) && Ellipsis() ∈ flatten(right) && throw("Found ellipsis on right side but not left side: $(pattern_string(left, right))")

    # Arrays whose index-tuples contain an ellipsis anywhere (possibly nested)
    inds = findall(t -> Ellipsis() ∈ flatten(t), left)
    replacements = [ellipsis_replacement(side, N) for (side, N) in zip(left[inds], Ns[inds])]
    allequal(replacements) || throw("Ellipses used for different number of dimensions")
    replacement = only(unique(replacements))

    # Recursively replace the first ellipsis occurrence in a (possibly nested) tuple
    function replace_in_tuple(t)
        if Ellipsis() in t
            return insertat(t, only(findfirst(==(Ellipsis()), t)), replacement)
        end
        for (i, el) in pairs(t)
            if el isa Tuple && Ellipsis() ∈ flatten(el)
                return insertat(t, i, (replace_in_tuple(el),))
            end
        end
        return t
    end

    new_left = Tuple(replace_in_tuple(t) for t in left)
    new_right = replace_in_tuple(right)
    return QuoteNode((new_left, new_right))
end
