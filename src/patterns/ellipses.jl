function ellipsis_replacement(side, N)
    count(==(..), side) <= 1 || throw("At most one ellipsis is allowed: $side")
    return anonymous_symbols(:__ellipsis, N - length(side) + 1)
end

function replace_ellipses(left, right, N)
    if (..) ∈ flatten(left)
        (..) ∉ left && throw("Ellipsis is not allowed to be nested on left side.")
    else
        (..) ∈ flatten(right) && throw("Ellipsis found on right side but not on left side: $(left --> right)")
        return :($(left --> right))
    end 
    count(==(..), flatten(left)) == 1 || throw("Only one ellipsis is allowed on left side: $(left --> right)")
    replacement = ellipsis_replacement(left, N)
    new_left = insertat(left, only(findfirst(==(..), left)), replacement)
    new_right = if (..) in flatten(right)
        if (..) in right
            insertat(right, only(findfirst(==(..), right)), replacement)
        else
            # ellipsis is in a nested tuple
            i = findfirst(t -> t isa Tuple && (..) in t, right)
            t = right[i]
            j = findfirst(==(..), t)
            insertat(right, i, (insertat(t, j, replacement),))
        end
    else
        right
    end
    new_left, new_right
end

function replace_ellipses_parse_shape(pattern, N)
    (..) in pattern || return pattern
    count(==(..), pattern) == 1 || throw("Only one ellipsis is allowed: $pattern")
    ellipsis_index = findfirst(==(..), pattern)
    new_pattern = insertat(pattern, ellipsis_index, ntuple(Returns(-), N - length(pattern) + 1))
    return new_pattern
end
