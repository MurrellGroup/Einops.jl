function ellipsis_replacement(side, N)
    count(==(..), side) <= 1 || throw("At most one ellipsis is allowed: $pattern")
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

@generated function replace_ellipses_einsum(::ArrowPattern{left,right}, ::Val{Ns}) where {left,right,Ns}
    pattern = left --> right
    (..) ∉ flatten(left) && (..) ∉ flatten(right) && return :($pattern)
    (..) ∉ flatten(left) && (..) in flatten(right) && throw("Found ellipsis on right side but not left side: $pattern")
    inds = findall(t -> (..) in t, left)
    replacements = [ellipsis_replacement(side, N) for (side, N) in zip(left[inds], Ns[inds])]
    allequal(replacements) || throw("Ellipses used for different number of dimensions")
    replacement = only(unique(replacements))
    new_left = Tuple((..) in t ? insertat(t, findfirst(==(..), t), replacement) : t for t in left)
    new_right = (..) in right ? insertat(right, findfirst(==(..), right), replacement) : right
    :($(new_left --> new_right))
end
