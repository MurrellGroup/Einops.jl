"""
    ..

For patterns constructed with `-->`, one can use `..` (from EllipsisNotation.jl) to represent multiple dimensions.

# Examples

```jldoctest
julia> rearrange(rand(2,3,4), (:a, ..) --> (.., :a)) |> size
(3, 4, 2)
```
"""
..

function ellipsis_replacement(side, N)
    count(==(..), side) <= 1 || throw("At most one ellipsis is allowed: $side")
    return anonymous_symbols(:__ellipsis, N - length(side) + 1)
end

function replace_ellipses_left(left, N)
    n = count(==(..), left)
    n == 0 && return left
    n == 1 || throw("Only one ellipsis is allowed on left side: $left")
    replacement = ellipsis_replacement(left, N)
    new_left = insertat(left, only(findfirst(==(..), left)), replacement)
    return new_left
end

function replace_ellipses_right(left, right, N)
    replacement = ellipsis_replacement(left, N)
    if (..) in flatten(right)
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
end

function replace_ellipses(left, right, N)
    if (..) ∈ flatten(left)
        (..) ∉ left && throw("Ellipsis is not allowed to be nested on left side.")
    else
        (..) ∈ flatten(right) && throw("Ellipsis found on right side but not on left side: $(left --> right)")
        return :($(left --> right))
    end
    new_left = replace_ellipses_left(left, N)
    new_right = replace_ellipses_right(left, right, N)
    new_left, new_right
end

function replace_ellipses_parse_shape(pattern, N)
    (..) in pattern || return pattern
    count(==(..), pattern) == 1 || throw("Only one ellipsis is allowed: $pattern")
    ellipsis_index = findfirst(==(..), pattern)
    new_pattern = insertat(pattern, ellipsis_index, ntuple(Returns(-), N - length(pattern) + 1))
    return new_pattern
end

function replace_ellipses_collapse(right, N)
    (..) ∈ flatten(right) || return right
    count(==(..), flatten(right)) == 1 || throw("Only one ellipsis is allowed: $right")
    symbol_count = length(extract(Symbol, right))
    remaining = N - symbol_count
    remaining >= 0 || throw("Ellipsis represents a negative number of dimensions: $right with N=$N")
    replacement = anonymous_symbols(:__ellipsis, remaining)
    if (..) in right
        return insertat(right, only(findfirst(==(..), right)), replacement)
    else
        # ellipsis is in a nested tuple
        i = findfirst(t -> t isa Tuple && (..) in t, right)
        t = right[i]
        j = findfirst(==(..), t)
        return insertat(right, i, (insertat(t, j, replacement),))
    end
end

@generated function replace_ellipses_einsum(::ArrowPattern{left,right}, ::Val{Ns}) where {left,right,Ns}
    pattern = left --> right
    (..) ∉ flatten(left) && (..) ∉ flatten(right) && return :($pattern)
    (..) ∉ flatten(left) && (..) ∈ flatten(right) && throw("Found ellipsis on right side but not left side: $pattern")

    # Arrays whose index-tuples contain an ellipsis anywhere (possibly nested)
    inds = findall(t -> (..) ∈ flatten(t), left)
    replacements = [ellipsis_replacement(side, N) for (side, N) in zip(left[inds], Ns[inds])]
    allequal(replacements) || throw("Ellipses used for different number of dimensions")
    replacement = only(unique(replacements))

    # Recursively replace the first ellipsis occurrence in a (possibly nested) tuple
    function replace_in_tuple(t)
        if (..) in t
            return insertat(t, only(findfirst(==(..), t)), replacement)
        end
        for (i, el) in pairs(t)
            if el isa Tuple && (..) ∈ flatten(el)
                return insertat(t, i, (replace_in_tuple(el),))
            end
        end
        return t
    end

    new_left = Tuple(replace_in_tuple(t) for t in left)
    new_right = replace_in_tuple(right)
    :($(new_left --> new_right))
end
