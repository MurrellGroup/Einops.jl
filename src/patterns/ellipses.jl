# An input side in a message: the left side, or one operand of several.
side_name(k, n) = n == 1 ? "on the left side" : "in operand $k"
input_name(k, n) = n == 1 ? "the input" : "operand $k"

# What every op asks of a pattern's ellipses: at most one to a side (to each operand of an
# einsum), outside any group on the left, and on the right only if also on the left.
function check_ellipses(inputs::Tuple, right)
    for (k, side) in enumerate(inputs)
        where = side_name(k, length(inputs))
        count(==(Ellipsis()), flatten(side)) <= 1 || throw(ArgumentError("`...` appears more than once $where"))
        Ellipsis() ∈ flatten(side) && Ellipsis() ∉ side && throw(ArgumentError("`...` can't be in a group $where"))
    end
    count(==(Ellipsis()), flatten(right)) <= 1 || throw(ArgumentError("`...` appears more than once on the right side"))
    Ellipsis() ∈ flatten(right) && all(side -> Ellipsis() ∉ side, inputs) && throw(ArgumentError("`...` is only on the right side"))
    return nothing
end

# The rank a side asks of its array: its length, or at least what is left beside an ellipsis.
function check_rank(side, N, input="the input")
    ellipsis = Ellipsis() in side
    n = length(side) - ellipsis
    (ellipsis ? N >= n : N == n) || throw_rank(N, n, ellipsis, input)
    return nothing
end

@non_differentiable check_rank(::Any...)

ellipsis_replacement(side, N) = anonymous_symbols(:__ellipsis, N - length(side) + 1)

function replace_ellipses_left(left, N)
    Ellipsis() in left || return left
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
    check_ellipses((left,), right)
    check_rank(left, N)
    Ellipsis() ∈ left || return left, right
    new_left = replace_ellipses_left(left, N)
    new_right = replace_ellipses_right(left, right, N)
    new_left, new_right
end

function replace_ellipses_parse_shape(pattern, N)
    Ellipsis() in pattern || return pattern
    ellipsis_index = findfirst(==(Ellipsis()), pattern)
    new_pattern = insertat(pattern, ellipsis_index, ntuple(Returns(-), N - length(pattern) + 1))
    return new_pattern
end

function replace_ellipses_collapse(right, N)
    Ellipsis() ∈ flatten(right) || return right
    remaining = N - length(extract(Symbol, right))
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
# `check_einsum` has made sure every ellipsis spans as many axes.
@generated function replace_ellipses_einsum(::Val{left}, ::Val{right}, ::Val{Ns}) where {left,right,Ns}
    Ellipsis() ∉ flatten(left) && return QuoteNode((left, right))
    k = findfirst(side -> Ellipsis() ∈ side, left)
    replacement = ellipsis_replacement(left[k], Ns[k])

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
