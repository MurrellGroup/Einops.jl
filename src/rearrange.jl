"""
    rearrange(x, left --> right; context...)

Rearrange the axes of `x` according to the pattern specified by `left --> right`.

Can always be expressed as a `reshape` + `permutedims` + `reshape`.

# Examples

```jldoctest
julia> x = rand(2,3,5);

julia> y = rearrange(x, einops"a b c -> c b a");

julia> size(y)
(5, 3, 2)

julia> y == permutedims(x, (3,2,1))
true

julia> z = rearrange(x, (:a, :b, :c) --> (:a, (:c, :b)));

julia> size(z)
(2, 15)

julia> z == reshape(permutedims(x, (1,3,2)), 2,5*3)
true
```
"""
@generated function rearrange(x, ::ArrowPattern{L,R}; context...) where {L,R}
    N = ndims(x)
    left, right = replace_ellipses(L, R, N)
    quote
        context = NamedTuple(context)
        $(rearrange_body(N, left, right, pairs_type_to_names(context)))
    end
end

# `rearrange`'s plan, factored out so `@rearrange` can splice it inline. `left`/`right`
# are ellipsis-resolved for `N`; the block operates on `x`/`context` and returns `x`.
function rearrange_body(N, left, right, context_names)
    shape_in = get_shape_in(N, left, context_names)
    permutation = get_permutation(extract(Symbol, left), extract(Symbol, right))
    shape_out = get_shape_out(right)
    body = Expr(:block)
    isnothing(shape_in) || push!(body.args, :(x = Rewrap.reshape(x, $shape_in)))
    permutation === ntuple(identity, length(permutation)) || push!(body.args, :(x = $(Rewrap.Permute(permutation))(x)))
    isnothing(shape_out) || push!(body.args, :(x = Rewrap.reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end

# Ellipsis variant: rank is unknown at expansion, so the ellipsis run folds to `Keep(m)`
# and the permutation expands via `Val(ndims(x))` (staging helpers in `utils.jl`).
function rearrange_body_ellipsis(L, R, context_names)
    left = replace_ellipsis_placeholder(L)
    right = replace_ellipsis_placeholder(R)
    m = ELLIPSIS_M
    shape_in = get_shape_in(length(left), left, context_names)
    permutation = get_permutation(extract(Symbol, left), extract(Symbol, right))
    shape_out = get_shape_out(right)
    pli = findfirst(==(ELLIPSIS_PLACEHOLDER), extract(Symbol, left))
    isnothing(shape_in) || expand_keep!(shape_in, findfirst(==(ELLIPSIS_PLACEHOLDER), left), m)
    perm = permutation === ntuple(identity, length(permutation)) ? nothing : permute_run_expr(permutation, pli)
    isnothing(shape_out) || expand_shape_out!(shape_out, right, m)
    body = Expr(:block, ellipsis_m_binding(L))
    isnothing(shape_in) || push!(body.args, :(x = Rewrap.reshape(x, $shape_in)))
    isnothing(perm) || push!(body.args, :(x = $perm))
    isnothing(shape_out) || push!(body.args, :(x = Rewrap.reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end

@generated function expand(x, ::Val{L}; context...) where {L}
    N = ndims(x)
    left = replace_ellipses_left(L, N)
    shape_in = get_shape_in(N, left, pairs_type_to_names(context); allow_repeats=true)
    body = Expr(:block, :(context = NamedTuple(context)))
    isnothing(shape_in) || push!(body.args, :(x = reshape(x, $shape_in)))
    push!(body.args, :(return x))
    return body
end

@generated function collapse(x, ::Val{R}; context...) where {R}
    N = ndims(x)
    right = replace_ellipses_collapse(R, N)
    shape_out = get_shape_out(right)
    body = Expr(:block, :(context = NamedTuple(context)))
    isnothing(shape_out) || push!(body.args, :(x = reshape(x, $shape_out)))
    push!(body.args, :(return x))
    return body
end