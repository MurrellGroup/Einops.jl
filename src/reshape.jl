"""
    reshape(x::AbstractArray, left --> right; context...)

Reshape `x` according to the pattern `left --> right`.

Unlike [`rearrange`](@ref), this does not permute dimensions - symbols must appear
in the same order on both sides.

# Examples

```jldoctest
julia> x = rand(1, 6, 2, 3);

julia> reshape(x, einops"1 (a b) ... -> a (b ...)"; a=2) |> size
(2, 18)

julia> x = rand(2, 3, 4);

julia> reshape(x, (:a, :b, :c) --> ((:a, :b), :c)) |> size
(6, 4)
```
"""
@generated function Base.reshape(x, ::ArrowPattern{L,R}; context...) where {L,R}
    N = ndims(x)
    left, right = replace_ellipses(L, R, N)
    quote
        context = NamedTuple(context)
        $(reshape_body(N, left, right, pairs_type_to_names(context)))
    end
end

# See `rearrange_body` for the shared conventions.
function reshape_body(N, left, right, context_names)
    left_symbols = extract(Symbol, left)
    right_symbols = extract(Symbol, right)
    left_symbols == right_symbols || throw(ArgumentError("reshape requires symbols in same order on both sides. Got $left_symbols vs $right_symbols. Use rearrange for permutations."))
    shape_in = get_shape_in(N, left, context_names)
    shape_out = get_shape_out(right)
    body = Expr(:block)
    isnothing(shape_in) || push!(body.args, :(x = Rewrap.reshape(x, $shape_in)))
    isnothing(shape_out) || push!(body.args, :(x = Rewrap.reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end

# Ellipsis variant; see `rearrange_body_ellipsis`.
function reshape_body_ellipsis(L, R, context_names)
    left = replace_ellipsis_placeholder(L)
    right = replace_ellipsis_placeholder(R)
    m = ELLIPSIS_M
    left_symbols = extract(Symbol, left)
    right_symbols = extract(Symbol, right)
    left_symbols == right_symbols || throw(ArgumentError("reshape requires symbols in same order on both sides. Got $left_symbols vs $right_symbols. Use rearrange for permutations."))
    shape_in = get_shape_in(length(left), left, context_names)
    shape_out = get_shape_out(right)
    isnothing(shape_in) || expand_keep!(shape_in, findfirst(==(ELLIPSIS_PLACEHOLDER), left), m)
    isnothing(shape_out) || expand_shape_out!(shape_out, right, m)
    body = Expr(:block, ellipsis_m_binding(L))
    isnothing(shape_in) || push!(body.args, :(x = Rewrap.reshape(x, $shape_in)))
    isnothing(shape_out) || push!(body.args, :(x = Rewrap.reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end
