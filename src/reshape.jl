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
@generated function Base.reshape(x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
    left, right = replace_ellipses(L, R, N)
    left_symbols = extract(Symbol, left)
    right_symbols = extract(Symbol, right)
    left_symbols == right_symbols || throw(ArgumentError("reshape requires symbols in same order on both sides. Got $left_symbols vs $right_symbols. Use rearrange for permutations."))
    shape_in = get_shape_in(N, left, pairs_type_to_names(context))
    shape_out = get_shape_out(right)
    quote
        context = NamedTuple(context)
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end
